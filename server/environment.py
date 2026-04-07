import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ChaosAgentAction, ChaosAgentObservation, ChaosAgentState, Scenario, ToolResult, ToolDesc
from server.fault_injector import FaultInjector
from server.grader import Grader
from server.tool_router import ToolRouter
from server.tools.live_tools import LiveTools
from server.tools.registry import get_all_tools

class RepeatTracker:
    def __init__(self):
        self.total_repeats = 0
        self.history = []

    def log_call(self, tool_name: str, args: dict):
        call_sig = (tool_name, str(args))
        if call_sig in self.history:
            self.total_repeats += 1
        self.history.append(call_sig)

class ChaosAgentEnvironment:
    def __init__(self):
        self.grader = Grader()
        self.fault_injector = FaultInjector()
        self.tool_router = ToolRouter()
        
        self.scenario = None
        self.state = None
        self.live_tools_handler = None
        self.repeat_tracker = None
        
    def reset(self, scenario: Scenario) -> ChaosAgentObservation:
        self.scenario = scenario
        self.live_tools_handler = LiveTools()
        self.repeat_tracker = RepeatTracker()
        
        self.state = ChaosAgentState(
            scenario_id=scenario.id,
            task_question=scenario.question,
            difficulty_tier=scenario.difficulty,
            steps_taken=0,
            max_steps=15, # Hardcoded max steps for now
            tools_called=[],
            faults_injected=0,
            is_done=False,
            cumulative_reward=0.0,
            curriculum_tier=scenario.difficulty,
            episodes_completed=0
        )
        
        all_tools = [ToolDesc(**td) for td in get_all_tools()]
        
        return ChaosAgentObservation(
            task_question=self.scenario.question,
            tool_result=None,
            available_tools=all_tools,
            warning=None,
            steps_taken=0,
            max_steps=self.state.max_steps
        )

    def step(self, action: ChaosAgentAction) -> tuple[ChaosAgentObservation, float, bool, dict]:
        if self.state is None or self.state.is_done:
            raise ValueError("Episode is already done or not started.")
            
        self.state.steps_taken += 1
        
        if hasattr(action, 'answer'):
            self.state.is_done = True
            reward = self._compute_reward(action)
            self.state.cumulative_reward = reward
            
            obs = ChaosAgentObservation(
                task_question=self.scenario.question,
                tool_result=ToolResult(tool_name="submit_answer", message="Answer submitted successfully."),
                steps_taken=self.state.steps_taken,
                max_steps=self.state.max_steps
            )
            return obs, reward, True, {"action_type": "submit"}

        tool_name = getattr(action, 'tool_name', None)
        arguments = getattr(action, 'arguments', {})
        
        self.state.tools_called.append(tool_name)
        self.repeat_tracker.log_call(tool_name, arguments)
        
        raw_result = self.tool_router.route(tool_name, arguments, self.scenario)
        
        is_live = getattr(raw_result, "get", lambda k, d=None: None)("_directive") == "always_live" or tool_name in ["calculator", "python_execute", "scratchpad_write", "scratchpad_read"]
        
        if is_live:
            clean_result = self.live_tools_handler.handle(tool_name, arguments)
        else:
            clean_result = raw_result

        final_result, injected = self.fault_injector.inject_if_needed(tool_name, clean_result, self.state.difficulty_tier)
        if injected:
            self.state.faults_injected += 1
            
        tool_result = ToolResult(
            tool_name=tool_name,
            result=final_result if isinstance(final_result, dict) and not "error" in final_result else None,
            error=final_result.get("error") if isinstance(final_result, dict) and "error" in final_result else None,
            message=final_result.get("message") if isinstance(final_result, dict) and "message" in final_result else None
        )
        
        if self.state.steps_taken >= self.state.max_steps:
            self.state.is_done = True
            return self._build_obs(tool_result), -0.5, True, {"reason": "max_steps_reached"}
            
        return self._build_obs(tool_result), 0.0, False, {}
        
    def _build_obs(self, tool_result: ToolResult) -> ChaosAgentObservation:
        return ChaosAgentObservation(
            task_question=self.scenario.question,
            tool_result=tool_result,
            steps_taken=self.state.steps_taken,
            max_steps=self.state.max_steps
        )
        
    def _compute_reward(self, submit_action) -> float:
        correctness = self.grader.grade(submit_action.answer, self.scenario)
        
        resilience = 0.0
        if correctness > 0.7 and self.state.faults_injected > 0:
            resilience = 0.2
            
        step_ratio = self.state.steps_taken / max(self.scenario.min_tools_needed * 2, 1)
        if step_ratio <= 1.5:
            efficiency = 0.1
        elif step_ratio <= 3.0:
            efficiency = 0.0
        else:
            efficiency = -0.1
            
        repeat_penalty = max(-0.2, -0.05 * self.repeat_tracker.total_repeats)
        
        total = (correctness * 0.7) + resilience + efficiency + repeat_penalty
        return max(-1.0, min(1.0, total))
