from __future__ import annotations

import random
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from models import (
    ChaosAgentAction,
    ChaosAgentObservation,
    ChaosAgentState,
    DifficultyTier,
    Scenario,
    SubmitAnswerAction,
    ToolDesc,
    ToolResult,
)
from server.curriculum import CurriculumController
from server.fault_injector import FaultInjector
from server.grader import Grader
from server.repeat_tracker import RepeatTracker
from server.scenario_repository import ScenarioRepository
from server.tool_router import ToolRouter
from server.tools.live_tools import LiveTools
from server.tools.registry import get_all_tools, is_known_tool, validate_tool_registry


class ChaosAgentEnvironment(Environment[ChaosAgentAction, ChaosAgentObservation, ChaosAgentState]):
    """OpenEnv-compatible environment for unreliable-tool resilience tasks."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    DEFAULT_MAX_STEPS = 15

    def __init__(self, scenario_repository: ScenarioRepository | None = None):
        super().__init__()
        validate_tool_registry()
        self.grader = Grader()
        self.tool_router = ToolRouter()
        self.scenario_repository = scenario_repository or ScenarioRepository.default()
        self.curriculum = CurriculumController()

        self._rng = random.Random()
        self._fault_injector = FaultInjector(self._rng)
        self._live_tools = LiveTools()
        self._repeat_tracker = RepeatTracker()
        self._scenario = self.scenario_repository.all()[0]
        self._state = ChaosAgentState(
            episode_id=str(uuid4()),
            scenario_id=self._scenario.id,
            task_question=self._scenario.question,
            difficulty_tier=self._scenario.difficulty,
            max_steps=self.DEFAULT_MAX_STEPS,
            curriculum_tier=self.curriculum.current_tier,
        )
        self._last_correctness = 0.0

    def reset(
        self,
        seed: int | Scenario | None = None,
        episode_id: str | None = None,
        scenario_id: str | None = None,
        difficulty: DifficultyTier | str | None = None,
        scenario: Scenario | dict[str, Any] | None = None,
        max_steps: int | None = None,
        **_: Any,
    ) -> ChaosAgentObservation:
        """Reset a session and return the initial tool catalog observation."""
        if isinstance(seed, Scenario):
            scenario = seed
            seed = None

        self._rng = random.Random(seed)
        self._fault_injector = FaultInjector(self._rng)
        self._live_tools = LiveTools()
        self._repeat_tracker = RepeatTracker()

        selected_scenario = self._select_scenario(
            scenario=scenario, scenario_id=scenario_id, difficulty=difficulty
        )
        self._scenario = selected_scenario
        max_step_count = max_steps or self.DEFAULT_MAX_STEPS

        self._state = ChaosAgentState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            scenario_id=selected_scenario.id,
            task_question=selected_scenario.question,
            difficulty_tier=selected_scenario.difficulty,
            max_steps=max_step_count,
            tools_called=[],
            faults_injected=0,
            is_done=False,
            cumulative_reward=0.0,
            curriculum_tier=self.curriculum.current_tier,
            episodes_completed=self.curriculum.episodes_in_tier,
            submitted_answer=None,
        )
        self._last_correctness = 0.0

        return ChaosAgentObservation(
            task_question=selected_scenario.question,
            scenario_id=selected_scenario.id,
            tool_result=None,
            available_tools=[ToolDesc(**tool) for tool in get_all_tools()],
            warning=None,
            steps_taken=0,
            max_steps=max_step_count,
            done=False,
            reward=0.0,
            metadata={
                "difficulty": selected_scenario.difficulty.value,
                "scenario_tags": selected_scenario.tags,
                "min_tools_needed": selected_scenario.min_tools_needed,
            },
        )

    def step(
        self,
        action: ChaosAgentAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> ChaosAgentObservation:
        """Execute a tool call or final answer submission."""
        del timeout_s
        if self._state.is_done:
            return self._observation(
                tool_result=ToolResult(
                    tool_name="environment",
                    error="Episode is already done. Call reset() to start a new episode.",
                ),
                reward=0.0,
                done=True,
            )

        self._state.step_count += 1

        if isinstance(action, SubmitAnswerAction) or action.is_submit:
            return self._handle_submit(action)
        return self._handle_tool_call(action)

    @property
    def state(self) -> ChaosAgentState:
        return self._state

    def _select_scenario(
        self,
        *,
        scenario: Scenario | dict[str, Any] | None,
        scenario_id: str | None,
        difficulty: DifficultyTier | str | None,
    ) -> Scenario:
        if scenario is not None:
            return scenario if isinstance(scenario, Scenario) else Scenario.model_validate(scenario)
        if scenario_id:
            return self.scenario_repository.get(scenario_id)
        selected_difficulty = difficulty or self.curriculum.current_tier
        return self.scenario_repository.choose(rng=self._rng, difficulty=selected_difficulty)

    def _handle_submit(self, action: ChaosAgentAction) -> ChaosAgentObservation:
        answer = action.answer or ""
        correctness = self.grader.grade(answer, self._scenario)
        reward = self._compute_reward(correctness)

        self._state.is_done = True
        self._state.submitted_answer = answer
        self._state.cumulative_reward = reward
        self._last_correctness = correctness
        self.curriculum.record_episode(correctness)

        return self._observation(
            tool_result=ToolResult(
                tool_name="submit_answer",
                message="Answer submitted successfully.",
            ),
            reward=reward,
            done=True,
            metadata={"correctness": correctness, "reasoning": action.reasoning},
        )

    def _handle_tool_call(self, action: ChaosAgentAction) -> ChaosAgentObservation:
        tool_name = action.tool_name or ""
        arguments = action.arguments

        if not is_known_tool(tool_name):
            return self._observation(
                tool_result=ToolResult(tool_name=tool_name or "unknown", error="Unknown tool"),
                reward=0.0,
                done=False,
            )

        self._state.tools_called.append(tool_name)
        warning = self._repeat_tracker.log_call(tool_name, arguments)

        clean_result = self._get_clean_tool_result(tool_name, arguments)
        final_result, injected, fault_mode = self._fault_injector.inject_if_needed(
            tool_name, clean_result, self._scenario.difficulty
        )
        if injected:
            self._state.faults_injected += 1

        tool_result = self._to_tool_result(
            tool_name=tool_name,
            result=final_result,
            fault_injected=injected,
            fault_mode=fault_mode.value if fault_mode else None,
        )

        if self._state.step_count >= self._state.max_steps:
            self._state.is_done = True
            self._state.cumulative_reward = -0.5
            return self._observation(
                tool_result=tool_result,
                reward=-0.5,
                done=True,
                warning=warning or "Maximum step count reached before answer submission.",
            )

        return self._observation(
            tool_result=tool_result,
            reward=0.0,
            done=False,
            warning=warning,
        )

    def _get_clean_tool_result(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        routed_result = self.tool_router.route(tool_name, arguments, self._scenario)
        if routed_result.get("_directive") == "always_live":
            return self._live_tools.handle(tool_name, arguments)
        return routed_result

    def _to_tool_result(
        self,
        *,
        tool_name: str,
        result: dict[str, Any],
        fault_injected: bool,
        fault_mode: str | None,
    ) -> ToolResult:
        error = result.get("error")
        message = result.get("message") or result.get("warning")
        if error is not None:
            payload = None
        elif "result" in result and len(result) == 1:
            payload = result["result"]
        else:
            payload = result

        return ToolResult(
            tool_name=tool_name,
            result=payload,
            error=str(error) if error is not None else None,
            message=str(message) if message is not None else None,
            fault_injected=fault_injected,
            fault_mode=fault_mode,
        )

    def _observation(
        self,
        *,
        tool_result: ToolResult,
        reward: float,
        done: bool,
        warning: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChaosAgentObservation:
        base_metadata: dict[str, Any] = {
            "faults_injected": self._state.faults_injected,
            "tools_called": list(self._state.tools_called),
            "difficulty": self._scenario.difficulty.value,
        }
        if metadata:
            base_metadata.update(metadata)

        return ChaosAgentObservation(
            task_question=self._scenario.question,
            scenario_id=self._scenario.id,
            tool_result=tool_result,
            available_tools=None,
            warning=warning,
            steps_taken=self._state.step_count,
            max_steps=self._state.max_steps,
            done=done,
            reward=reward,
            metadata=base_metadata,
        )

    def _compute_reward(self, correctness: float) -> float:
        resilience = 0.2 if correctness > 0.7 and self._state.faults_injected > 0 else 0.0

        step_ratio = self._state.step_count / max(self._scenario.min_tools_needed * 2, 1)
        if step_ratio <= 1.5:
            efficiency = 0.1
        elif step_ratio <= 3.0:
            efficiency = 0.0
        else:
            efficiency = -0.1

        repeat_penalty = max(-0.2, -0.05 * self._repeat_tracker.total_repeats)
        total = correctness * 0.7 + resilience + efficiency + repeat_penalty
        return max(-1.0, min(1.0, total))
