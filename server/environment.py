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
from server.task_workspace import TaskWorkspace
from server.tasks import BenchmarkTask, get_task
from server.tool_router import ToolRouter
from server.tools.live_tools import LiveTools
from server.tools.registry import get_all_tools, is_known_tool, validate_tool_registry


class ChaosAgentEnvironment(Environment[ChaosAgentAction, ChaosAgentObservation, ChaosAgentState]):
    """OpenEnv-compatible environment for unreliable-tool resilience tasks."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    DEFAULT_MAX_STEPS = 15
    RETRIEVAL_TOOLS = {
        "web_search",
        "fetch_url",
        "knowledge_base_lookup",
        "database_query",
        "document_search",
        "api_call",
        "read_file",
    }
    VERIFICATION_TOOLS = {
        "fact_check",
        "check_consistency",
        "compare_values",
        "validate_data",
        "validate_url",
        "hash_verify",
        "json_query",
        "text_extract",
    }
    COMPUTE_TOOLS = {
        "calculator",
        "python_execute",
        "data_transform",
    }
    ARTIFACT_TOOLS = {
        "scratchpad_write",
        "write_file",
        "database_insert",
        "create_report",
        "send_notification",
        "schedule_task",
        "update_ticket",
        "request_human_review",
    }

    def __init__(self, scenario_repository: ScenarioRepository | None = None):
        super().__init__()
        validate_tool_registry()
        self.grader = Grader()
        self.tool_router = ToolRouter()
        self.scenario_repository = scenario_repository or ScenarioRepository.default()
        self.curriculum = CurriculumController()

        self._rng = random.Random()
        self._fault_injector = FaultInjector(self._rng)
        self._scenario = self.scenario_repository.all()[0]
        self._task: BenchmarkTask = get_task(self._scenario.benchmark_task_id)
        self._workspace = TaskWorkspace(scenario=self._scenario, task=self._task)
        self._live_tools = LiveTools(initial_files=self._workspace.initial_files)
        self._repeat_tracker = RepeatTracker()
        self._state = ChaosAgentState(
            episode_id=str(uuid4()),
            task_id=self._task.id,
            task_name=self._task.name,
            task_description=self._task.description,
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
        task_id: str | None = None,
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
        self._repeat_tracker = RepeatTracker()

        selected_task = self._select_task(
            task_id=task_id, scenario=scenario, scenario_id=scenario_id
        )
        selected_scenario = self._select_scenario(
            scenario=scenario,
            scenario_id=scenario_id,
            difficulty=difficulty,
            task_id=selected_task.id,
        )
        self._scenario = selected_scenario
        self._task = selected_task
        self._reset_runtime_services()
        max_step_count = max_steps or selected_task.max_steps

        self._state = ChaosAgentState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=selected_task.id,
            task_name=selected_task.name,
            task_description=selected_task.description,
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
            tool_failures_observed=0,
            tool_successes_observed=0,
            repeat_calls=0,
            warning_events_observed=0,
            retrieval_successes=0,
            verification_calls=0,
            compute_calls=0,
            artifact_actions=0,
            recovery_switches=0,
            last_tool_name=None,
            last_tool_error=None,
            final_correctness=0.0,
            cross_validation_score=0.0,
            task_score=0.0,
        )
        self._last_correctness = 0.0

        return ChaosAgentObservation(
            task_id=selected_task.id,
            task_name=selected_task.name,
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
                "task_id": selected_task.id,
                "task_name": selected_task.name,
                "task_description": selected_task.description,
                "task_guidance": list(selected_task.guidance),
                "difficulty": selected_scenario.difficulty.value,
                "scenario_tags": selected_scenario.tags,
                "min_tools_needed": selected_scenario.min_tools_needed,
            },
        )

    def _select_task(
        self,
        *,
        task_id: str | None,
        scenario: Scenario | dict[str, Any] | None,
        scenario_id: str | None,
    ) -> BenchmarkTask:
        if scenario is not None:
            loaded = (
                scenario if isinstance(scenario, Scenario) else Scenario.model_validate(scenario)
            )
            return get_task(loaded.benchmark_task_id)
        if scenario_id:
            return get_task(self.scenario_repository.get(scenario_id).benchmark_task_id)
        return get_task(task_id or "task1")

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
        task_id: str,
    ) -> Scenario:
        if scenario is not None:
            return scenario if isinstance(scenario, Scenario) else Scenario.model_validate(scenario)
        if scenario_id:
            selected = self.scenario_repository.get(scenario_id)
            if selected.benchmark_task_id != task_id:
                raise ValueError(
                    f"scenario_id={scenario_id!r} does not belong to task_id={task_id!r}"
                )
            return selected
        selected_difficulty = difficulty
        return self.scenario_repository.choose(
            rng=self._rng,
            difficulty=selected_difficulty,
            benchmark_task_id=task_id,
        )

    def _handle_submit(self, action: ChaosAgentAction) -> ChaosAgentObservation:
        answer = action.answer or ""
        correctness = self.grader.grade(answer, self._scenario)
        reward = self._compute_reward(correctness)
        task_score = self.grader.grade_task(
            task=self._task,
            scenario=self._scenario,
            state=self._state,
            correctness=correctness,
            answered=True,
        )

        self._state.is_done = True
        self._state.submitted_answer = answer
        self._state.cumulative_reward = reward
        self._state.final_correctness = correctness
        self._state.cross_validation_score = self.grader.cross_validation_score(
            self._scenario, self._state
        )
        self._state.task_score = task_score
        self._last_correctness = correctness
        self.curriculum.record_episode(correctness)

        return self._observation(
            tool_result=ToolResult(
                tool_name="submit_answer",
                message="Answer submitted successfully.",
            ),
            reward=reward,
            done=True,
            metadata={
                "correctness": correctness,
                "reasoning": action.reasoning,
                "task_score": task_score,
                "cross_validation_score": self._state.cross_validation_score,
            },
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
        self._state.repeat_calls = self._repeat_tracker.total_repeats
        if self._state.last_tool_error and self._state.last_tool_name != tool_name:
            self._state.recovery_switches += 1

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
        if tool_result.error is None:
            self._state.tool_successes_observed += 1
            if tool_name in self.RETRIEVAL_TOOLS:
                self._state.retrieval_successes += 1
        else:
            self._state.tool_failures_observed += 1

        if tool_name in self.VERIFICATION_TOOLS:
            self._state.verification_calls += 1
        if tool_name in self.COMPUTE_TOOLS:
            self._state.compute_calls += 1
        if tool_name in self.ARTIFACT_TOOLS and tool_result.error is None:
            self._state.artifact_actions += 1
        if warning or tool_result.fault_injected or tool_result.fault_mode:
            self._state.warning_events_observed += 1

        self._state.last_tool_name = tool_name
        self._state.last_tool_error = tool_result.error

        if self._state.step_count >= self._state.max_steps:
            self._state.is_done = True
            self._state.cumulative_reward = -0.5
            self._state.cross_validation_score = self.grader.cross_validation_score(
                self._scenario, self._state
            )
            self._state.task_score = self.grader.grade_task(
                task=self._task,
                scenario=self._scenario,
                state=self._state,
                correctness=0.0,
                answered=False,
            )
            return self._observation(
                tool_result=tool_result,
                reward=-0.5,
                done=True,
                warning=warning or "Maximum step count reached before answer submission.",
                metadata={
                    "correctness": 0.0,
                    "task_score": self._state.task_score,
                    "cross_validation_score": self._state.cross_validation_score,
                },
            )

        return self._observation(
            tool_result=tool_result,
            reward=0.0,
            done=False,
            warning=warning,
        )

    def _get_clean_tool_result(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        routed_result = self.tool_router.route(
            tool_name,
            arguments,
            self._scenario,
            workspace=self._workspace,
        )
        if routed_result.get("_directive") == "always_live":
            return self._live_tools.handle(tool_name, arguments)
        return routed_result

    def close(self) -> None:
        self._workspace.close()

    def _reset_runtime_services(self) -> None:
        self._workspace.close()
        self._workspace = TaskWorkspace(scenario=self._scenario, task=self._task)
        self._live_tools = LiveTools(initial_files=self._workspace.initial_files)

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
            "task_id": self._task.id,
            "task_name": self._task.name,
            "faults_injected": self._state.faults_injected,
            "tools_called": list(self._state.tools_called),
            "difficulty": self._scenario.difficulty.value,
            "repeat_calls": self._state.repeat_calls,
            "tool_failures_observed": self._state.tool_failures_observed,
            "tool_successes_observed": self._state.tool_successes_observed,
            "warning_events_observed": self._state.warning_events_observed,
            "retrieval_successes": self._state.retrieval_successes,
            "verification_calls": self._state.verification_calls,
            "compute_calls": self._state.compute_calls,
            "artifact_actions": self._state.artifact_actions,
            "recovery_switches": self._state.recovery_switches,
        }
        if metadata:
            base_metadata.update(metadata)

        return ChaosAgentObservation(
            task_id=self._task.id,
            task_name=self._task.name,
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
