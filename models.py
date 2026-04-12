from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JSONValue = dict[str, Any] | list[Any] | str | int | float | bool | None
JSONDict = dict[str, JSONValue]


class DifficultyTier(str, Enum):
    WARMUP = "warmup"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class FactType(str, Enum):
    NUMERIC = "numeric"
    TEXT = "text"
    BOOLEAN = "boolean"
    DATE = "date"


class Fact(BaseModel):
    """A single fact that must appear in the submitted answer."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1)
    value: JSONValue
    type: FactType
    tolerance: float = Field(default=0.0, ge=0.0)
    alternatives: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0.0)


class Scenario(BaseModel):
    """A deterministic task plus its pre-computed tool outputs and answer key."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    benchmark_task_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    answer: JSONDict
    required_facts: list[Fact] = Field(default_factory=list)
    tool_data: dict[str, JSONValue] = Field(default_factory=dict)
    difficulty: DifficultyTier
    min_tools_needed: int = Field(default=1, ge=1)
    tags: list[str] = Field(default_factory=list)
    cross_validation_tools: list[list[str]] = Field(default_factory=list)


class ToolDesc(BaseModel):
    """Description surfaced to the agent in the initial observation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    """Result returned from a tool call."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    result: JSONValue = None
    error: str | None = None
    message: str | None = None
    fault_injected: bool = False
    fault_mode: str | None = None


class ChaosAgentAction(Action):
    """Single OpenEnv action model for calling tools or submitting an answer."""

    type: Literal["call_tool", "submit_answer"] = Field(
        default="call_tool",
        description="Action discriminator: call_tool or submit_answer",
    )
    tool_name: str | None = Field(default=None, description="Tool to call when type is call_tool")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the selected tool"
    )
    answer: str | None = Field(default=None, description="Final answer when type is submit_answer")
    reasoning: str = Field(default="", description="Brief explanation for the submitted answer")

    @field_validator("tool_name")
    @classmethod
    def _strip_tool_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _validate_by_type(self) -> "ChaosAgentAction":
        if self.type == "call_tool" and not self.tool_name:
            raise ValueError("tool_name is required when type='call_tool'")
        if self.type == "submit_answer" and not self.answer:
            raise ValueError("answer is required when type='submit_answer'")
        return self

    @property
    def is_submit(self) -> bool:
        return self.type == "submit_answer"


class CallToolAction(ChaosAgentAction):
    """Convenience model for direct Python use and tests."""

    type: Literal["call_tool"] = "call_tool"
    tool_name: str
    answer: None = None


class SubmitAnswerAction(ChaosAgentAction):
    """Convenience model for direct Python use and tests."""

    type: Literal["submit_answer"] = "submit_answer"
    tool_name: None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    answer: str


class ChaosAgentObservation(Observation):
    """Observation returned after reset and each action."""

    task_id: str = ""
    task_name: str = ""
    task_question: str
    scenario_id: str
    tool_result: ToolResult | None = None
    available_tools: list[ToolDesc] | None = Field(
        default=None, description="Full tool catalog, included only after reset"
    )
    warning: str | None = None
    steps_taken: int = Field(default=0, ge=0)
    max_steps: int = Field(default=15, ge=1)


class ChaosAgentState(State):
    """Internal state exposed through OpenEnv's /state endpoint."""

    task_id: str = "task1"
    task_name: str = ""
    task_description: str = ""
    scenario_id: str = ""
    task_question: str = ""
    difficulty_tier: DifficultyTier = DifficultyTier.WARMUP
    max_steps: int = Field(default=15, ge=1)
    tools_called: list[str] = Field(default_factory=list)
    faults_injected: int = Field(default=0, ge=0)
    is_done: bool = False
    cumulative_reward: float = 0.0
    curriculum_tier: DifficultyTier = DifficultyTier.WARMUP
    episodes_completed: int = Field(default=0, ge=0)
    submitted_answer: str | None = None
    tool_failures_observed: int = Field(default=0, ge=0)
    tool_successes_observed: int = Field(default=0, ge=0)
    repeat_calls: int = Field(default=0, ge=0)
    warning_events_observed: int = Field(default=0, ge=0)
    retrieval_successes: int = Field(default=0, ge=0)
    verification_calls: int = Field(default=0, ge=0)
    compute_calls: int = Field(default=0, ge=0)
    artifact_actions: int = Field(default=0, ge=0)
    recovery_switches: int = Field(default=0, ge=0)
    last_tool_name: str | None = None
    last_tool_error: str | None = None
    final_correctness: float = Field(default=0.0, ge=0.0, le=1.0)
    cross_validation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    task_score: float = Field(default=0.0, ge=0.0, le=1.0)
