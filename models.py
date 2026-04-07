from typing import Any, Optional, Union
from pydantic import BaseModel, Field

# === Ground Truth System ===

class Fact(BaseModel):
    """A single fact that must appear in the answer."""
    key: str                         # e.g., "population"
    value: Any                       # e.g., 67390000
    type: str                        # "numeric", "text", "boolean", "date"
    tolerance: float = 0.0           # For numeric: relative tolerance (0.05 = 5%)
    alternatives: Optional[list[str]] = None   # Acceptable alternative text values

class Scenario(BaseModel):
    """A single evaluation scenario."""
    id: str                          # Unique identifier
    question: str                    # Natural language question for the agent
    answer: dict                     # Ground truth answer (key-value facts)
    required_facts: list[Fact]       # What the agent must include in the answer
    tool_data: dict[str, Any]        # Pre-computed clean outputs for each relevant tool
    difficulty: str                  # warmup / beginner / intermediate / expert
    min_tools_needed: int            # Minimum tools to answer correctly
    tags: list[str]                  # Domain tags for curriculum tracking
    cross_validation_tools: list[list[str]]  # Groups of tools that can verify each other


# === Actions ===
class CallToolAction(BaseModel):
    """Call any of the 30 available tools."""
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")

class SubmitAnswerAction(BaseModel):
    """Submit the final answer. Terminates the episode."""
    answer: str = Field(..., description="The agent's final answer")
    reasoning: str = Field(default="", description="How the agent arrived at this answer")

ChaosAgentAction = Union[CallToolAction, SubmitAnswerAction]

# === Observations ===
class ToolDesc(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON Schema

class ToolResult(BaseModel):
    tool_name: str
    result: Optional[dict] = None
    error: Optional[str] = None
    message: Optional[str] = None

class ChaosAgentObservation(BaseModel):
    task_question: str
    tool_result: Optional[ToolResult] = None
    available_tools: Optional[list[ToolDesc]] = None  # Only in first obs
    warning: Optional[str] = None
    steps_taken: int
    max_steps: int

# === State ===
class ChaosAgentState(BaseModel):
    scenario_id: str
    task_question: str
    difficulty_tier: str
    steps_taken: int
    max_steps: int
    tools_called: list[str]
    faults_injected: int
    is_done: bool
    cumulative_reward: float
    curriculum_tier: str
    episodes_completed: int
