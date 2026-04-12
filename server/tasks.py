from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkTask:
    """Named benchmark task exposed to inference and grading."""

    id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    target_tool_diversity: int
    ideal_steps: int
    target_recovery_switches: int
    target_verification_calls: int
    target_compute_calls: int
    target_artifact_actions: int
    guidance: tuple[str, ...]


TASKS: dict[str, BenchmarkTask] = {
    "task1": BenchmarkTask(
        id="task1",
        name="Single-Failure Recovery",
        difficulty="easy",
        description=(
            "Resolve a short factual query when one retrieval path may fail, "
            "timeout, or return incomplete data. The agent should recover by "
            "switching tools instead of repeating the same failing action."
        ),
        max_steps=6,
        target_tool_diversity=2,
        ideal_steps=4,
        target_recovery_switches=1,
        target_verification_calls=0,
        target_compute_calls=0,
        target_artifact_actions=0,
        guidance=(
            "Use an alternate retrieval tool when the first path fails or returns weak evidence.",
            "Avoid retry loops against the same failing tool call.",
        ),
    ),
    "task2": BenchmarkTask(
        id="task2",
        name="Cross-Tool Conflict Resolution",
        difficulty="medium",
        description=(
            "Solve a comparison or derivation task where evidence must be "
            "cross-checked across at least two independent tools before "
            "submitting the answer."
        ),
        max_steps=8,
        target_tool_diversity=3,
        ideal_steps=6,
        target_recovery_switches=1,
        target_verification_calls=1,
        target_compute_calls=1,
        target_artifact_actions=0,
        guidance=(
            "Cross-check at least two independent sources before answering.",
            "Use a compute or verification tool to reconcile conflicting evidence.",
        ),
    ),
    "task3": BenchmarkTask(
        id="task3",
        name="Cascading Failure Adaptation",
        difficulty="hard",
        description=(
            "Handle higher-stakes verification tasks under multiple faulty or "
            "conflicting tool outcomes. The agent must adapt, diversify tool "
            "usage, and avoid brittle retry loops."
        ),
        max_steps=10,
        target_tool_diversity=4,
        ideal_steps=7,
        target_recovery_switches=2,
        target_verification_calls=2,
        target_compute_calls=1,
        target_artifact_actions=1,
        guidance=(
            "Expect multiple faulty or stale results and adapt your strategy after failures.",
            "Keep explicit evidence through notes, files, or reports before submitting the answer.",
            "Use verification tools instead of trusting a single source.",
        ),
    ),
}


def get_task(task_id: str) -> BenchmarkTask:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise KeyError(f"Unknown task_id: {task_id}") from exc


def all_tasks() -> list[BenchmarkTask]:
    return list(TASKS.values())
