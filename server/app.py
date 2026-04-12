from __future__ import annotations

try:
    from openenv.core.env_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError("Install dependencies with `uv sync` before starting the server.") from exc

from fastapi import Body
from pydantic import BaseModel

from models import ChaosAgentAction, ChaosAgentObservation, ChaosAgentState
from server.environment import ChaosAgentEnvironment
from server.grader import Grader
from server.scenario_repository import ScenarioRepository
from server.tasks import all_tasks, get_task


app = create_app(
    ChaosAgentEnvironment,
    ChaosAgentAction,
    ChaosAgentObservation,
    env_name="chaosagent",
    max_concurrent_envs=4,
)


class GradeRequest(BaseModel):
    task_id: str
    scenario_id: str
    state: ChaosAgentState
    correctness: float = 0.0
    answered: bool = True


@app.get("/tasks")
def list_tasks() -> dict[str, list[dict[str, object]]]:
    return {
        "tasks": [
            {
                "id": task.id,
                "name": task.name,
                "difficulty": task.difficulty,
                "description": task.description,
                "max_steps": task.max_steps,
                "target_tool_diversity": task.target_tool_diversity,
                "target_recovery_switches": task.target_recovery_switches,
                "target_verification_calls": task.target_verification_calls,
                "target_compute_calls": task.target_compute_calls,
                "target_artifact_actions": task.target_artifact_actions,
                "guidance": list(task.guidance),
            }
            for task in all_tasks()
        ]
    }


@app.post("/grade")
def grade_episode(request: GradeRequest = Body(...)) -> dict[str, float | str]:
    scenario = ScenarioRepository.default().get(request.scenario_id)
    task = get_task(request.task_id)
    grader = Grader()
    score = grader.grade_task(
        task=task,
        scenario=scenario,
        state=request.state,
        correctness=request.correctness,
        answered=request.answered,
    )
    return {"task_id": task.id, "scenario_id": scenario.id, "score": score}


def main() -> None:
    """Run the OpenEnv FastAPI server; local validation looks for main()."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
