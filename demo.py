from __future__ import annotations

from models import CallToolAction, SubmitAnswerAction
from server.environment import ChaosAgentEnvironment


def main() -> None:
    env = ChaosAgentEnvironment()
    obs = env.reset(seed=7, scenario_id="W01")
    print(f"Question: {obs.task_question}")

    obs = env.step(
        CallToolAction(
            tool_name="database_query",
            arguments={"sql": "SELECT population FROM countries WHERE name='Germany'"},
        )
    )
    print(f"Tool result: {obs.tool_result.model_dump() if obs.tool_result else None}")

    obs = env.step(
        SubmitAnswerAction(
            answer="Germany has a population of 83,200,000.",
            reasoning="Retrieved from the deterministic database query result.",
        )
    )
    print(f"Done: {str(obs.done).lower()}")
    print(f"Reward: {float(obs.reward or 0.0):.2f}")


if __name__ == "__main__":
    main()
