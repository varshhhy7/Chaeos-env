from __future__ import annotations

from models import CallToolAction, SubmitAnswerAction
from server.environment import ChaosAgentEnvironment


def _run_task_one(env: ChaosAgentEnvironment) -> None:
    obs = env.reset(seed=7, scenario_id="W01", task_id="task1")
    print(f"[task1] Question: {obs.task_question}")

    obs = env.step(
        CallToolAction(
            tool_name="database_query",
            arguments={"sql": "SELECT population FROM countries WHERE name='Germany'"},
        )
    )
    print(f"[task1] database_query -> {obs.tool_result.model_dump() if obs.tool_result else None}")

    obs = env.step(
        SubmitAnswerAction(
            answer="Germany has a population of 83,200,000.",
            reasoning="Retrieved from the countries table.",
        )
    )
    print(f"[task1] reward={float(obs.reward or 0.0):.2f} score={obs.metadata['task_score']:.3f}")


def _run_task_two(env: ChaosAgentEnvironment) -> None:
    obs = env.reset(seed=7, scenario_id="I01", task_id="task2")
    print(f"[task2] Question: {obs.task_question}")

    obs = env.step(
        CallToolAction(
            tool_name="database_query",
            arguments={
                "sql": (
                    "SELECT name, population, area_km2 FROM countries "
                    "WHERE name IN ('India', 'Australia')"
                )
            },
        )
    )
    print(f"[task2] database_query -> {obs.tool_result.model_dump() if obs.tool_result else None}")

    obs = env.step(
        CallToolAction(
            tool_name="calculator",
            arguments={"expression": "(1428627663/3287263)/(26473055/7692024)"},
        )
    )
    print(f"[task2] calculator -> {obs.tool_result.model_dump() if obs.tool_result else None}")

    obs = env.step(
        SubmitAnswerAction(
            answer=(
                "India has the higher population density. "
                "India is about 126.4 times denser than Australia."
            ),
            reasoning="Computed density ratio from the countries table.",
        )
    )
    print(f"[task2] reward={float(obs.reward or 0.0):.2f} score={obs.metadata['task_score']:.3f}")


def _run_task_three(env: ChaosAgentEnvironment) -> None:
    obs = env.reset(seed=7, scenario_id="E01", task_id="task3")
    print(f"[task3] Question: {obs.task_question}")

    obs = env.step(
        CallToolAction(
            tool_name="database_query",
            arguments={
                "sql": (
                    "SELECT revenue_b, quarter FROM financials "
                    "WHERE company='NovaTech' ORDER BY quarter DESC LIMIT 1"
                )
            },
        )
    )
    print(f"[task3] database_query -> {obs.tool_result.model_dump() if obs.tool_result else None}")

    obs = env.step(
        CallToolAction(
            tool_name="fact_check",
            arguments={"claim": "NovaTech reported revenue of $2.30B"},
        )
    )
    print(f"[task3] fact_check -> {obs.tool_result.model_dump() if obs.tool_result else None}")

    obs = env.step(
        CallToolAction(
            tool_name="create_report",
            arguments={
                "content": (
                    "NovaTech claim is inaccurate. Database revenue and filing checks "
                    "both point to 1.87B."
                )
            },
        )
    )
    print(f"[task3] create_report -> {obs.tool_result.model_dump() if obs.tool_result else None}")

    obs = env.step(
        SubmitAnswerAction(
            answer=(
                "The claim is not accurate. NovaTech's actual revenue last quarter was $1.87B."
            ),
            reasoning="Verified with database data and fact_check output, then recorded a report.",
        )
    )
    print(f"[task3] reward={float(obs.reward or 0.0):.2f} score={obs.metadata['task_score']:.3f}")


def main() -> None:
    env = ChaosAgentEnvironment()
    try:
        _run_task_one(env)
        _run_task_two(env)
        _run_task_three(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
