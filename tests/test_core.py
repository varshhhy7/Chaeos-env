from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pytest import CaptureFixture, MonkeyPatch

from client import ChaosAgentEnv
import inference
from models import (
    CallToolAction,
    DifficultyTier,
    Fact,
    FactType,
    Scenario,
    SubmitAnswerAction,
)
from server.app import app
from server.environment import ChaosAgentEnvironment
from server.grader import Grader
from server.scenario_repository import ScenarioRepository
from server.task_workspace import TaskWorkspace
from server.tasks import all_tasks, get_task
from server.tool_router import ToolRouter
from server.tools.live_tools import LiveTools
from server.tools.registry import get_all_tools, validate_tool_registry


def test_registry_and_scenario_fixture_sizes() -> None:
    validate_tool_registry()
    assert len(get_all_tools()) == 30
    assert len(ScenarioRepository.default().all()) == 55
    assert [task.id for task in all_tasks()] == ["task1", "task2", "task3"]


def test_environment_tool_call_and_submit() -> None:
    env = ChaosAgentEnvironment()
    reset_obs = env.reset(seed=7, scenario_id="W01", task_id="task1")

    assert reset_obs.scenario_id == "W01"
    assert reset_obs.task_id == "task1"
    assert reset_obs.available_tools is not None
    assert len(reset_obs.available_tools) == 30
    assert reset_obs.metadata["task_guidance"]

    tool_obs = env.step(
        CallToolAction(
            tool_name="database_query",
            arguments={"sql": "SELECT population FROM countries WHERE name='Germany'"},
        )
    )

    assert tool_obs.tool_result is not None
    assert tool_obs.tool_result.error is None
    assert tool_obs.tool_result.result == [{"population": 83_200_000}]
    assert tool_obs.available_tools is None
    assert tool_obs.metadata["retrieval_successes"] == 1

    final_obs = env.step(
        SubmitAnswerAction(
            answer="Germany has a population of 83,200,000.",
            reasoning="database result",
        )
    )

    assert final_obs.done is True
    assert final_obs.reward is not None
    assert final_obs.reward > 0.7
    assert final_obs.metadata["correctness"] == 1.0
    assert 0.0 < final_obs.metadata["task_score"] < 1.0


def test_tool_router_matches_entity_arguments() -> None:
    scenario = ScenarioRepository.default().get("B04")
    workspace = TaskWorkspace(scenario=scenario, task=get_task(scenario.benchmark_task_id))
    result = ToolRouter().route(
        "knowledge_base_lookup",
        {"entity": "Tokyo"},
        scenario,
        workspace=workspace,
    )

    assert result["country"] == "Japan"


def test_grader_handles_multiple_numeric_facts_and_units() -> None:
    scenario = Scenario(
        id="G01",
        benchmark_task_id="task1",
        question="test",
        answer={"population": 83_200_000, "actual_revenue_b": 1.87},
        required_facts=[
            Fact(
                key="population",
                value=83_200_000,
                type=FactType.NUMERIC,
                tolerance=0.02,
            ),
            Fact(
                key="actual_revenue_b",
                value=1.87,
                type=FactType.NUMERIC,
                tolerance=0.05,
            ),
        ],
        tool_data={},
        difficulty=DifficultyTier.WARMUP,
        min_tools_needed=1,
    )

    score = Grader().grade(
        "Population is 83.2 million and actual revenue was $1.87B.",
        scenario,
    )

    assert score == 1.0


def test_live_tools_are_stateful_and_bounded() -> None:
    tools = LiveTools(initial_files={"briefing/current_task.md": "hello"})

    assert tools.handle("calculator", {"expression": "sqrt(81) + 1"})["result"] == 10.0
    assert tools.handle("scratchpad_write", {"key": "answer", "value": 42})["status"] == "success"
    assert tools.handle("scratchpad_read", {"key": "answer"})["result"] == 42
    assert tools.handle("read_file", {"path": "briefing/current_task.md"})["content"] == "hello"

    python_result = tools.handle("python_execute", {"code": "print(2 + 2)"})
    assert python_result["status"] == "success"
    assert python_result["stdout"].strip() == "4"


def test_workspace_backed_tooling_uses_sql_and_documents() -> None:
    scenario = ScenarioRepository.default().get("E01")
    workspace = TaskWorkspace(scenario=scenario, task=get_task(scenario.benchmark_task_id))
    router = ToolRouter()

    sql_result = router.route(
        "database_query",
        {
            "sql": (
                "SELECT revenue_b, quarter FROM financials "
                "WHERE company='NovaTech' ORDER BY quarter DESC LIMIT 1"
            )
        },
        scenario,
        workspace=workspace,
    )
    document_result = router.route(
        "document_search",
        {"query": "NovaTech quarterly filing revenue"},
        scenario,
        workspace=workspace,
    )

    assert sql_result["result"] == [{"revenue_b": 1.87, "quarter": "Q3-2025"}]
    assert document_result["result"][0]["title"] == "NovaTech quarterly filing"


def test_openenv_fastapi_contract() -> None:
    client = TestClient(app)

    assert client.get("/health").json()["status"] == "healthy"
    assert len(client.get("/tasks").json()["tasks"]) == 3

    schema = client.get("/schema").json()
    assert "action" in schema
    assert "observation" in schema
    assert "state" in schema

    reset = client.post("/reset", json={"seed": 7, "task_id": "task1", "scenario_id": "W01"})
    assert reset.status_code == 200
    assert reset.json()["done"] is False

    step = client.post(
        "/step",
        json={
            "action": {
                "type": "submit_answer",
                "answer": "Germany has a population of 83,200,000.",
            }
        },
    )
    assert step.status_code == 200
    assert step.json()["done"] is True


def test_inference_token_fallback_and_output_format(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "dummy-token")

    assert inference._validate_hf_token() == "dummy-token"

    inference._emit_step(
        step=1,
        action="demo",
        reward=1,
        done=True,
        error=None,
    )
    inference._emit_end(success=False, steps=1, score=1, rewards=[1.0])

    output = capsys.readouterr().out
    assert "[STEP] step=1 action=demo reward=1.00 done=true error=null" in output
    assert "[END] success=false steps=1 score=1.00 rewards=1.00" in output


def test_inference_missing_token_raises(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN is required"):
        inference._validate_hf_token()


def test_client_parser_keeps_top_level_metadata() -> None:
    observation = ChaosAgentEnv._parse_observation(
        {
            "observation": {
                "task_question": "q",
                "scenario_id": "W01",
            },
            "done": False,
            "reward": 0.0,
            "metadata": {"compute_calls": 1},
        }
    )

    assert observation.metadata["compute_calls"] == 1


def test_task_grader_clamps_scores_to_open_interval() -> None:
    env = ChaosAgentEnvironment()
    env.reset(seed=7, scenario_id="W01", task_id="task1")
    final_obs = env.step(
        SubmitAnswerAction(
            answer="Germany has a population of 83,200,000.",
            reasoning="database result",
        )
    )

    assert 0.0 < final_obs.metadata["task_score"] < 1.0


def test_task_three_rewards_evidence_management() -> None:
    env = ChaosAgentEnvironment()
    env.reset(seed=7, scenario_id="E01", task_id="task3")

    env.step(
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
    env.step(
        CallToolAction(
            tool_name="fact_check",
            arguments={"claim": "NovaTech reported revenue of $2.30B"},
        )
    )
    env.step(
        CallToolAction(
            tool_name="create_report",
            arguments={"content": "Claim is inaccurate; database and filing disagree with rumor."},
        )
    )

    state = env.state
    assert state.verification_calls >= 1
    assert state.artifact_actions >= 1
