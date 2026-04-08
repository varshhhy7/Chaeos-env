---
title: ChaosAgent OpenEnv
emoji: "🛠️"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# ChaosAgent

ChaosAgent is an OpenEnv environment for testing whether agents can answer
questions when their tools may time out, rate-limit, return stale data, silently
drop fields, truncate responses, or corrupt values.

The environment is deterministic by design: every scenario has a precomputed
answer key and precomputed tool outputs. Retrieval tools route through that
scenario data, while pure utility tools such as calculation, JSON querying,
schema validation, scratchpad memory, and virtual file/ticket actions execute
inside the episode.

## OpenEnv Entry Points

- Manifest: `openenv.yaml`
- Server app: `server.app:app`
- Local server command: `uv run server --port 8000`
- Typed client: `client.ChaosAgentEnv`
- Action model: `models.ChaosAgentAction`
- Observation model: `models.ChaosAgentObservation`
- Baseline inference script: `python inference.py --env-url http://127.0.0.1:8000`
- Demo script: `python demo.py`

## What Is Implemented

- 30 distinct tools across retrieval, computation, state, validation, and action categories.
- 55 deterministic scenario fixtures across warmup, beginner, intermediate, and expert tiers.
- Seeded fault injection with tier-specific probabilities and never-fail internal tools.
- Programmatic grading for numeric, text, boolean, and date facts.
- Per-episode scratchpad, virtual file store, notifications, ticket updates, reports, and scheduled tasks.
- OpenEnv-compatible FastAPI server with `/reset`, `/step`, `/state`, `/schema`, `/metadata`, `/health`, and `/mcp`.
- Typed WebSocket client for agents and tests.
- Pytest, Ruff, Mypy, local OpenEnv validation, and runtime OpenEnv validation coverage.

## Quickstart

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy models.py server client.py tests
uv run openenv validate --verbose
```

Run the environment locally:

```bash
uv run server --host 127.0.0.1 --port 8000
```

Validate a running server:

```bash
uv run openenv validate --url http://127.0.0.1:8000
```

Run the deterministic local demo:

```bash
uv run python demo.py
```

Run the Round 1 inference script against a running environment:

```bash
uv run python inference.py --env-url http://127.0.0.1:8000 --scenario-id W01
```

`inference.py` reads:

- `API_BASE_URL`, defaulting to `https://router.huggingface.co/v1`
- `MODEL_NAME`, defaulting to `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN`; if absent, it falls back to `HUGGINGFACE_API_KEY`
- `ENV_URL`, defaulting to `http://127.0.0.1:8000`
- `LOCAL_IMAGE_NAME`, optional; when set, `inference.py` starts that local Docker image
  through `ChaosAgentEnv.from_docker_image(...)`

The inference output uses only:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
```

Rewards and scores are formatted to two decimals, booleans are lower-case, and
`[END]` is emitted even when an exception occurs.

## Example Direct Use

```python
from models import CallToolAction, SubmitAnswerAction
from server.environment import ChaosAgentEnvironment

env = ChaosAgentEnvironment()
obs = env.reset(seed=7, scenario_id="W01")

obs = env.step(
    CallToolAction(
        tool_name="database_query",
        arguments={"sql": "SELECT population FROM countries WHERE name='Germany'"},
    )
)

obs = env.step(
    SubmitAnswerAction(
        answer="Germany has a population of 83,200,000.",
        reasoning="Checked the deterministic database query result.",
    )
)

print(obs.reward, obs.done)
```

## Design Notes

ChaosAgent intentionally does not require live web/API keys during evaluation.
For OpenEnv and RL training, reproducibility is more important than live API
freshness: the agent must recover the scenario's hidden facts from tool calls,
and the environment must be gradable without network variance. Utility tools are
implemented as real deterministic episode tools rather than no-op mocks.

To add new tasks, extend the structured fixture records in
`server/scenario_repository.py`; the repository validates that the default
fixture set contains exactly 55 scenarios.

## Submission Notes

- GitHub remote currently configured locally: `https://github.com/varshhhy7/Chaeos-env.git`
- Requirements file: `requirements.txt`
- Docker image entrypoint: `uvicorn server.app:app --host 0.0.0.0 --port 8000`
- Hugging Face Space repo URL: `https://huggingface.co/spaces/Prahaladha/chaosagent-openenv`
- Hugging Face Space runtime URL: `https://prahaladha-chaosagent-openenv.hf.space`
