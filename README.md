---
title: ChaosAgent OpenEnv
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
suggested_hardware: cpu-basic
header: mini
short_description: OpenEnv benchmark for resilient tool use under failures.
pinned: false
---

# ChaosAgent

ChaosAgent is an OpenEnv benchmark for agents operating under unreliable tools.
The core problem stays the same across the benchmark: the agent must answer
correctly while tools may time out, rate-limit, return stale data, silently
drop fields, truncate responses, or corrupt values.

The benchmark is organized as three explicit tasks:

- `task1`: single-failure recovery
- `task2`: cross-tool conflict resolution
- `task3`: cascading failure adaptation

Scenarios provide the question and answer key, but the tools execute against a
real per-episode workspace backed by internal systems:

- SQLite tables for structured data
- indexed document corpus
- synthetic URLs and internal APIs
- virtual files, notes, reports, notifications, and tickets

That keeps the environment reproducible for grading without reducing the tool
surface to hard-coded string lookup.

## OpenEnv Entry Points

- Manifest: `openenv.yaml`
- Server app: `server.app:app`
- Local server command: `uv run server --port 8000`
- Typed client: `client.ChaosAgentEnv`
- Action model: `models.ChaosAgentAction`
- Observation model: `models.ChaosAgentObservation`
- Demo script: `python demo.py`
- Inference script: `python inference.py --env-url http://127.0.0.1:8000`

## What Is Implemented

- 30 tools across retrieval, computation, storage, validation, and action
- 55 deterministic scenarios across warmup, beginner, intermediate, and expert tiers
- 3 named tasks with distinct grading rubrics and scores clamped to `(0, 1)`
- per-episode workspace-backed retrieval instead of static scenario tool payloads
- seeded fault injection with tier-specific probabilities
- typed OpenEnv FastAPI server, typed client, tests, lint, type checking, and validator coverage

## Benchmark Tasks

| Task | Focus | Scenario pool | Max steps | What the grader rewards |
|------|-------|---------------|-----------|--------------------------|
| `task1` | recover from one bad retrieval path | warmup + beginner | 6 | correctness, switching away from failing calls, efficient retrieval |
| `task2` | reconcile conflicting evidence | intermediate | 8 | correctness, cross-validation, compute/verification usage |
| `task3` | adapt under repeated faults | expert | 10 | correctness, resilience, verification, evidence management |

Task guidance is included in reset metadata so an LLM agent can tell what
behavior is expected without seeing the hidden answer.

## Quickstart

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy models.py server client.py inference.py demo.py tests
uv run openenv validate --verbose
```

Run the server:

```bash
uv run server --host 127.0.0.1 --port 8000
```

Run the demo:

```bash
uv run python demo.py
```

Run inference against a running server:

```bash
uv run python inference.py --env-url http://127.0.0.1:8000
```

## Inference Contract

`inference.py` is at the repo root and uses the OpenAI Python client for all
LLM calls.

Environment variables:

- `API_BASE_URL`, default `https://router.huggingface.co/v1`
- `MODEL_NAME`, default `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN`, required; falls back to `HUGGINGFACE_API_KEY`
- `ENV_URL`, default `http://127.0.0.1:8000`
- `LOCAL_IMAGE_NAME`, optional for `from_docker_image(...)`

Stdout format:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
```

`[END]` is always emitted, rewards and scores are formatted to two decimals, and
booleans are lower-case.

## Design Notes

ChaosAgent is designed to be trainable for LLM agents:

- typed action and observation contracts
- explicit tool schemas on reset
- deterministic backends and graders
- task guidance in metadata
- task scores shaped around recovery, cross-validation, computation, and evidence management

It deliberately avoids depending on live external APIs for benchmark
correctness. External APIs can be added later as optional tool adapters, but the
grading path stays deterministic.

## Submission Surface

- Public GitHub repo: `https://github.com/varshhhy7/Chaeos-env`
- Hugging Face Space repo: `https://huggingface.co/spaces/Prahaladha/chaosagent-openenv`
- Hugging Face runtime URL: `https://prahaladha-chaosagent-openenv.hf.space`
- Requirements file: `requirements.txt`
- Docker entrypoint: `uvicorn server.app:app --host 0.0.0.0 --port 8000`
