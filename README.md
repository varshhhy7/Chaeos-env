# 🌩️ ChaosAgent

> An OpenEnv benchmark and RL training environment for LLM agents operating under unreliable tools.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenEnv Supported](https://img.shields.io/badge/OpenEnv-Supported-brightgreen.svg)](https://hf.co/spaces/openenv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-orange)](https://huggingface.co/spaces/Prahaladha/chaosagent-openenv)

---

## Why ChaosAgent?

Most agent benchmarks assume a perfect world. Tools return clean data. APIs respond instantly. Nothing goes wrong.

Production doesn't work that way.

Real systems time out. APIs rate-limit. Data gets silently truncated, stale, or corrupted. An agent that can only reason under ideal conditions is brittle the moment it leaves the lab.

**ChaosAgent fills this gap.** It is a deterministic, reproducible benchmark that tests whether an LLM agent can reason its way through unreliable tool responses to still arrive at the correct answer. It is also designed as an RL training environment — shaped rewards, tiered curriculum, and deterministic grading make it stable enough to actually train on.

---

## Table of Contents

- [Core Concept](#core-concept)
- [Benchmark Tasks](#benchmark-tasks)
- [Fault Injection](#fault-injection)
- [Tool Ecosystem](#tool-ecosystem)
- [Environment Architecture](#environment-architecture)
- [RL Training Design](#rl-training-design)
- [Quickstart](#quickstart)
- [Running Inference](#running-inference)
- [Inference Contract](#inference-contract)
- [Project Structure](#project-structure)
- [OpenEnv Entry Points](#openenv-entry-points)
- [Submission Surface](#submission-surface)
- [Contributing](#contributing)

---

## Core Concept

The benchmark is organized around three escalating failure modes:

| Task | Core Challenge |
|------|---------------|
| `task1` | Recover from a single bad retrieval path |
| `task2` | Reconcile conflicting evidence across multiple tools |
| `task3` | Adapt under cascading, repeated failures |

Every scenario poses a question with a hidden answer key. Tools execute against a **real per-episode workspace** — not hardcoded string payloads — so the environment is reproducible without being gameable. The workspace is backed by:

- SQLite tables for structured data
- An indexed document corpus
- Synthetic URLs and internal APIs
- Virtual files, notes, reports, notifications, and tickets

---

## Benchmark Tasks

| Task | Focus | Scenario pool | Max steps | What the grader rewards |
|------|-------|---------------|-----------|-------------------------|
| `task1` | Recover from one bad retrieval path | warmup + beginner | 6 | Correctness, switching away from failing calls, efficient retrieval |
| `task2` | Reconcile conflicting evidence | intermediate | 8 | Correctness, cross-validation, compute and verification usage |
| `task3` | Adapt under repeated faults | expert | 10 | Correctness, resilience, verification, evidence management |

Task guidance is embedded in reset metadata so an LLM agent understands what recovery behavior is expected without seeing the hidden answer.

**55 deterministic scenarios** are distributed across four difficulty tiers:

```
warmup       →  basic facts, clean data
beginner     →  minor timeouts, simple single-step retrieval  
intermediate →  active fault injection, multi-step verification required
expert       →  high-probability faults, heavy cross-referencing, scratchpad utilization
```

---

## Fault Injection

ChaosAgent injects faults **deterministically via seeded randomness**, not live chaos. This is a deliberate design choice: reproducibility is required for fair grading and stable RL training.

Fault types injected per tier:

| Fault Type | Description |
|------------|-------------|
| **Timeout** | Tool call hangs and returns no data |
| **Rate limit** | Tool refuses the call and signals backoff |
| **Stale data** | Tool returns an outdated version of the record |
| **Silent field drop** | Response is returned but one or more fields are missing without error |
| **Truncation** | Response is cut off mid-content |
| **Value corruption** | A field is returned with a plausible but incorrect value |

Fault probabilities scale with tier. Expert scenarios inject faults aggressively and penalize agents for repeating broken calls via an **escalating repeat-call penalty tracker**.

---

## Tool Ecosystem

ChaosAgent ships **30 integrated tools** across five execution categories:

| Category | Examples |
|----------|---------|
| **Retrieval** | Web search, database query, document lookup, external API fetch |
| **Computation** | Arithmetic, aggregation, date calculations |
| **Validation** | Schema checks, structural parsing, JSON querying |
| **Storage** | Scratchpad memory, virtual filesystem read/write |
| **Action** | Ticket management, notifications, report generation |

All tools are typed. Explicit tool schemas are provided to the agent on reset so it can reason about what's available without hallucinating interfaces.

---

## Environment Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Server                    │
│  /reset  /step  /state  /schema  /metadata          │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │     Environment Core       │
          │  - Scenario loader         │
          │  - Fault injector (seeded) │
          │  - Repeat call tracker     │
          │  - Step limiter            │
          └─────────────┬──────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────▼────┐    ┌──────▼──────┐  ┌───▼──────┐
   │ SQLite  │    │  Doc Corpus │  │ Virtual  │
   │  Tables │    │  (indexed)  │  │   APIs   │
   └─────────┘    └─────────────┘  └──────────┘
                        │
          ┌─────────────▼──────────────┐
          │     Programmatic Grader    │
          │  - No LLM in the loop      │
          │  - Extracts text / date /  │
          │    boolean / numeric       │
          │  - Scores clamped [0, 1]   │
          └────────────────────────────┘
```

The grader is **fully LLM-free and programmatic**. It precisely extracts structured values from the agent's final response string. Using an LLM to grade a benchmark about LLM reliability would be circular and fragile — we deliberately avoid it.

---

## RL Training Design

ChaosAgent is explicitly designed to be trainable, not just evaluable.

**Why it works as an RL environment:**

Good RL training requires a reward signal that is dense, meaningful, and hard to game. ChaosAgent provides exactly that. The agent is not simply scored on the final answer — intermediate behavior is rewarded and penalized throughout the episode:

- ✅ Switching away from a failing tool → rewarded
- ✅ Cross-validating conflicting evidence → rewarded  
- ✅ Using computation/verification tools → rewarded
- ❌ Repeating broken calls → penalized (escalating)
- ❌ Exceeding step budget → episode terminates

The **tiered curriculum** (warmup → beginner → intermediate → expert) allows an agent to learn progressively. Agents can be moved through tiers based on rolling accuracy, enabling curriculum learning out of the box.

The **deterministic backend** means every training run sees exactly the same fault patterns for the same seeds. This makes reward variance controllable and debugging tractable.

```
Episode loop:
  reset()  →  returns scenario, tool schemas, task guidance, metadata
  step()   →  takes action, returns observation, reward, done flag
  [repeat up to max_steps]
  grader   →  scores final answer, emits per-step reward list
```

All scores are strictly clamped to `[0.0, 1.0]`.

---

## Quickstart

### Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone https://github.com/varshhhy7/Chaeos-env.git
cd Chaeos-env

# Sync all dependencies
uv sync

# Run tests
uv run pytest -v

# Lint
uv run ruff check .

# Type check
uv run mypy models.py server client.py inference.py demo.py tests

# Validate against OpenEnv spec
uv run openenv validate --verbose
```

### Start the server

```bash
uv run server --host 127.0.0.1 --port 8000
```

Verify readiness:
```bash
curl http://127.0.0.1:8000/health
```

### Run the demo (no API key required)

```bash
uv run python demo.py
```

This runs a mock simulation of an episode with no external API calls — useful for verifying environment behavior locally.

---

## Running Inference

Make sure the server is running, then:

```bash
# Set your API key
export HF_TOKEN=your_token_here

# Run inference against a specific scenario
uv run python inference.py --env-url http://127.0.0.1:8000 --scenario-id W01
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API base URL |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use for inference |
| `API_KEY` | — | Preferred API key (evaluation) |
| `HF_TOKEN` | — | Local fallback; also falls back to `HUGGINGFACE_API_KEY` |
| `ENV_URL` | `http://127.0.0.1:8000` | ChaosAgent server URL |
| `LOCAL_IMAGE_NAME` | — | Optional, for `from_docker_image(...)` |

---

## Inference Contract

`inference.py` uses the OpenAI Python client for all LLM calls (compatible with HuggingFace's router).

**Stdout format:**

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
```

- `[END]` is always emitted regardless of success or failure
- Rewards and scores are formatted to two decimal places
- Booleans are lowercase (`true` / `false`)

---

## Project Structure

```
ChaosAgent/
├── server/
│   ├── app.py                  # FastAPI application
│   ├── environment.py          # Core episode loop
│   ├── scenario_repository.py  # 55 scenario definitions
│   ├── fault_injector.py       # Seeded fault injection
│   └── grader.py               # Programmatic grader (LLM-free)
├── client.py                   # Typed OpenEnv client (ChaosAgentEnv)
├── models.py                   # Action + Observation typed models
├── inference.py                # Inference entry point
├── demo.py                     # Mock demo (no API key needed)
├── tests/                      # Pytest test suite
├── openenv.yaml                # OpenEnv manifest
├── pyproject.toml              # uv-managed dependencies
└── requirements.txt            # Pip-compatible requirements
```

---

## OpenEnv Entry Points

| Entry Point | Value |
|-------------|-------|
| Manifest | `openenv.yaml` |
| Server app | `server.app:app` |
| Local server command | `uv run server --port 8000` |
| Typed client | `client.ChaosAgentEnv` |
| Action model | `models.ChaosAgentAction` |
| Observation model | `models.ChaosAgentObservation` |
| Demo script | `python demo.py` |
| Inference script | `python inference.py --env-url http://127.0.0.1:8000` |

---

## Submission Surface

| | |
|-|-|
| **GitHub** | [varshhhy7/Chaeos-env](https://github.com/varshhhy7/Chaeos-env) |
| **HuggingFace Space** | [Prahaladha/chaosagent-openenv](https://huggingface.co/spaces/Prahaladha/chaosagent-openenv) |
| **Runtime URL** | `https://prahaladha-chaosagent-openenv.hf.space` |
| **Docker entrypoint** | `uvicorn server.app:app --host 0.0.0.0 --port 8000` |

---

## Contributing

Contributions are welcome — especially new scenarios, additional fault types, and new tool adapters.

**To add a scenario:**

Open `server/scenario_repository.py` and add a new declarative configuration matching the schema in `models.py`. Scenarios must specify: question, answer key, tier, task assignment, and any scenario-specific workspace seed.

**To add a tool:**

Register the tool in the tool router, add its schema to the reset metadata, and write a corresponding test in `tests/`.

**Before submitting a PR:**

```bash
uv run pytest -v
uv run ruff check .
uv run mypy models.py server client.py inference.py demo.py tests
uv run openenv validate --verbose
```

All four must pass cleanly.

---

## Design Philosophy

ChaosAgent is built on three principles:

**Reproducibility over freshness.** All faults are seeded. All backends are deterministic. No live external APIs in the grading path. This makes it possible to train on, not just evaluate against.

**Behavior-shaped rewards.** Scoring is not binary. The grader rewards the *quality of reasoning under failure*, not just whether the final answer is correct. An agent that blindly retries a broken tool five times and gets lucky is scored differently from one that detects the failure and switches strategies.

**No LLM in the loop for grading.** Using a language model to judge a language model benchmark introduces the very unreliability we are trying to measure. The grader is programmatic, deterministic, and transparent.

---

*Built for [OpenEnv](https://hf.co/spaces/openenv). Designed to make LLM agents more robust in the real world.*
