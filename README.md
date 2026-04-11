---
title: ChaosAgent V2 Environment
emoji: "🌩️"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# 🌩️ ChaosAgent V2: Unreliable-Tool Resilience Environment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenEnv Supported](https://img.shields.io/badge/OpenEnv-Supported-brightgreen.svg)](https://hf.co/spaces/openenv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**ChaosAgent V2** is a dynamic OpenEnv assessment environment engineered specifically to test Large Language Model (LLM) agents against **the harsh realities of production systems.** Unlike standard benchmarks where tools return clean data instantly, ChaosAgent systematically injects deterministic faults.

Agents tested against ChaosAgent must navigate timeouts, rate-limits, stale data, silent field drops, truncated responses, corrupted values, and unreliable retrieval mechanisms to solve highly complex, multi-step scenarios.

---

## 🌟 Core Architecture & Features

### 1. Deterministic Multi-Tiered Curriculum
ChaosAgent ships with a dynamic **Curriculum Controller** that seamlessly moves agents through multiple difficulty tiers based on rolling accuracy:
*   **Warmup**: Basic facts, clean data.
*   **Beginner**: Minor timeouts and simple single-step retrieval.
*   **Intermediate**: Active fault injection (e.g., rate-limits, schema drops), requiring multi-step verification.
*   **Expert**: High-probability faults heavily penalizing repeat-calls; tasks require extensive cross-referencing and scratchpad utilization.
*(Includes 55 distinct deterministic scenario fixtures).*

### 2. Comprehensive Tool Ecosystem
A massive catalog of **30 integrated tools** split into distinct execution categories:
*   **Retrieval Tools**: Web search, databases, external API endpoints.
*   **Internal Utilities**: JSON querying, structural validation, parsing operations, and calculation.
*   **Stateful Operations**: Virtual ticket tracking, notifications, filesystem, and scratchpad memory management.

### 3. Programmatic Precision Grading
An isolated, LLM-free programmatic grading system that precisely extracts text, date, boolean, and mathematical values from the agent's final string responses. **Scoring strictly ensures ranges evaluate securely between `[0.0, 1.0]`** (addressing previous validation pipeline errors). 

### 4. Headless OpenEnv Standard
Fully implemented OpenEnv API compliant FastAPI container:
*   Standardized `/reset`, `/step`, `/state`, `/schema` and `/metadata` endpoints.
*   Typed WebSocket clients seamlessly bridging the environment and any third-party orchestrator.

---

## 📊 What's New: Recent Changes & Fixes

During the most recent V2 deployment milestone, the following major updates were securely implemented and verified:
1.  **Refined Evaluation Boundaries**: Solved OpenEnv pipeline validation errors (`"scores are out of range"`) by mathematically bounding all metric accumulations within absolute `[0.0, 1.0]` bounds for validation compliance.
2.  **Modernized Dependency Orchestration**: Successfully migrated the entire environment to the high-performance `uv` package manager (`pyproject.toml` and `uv.lock`) securing build determinism and optimizing `uv sync` operations.
3.  **Comprehensive E2E Test Suite Passed**: Re-verified the `server/environment.py` loop against `pytest`. Currently passing 100% test coverage for the tool router, environment limits, live tools, and FastAPI contracts.
4.  **Advanced Repeat Tracker Penalty Scheme**: Agents are fundamentally discouraged from spinning into local loops by an escalating penalty tracker enforcing intelligent fallback behavior.

---

## 🚀 Quickstart Guide

This environment relies on `uv` for ultra-fast dependency resolution. 

### Installation & Validation

```bash
# Clone the repository
git clone https://github.com/varshhhy7/Chaeos-env.git
cd Chaeos-env

# Instantly sync standard and dev environments
uv sync

# Run the programmatic tests and typing validation checks
uv run pytest -v
uv run ruff check .
uv run mypy models.py server client.py tests

# Validate environment schemas against the OpenEnv spec locally
uv run openenv validate --verbose
```

### Running Locally

To manually spin up the FastAPI-powered environment server locally to handle agent requests over port 8000:

```bash
uv run server --host 127.0.0.1 --port 8000
```
*(Verify server readiness asynchronously by querying `http://127.0.0.1:8000/health`).*

---

## 💻 Running AI Inferences Against the Environment

ChaosAgent natively provides an entry point that runs inferences against a running server. Make sure the local server is running (see above) and use the `inference.py` engine.

```bash
# Export your HuggingFace key (powershell example)
$env:HF_TOKEN="your_token_here"

# Run inference against Warmup scenario 01 utilizing the default Qwen2.5 72B Instruct Model
uv run python inference.py --env-url http://127.0.0.1:8000 --scenario-id W01
```

If you wish to simply observe how an environment state manages action steps safely without API tokens, run the mock demo simulation:
```bash
uv run python demo.py
```

---

## 📝 Design Philosophy

ChaosAgent intentionally strips API complexities to simulate network problems deterministically. **Reproducibility is prized over real-time API freshness.** An agent must prove it comprehends errors structurally and logically to succeed, making it the perfect benchmark dataset for advanced Agentic RL Frameworks (RLHF / PPO implementations).

### Contributing & Modifying Scenarios
To contribute custom workflows, open `server/scenario_repository.py` and implement new declarative configurations ensuring they match the schema defined in `models.py`.

---

## 🔗 Submission Details & Cloud Implementations
*   **Github Remote:** `https://github.com/varshhhy7/Chaeos-env.git`
*   **Environment Provider Manifest:** `openenv.yaml`
*   **Production HuggingFace Space URL (Deployment Target):** [Prahaladha HF Space Endpoint](https://huggingface.co/spaces/Prahaladha/chaosagent-openenv)
*   **Runtime:** `uvicorn server.app:app --host 0.0.0.0 --port 8000`
