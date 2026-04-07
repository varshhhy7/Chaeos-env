# ChaosAgent v2: Unreliable Tools Resilience Environment

## Overview

ChaosAgent is an advanced testing environment designed to evaluate the resilience of large language model (LLM) agents. In real-world deployments, tools and APIs frequently fail, return stale data, timeout, or present silent corruption. ChaosAgent systematically tests an agent's ability to cross-validate data, handle exceptions, and navigate a deliberately unreliable tool ecosystem to arrive at the correct ground truth.

Version 2 introduces a fully deterministic scenario database, replacing the fragility of live APIs with pre-computed ground truths and a strictly programmatic evaluation criteria.

## Architecture and Core Infrastructure

The core engine has been successfully implemented with the following key components:

### 1. Environment and State Management
The `ChaosAgentEnvironment` serves as the primary orchestration layer. It tracks episodic state, maintains the continuous curriculum tier, monitors cyclic agent behaviors, and compiles penalty/bonus rewards based on step efficiency and tool utilization patterns.

### 2. Fault Injector Engine
To accurately simulate production instability, the environment features a sophisticated deterministic `FaultInjector`. Injection rates scale dynamically across four curriculum tiers (Warmup, Beginner, Intermediate, Expert) and include six distinct failure modes:
* **TIMEOUT**: Simulates exceeding maximum request times.
* **RATE_LIMIT**: Simulates HTTP 429 Too Many Requests errors.
* **STALE_DATA**: Returns historically outdated information.
* **SILENT_FAIL**: Returns empty or null values without an explicit error.
* **PARTIAL_RESPONSE**: Truncates list or nested dictionary formations in payloads.
* **CORRUPT_FIELD**: Subtly mutates numeric values or appends corruption flags to strings.

### 3. Scenario Tool Router
The environment relies on a suite of pre-computed scenario databases. The `ToolRouter` evaluates incoming agent queries using Jaccard-style overlap algorithms to fuzzy-match agent parameters against known database queries. This eliminates external network dependency and guarantees reproducible evaluations.

### 4. Programmatic Grader
To ensure stability during reinforcement learning (e.g., GRPO training), LLM-based grading has been excised. The programmatic `Grader` evaluates agent submissions through strict tolerance thresholding for numeric extractions, sub-string matching, and boolean logic validation.

### 5. Tool Catalog and Live Equivalents
The system implements a refined catalog of exactly 30 tools mapped across five domain capabilities (Information Retrieval, Computation & Transformation, Storage & State, Validation & Verification, and Communication & Action). While most tools route via the pre-computed scenario database, computation-heavy tools (e.g., calculator, Python execution, and working memory scratchpads) execute live natively within the environment to reflect internal agent cognition.

## Project Structure

```text
Chaeos_Agent/
├── server/
│   ├── __init__.py
│   ├── environment.py         # Main episode orchestration
│   ├── fault_injector.py      # Failure mode probability injection
│   ├── grader.py              # Formulaic answer evaluation
│   ├── tool_router.py         # Fuzzy matching scenarios
│   └── tools/
│       ├── __init__.py
│       ├── registry.py        # Pydantic schemas for the 30 tools
│       └── live_tools.py      # Internal calculator/execution logic
├── models.py                  # Core Pydantic state architectures
├── openenv.yaml               # OpenEnv compatibility manifest
├── pyproject.toml             # Dependency and build configuration
├── Dockerfile                 # Containerization outline
└── test_day1.py               # E2E validation script
```

## Quickstart and Testing

The current project uses `uv` for high-performance dependency resolution and virtual environment standardization. 

To initialize the environment and run the core pipeline simulation check:

```bash
uv venv
uv pip install -e .
uv run python test_day1.py
```

The validation script runs a dry initialization of the platform, routes a mock query mimicking an agent's request regarding demographic constants, retrieves simulated structured tool answers, grades the output independently, and returns a verified final episodic reward.
