from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from client import ChaosAgentEnv
from models import ChaosAgentAction, ChaosAgentObservation


DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_ENV_URL = "http://127.0.0.1:8000"
DEFAULT_TASK_NAME = "chaosagent"
DEFAULT_BENCHMARK = "chaosagent"

API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
ENV_URL = os.getenv("ENV_URL", DEFAULT_ENV_URL)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
TASK_NAME = os.getenv("TASK_NAME", DEFAULT_TASK_NAME)
BENCHMARK = os.getenv("BENCHMARK", DEFAULT_BENCHMARK)


SYSTEM_PROMPT = """You are solving a ChaosAgent OpenEnv episode.
You must reply with exactly one JSON object and no markdown.
Use this schema to call a tool:
{"type":"call_tool","tool_name":"tool_name","arguments":{"arg":"value"}}
Use this schema to submit the final answer:
{"type":"submit_answer","answer":"final answer","reasoning":"short reasoning"}
Prefer cross-checking facts when tools fail or disagree."""


@dataclass
class EpisodeResult:
    success: bool = False
    steps: int = 0
    score: float = 0.0
    rewards: list[float] = field(default_factory=list)


def _lower_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float | int | None) -> str:
    return f"{float(value or 0.0):.2f}"


def _format_action(action: ChaosAgentAction | str) -> str:
    if isinstance(action, str):
        return action
    return json.dumps(action.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":"))


def _single_line(value: str) -> str:
    return value.replace("\r", "\\r").replace("\n", "\\n")


def _format_error(error: str | None) -> str:
    if not error:
        return "null"
    return _single_line(str(error))


def _emit_start(task: str, env: str, model_name: str) -> None:
    print(
        f"[START] task={_single_line(task)} env={_single_line(env)} model={_single_line(model_name)}",
        flush=True,
    )


def _emit_step(
    *,
    step: int,
    action: ChaosAgentAction | str,
    reward: float | int | None,
    done: bool,
    error: str | None = None,
) -> None:
    safe_action = _single_line(_format_action(action))
    print(
        "[STEP] "
        f"step={step} "
        f"action={safe_action} "
        f"reward={_format_reward(reward)} "
        f"done={_lower_bool(done)} "
        f"error={_format_error(error)}",
        flush=True,
    )


def _emit_end(
    *,
    success: bool,
    steps: int,
    score: float | int | None,
    rewards: list[float],
) -> None:
    rewards_str = ",".join(_format_reward(reward) for reward in rewards)
    print(
        "[END] "
        f"success={_lower_bool(success)} "
        f"steps={steps} "
        f"score={_format_reward(score)} "
        f"rewards={rewards_str}",
        flush=True,
    )


def _validate_hf_token() -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required. Set HF_TOKEN, or set HUGGINGFACE_API_KEY for this project."
        )
    return token


def _make_openai_client() -> OpenAI:
    return OpenAI(api_key=_validate_hf_token(), base_url=API_BASE_URL)


def _observation_payload(observation: ChaosAgentObservation) -> dict[str, Any]:
    return observation.model_dump(mode="json", exclude_none=True)


def _messages_for_observation(
    history: list[dict[str, str]], observation: ChaosAgentObservation
) -> list[dict[str, str]]:
    observation_json = json.dumps(_observation_payload(observation), sort_keys=True)
    return [
        *history,
        {
            "role": "user",
            "content": f"Current observation:\n{observation_json}",
        },
    ]


def _parse_action(model_text: str) -> ChaosAgentAction:
    cleaned = model_text.strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match is None:
            return ChaosAgentAction(
                type="submit_answer",
                answer=cleaned,
                reasoning="Model returned non-JSON text.",
            )
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object")
    return ChaosAgentAction.model_validate(payload)


def _next_action(
    client: OpenAI,
    history: list[dict[str, str]],
    observation: ChaosAgentObservation,
) -> tuple[ChaosAgentAction, str]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=cast(
            list[ChatCompletionMessageParam],
            _messages_for_observation(history, observation),
        ),
        temperature=0,
    )
    model_text = response.choices[0].message.content or ""
    return _parse_action(model_text), model_text


def _score_from_step(
    observation: ChaosAgentObservation,
    reward: float,
    done: bool,
) -> float:
    if done and "correctness" in observation.metadata:
        return float(observation.metadata.get("correctness", 0.0) or 0.0)
    return max(0.0, min(1.0, reward))


async def run_episode(
    *,
    env_url: str,
    local_image_name: str | None,
    scenario_id: str | None,
    seed: int | None,
    max_agent_steps: int,
) -> EpisodeResult:
    llm_client = _make_openai_client()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    episode = EpisodeResult()
    env: ChaosAgentEnv | None = None

    try:
        if local_image_name:
            env = await ChaosAgentEnv.from_docker_image(local_image_name)
        else:
            env = await ChaosAgentEnv(base_url=env_url).connect()

        reset_kwargs: dict[str, Any] = {}
        if scenario_id is not None:
            reset_kwargs["scenario_id"] = scenario_id
        if seed is not None:
            reset_kwargs["seed"] = seed

        result = await env.reset(**reset_kwargs)
        observation = result.observation

        for step in range(1, max_agent_steps + 1):
            episode.steps = step
            action, raw_model_text = _next_action(llm_client, history, observation)
            history.append({"role": "assistant", "content": raw_model_text})

            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)
            episode.rewards.append(reward)
            episode.score = _score_from_step(observation, reward, bool(result.done))
            correctness = float(observation.metadata.get("correctness", 0.0) or 0.0)
            episode.success = bool(result.done and correctness > 0.5)

            tool_error = None
            if observation.tool_result is not None:
                tool_error = observation.tool_result.error

            _emit_step(
                step=step,
                action=action,
                reward=reward,
                done=bool(result.done),
                error=tool_error,
            )

            if result.done:
                break

        episode.score = max(0.0, min(1.0, episode.score))
        return episode
    finally:
        if env is not None:
            await env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ChaosAgent inference episode.")
    parser.add_argument("--env-url", default=ENV_URL)
    parser.add_argument("--local-image-name", default=LOCAL_IMAGE_NAME)
    parser.add_argument("--scenario-id", default=os.getenv("SCENARIO_ID"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-agent-steps", type=int, default=12)
    args = parser.parse_args()

    episode = EpisodeResult()
    caught: BaseException | None = None

    _emit_start(task=TASK_NAME, env=BENCHMARK, model_name=MODEL_NAME)
    try:
        episode = asyncio.run(
            run_episode(
                env_url=args.env_url,
                local_image_name=args.local_image_name,
                scenario_id=args.scenario_id,
                seed=args.seed,
                max_agent_steps=args.max_agent_steps,
            )
        )
    except Exception as exc:
        caught = exc
    finally:
        _emit_end(
            success=episode.success,
            steps=episode.steps,
            score=episode.score,
            rewards=episode.rewards,
        )

    if caught is not None:
        raise caught


if __name__ == "__main__":
    main()
