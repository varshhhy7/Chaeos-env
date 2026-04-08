from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import ChaosAgentAction, ChaosAgentObservation, ChaosAgentState, ToolResult


class ChaosAgentEnv(EnvClient[ChaosAgentAction, ChaosAgentObservation, ChaosAgentState]):
    """Typed WebSocket client for a running ChaosAgent OpenEnv server."""

    def _step_payload(self, action: ChaosAgentAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[ChaosAgentObservation]:
        observation = self._parse_observation(payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict[str, Any]) -> ChaosAgentState:
        return ChaosAgentState.model_validate(payload)

    @staticmethod
    def _parse_observation(payload: dict[str, Any]) -> ChaosAgentObservation:
        obs_data = payload.get("observation", {})
        if not isinstance(obs_data, dict):
            obs_data = {}

        tool_result_data = obs_data.get("tool_result")
        if isinstance(tool_result_data, dict):
            obs_data["tool_result"] = ToolResult.model_validate(tool_result_data)

        return ChaosAgentObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", obs_data.get("done", False)),
                "reward": payload.get("reward", obs_data.get("reward")),
                "metadata": obs_data.get("metadata", {}),
            }
        )
