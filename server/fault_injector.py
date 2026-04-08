from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from enum import Enum
from typing import Any

from models import DifficultyTier


class FaultMode(str, Enum):
    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    STALE_DATA = "STALE_DATA"
    SILENT_FAIL = "SILENT_FAIL"
    PARTIAL_RESPONSE = "PARTIAL_RESPONSE"
    CORRUPT_FIELD = "CORRUPT_FIELD"


class FaultInjector:
    """Seeded fault injector with bounded per-tier failure probabilities."""

    TIER_PROBS: dict[DifficultyTier, dict[FaultMode, float]] = {
        DifficultyTier.WARMUP: {
            FaultMode.TIMEOUT: 0.08,
            FaultMode.RATE_LIMIT: 0.04,
        },
        DifficultyTier.BEGINNER: {
            FaultMode.TIMEOUT: 0.10,
            FaultMode.RATE_LIMIT: 0.06,
            FaultMode.STALE_DATA: 0.08,
        },
        DifficultyTier.INTERMEDIATE: {
            FaultMode.TIMEOUT: 0.12,
            FaultMode.RATE_LIMIT: 0.08,
            FaultMode.STALE_DATA: 0.10,
            FaultMode.SILENT_FAIL: 0.06,
            FaultMode.PARTIAL_RESPONSE: 0.08,
        },
        DifficultyTier.EXPERT: {
            FaultMode.TIMEOUT: 0.15,
            FaultMode.RATE_LIMIT: 0.10,
            FaultMode.STALE_DATA: 0.15,
            FaultMode.SILENT_FAIL: 0.10,
            FaultMode.PARTIAL_RESPONSE: 0.12,
            FaultMode.CORRUPT_FIELD: 0.08,
        },
    }

    NEVER_FAIL_TOOLS = {
        "submit_answer",
        "calculator",
        "python_execute",
        "scratchpad_write",
        "scratchpad_read",
    }

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def inject_if_needed(
        self, tool_name: str, result: Mapping[str, Any], tier: DifficultyTier | str
    ) -> tuple[dict[str, Any], bool, FaultMode | None]:
        if tool_name in self.NEVER_FAIL_TOOLS:
            return dict(result), False, None

        tier_key = DifficultyTier(tier)
        probs = self.TIER_PROBS.get(tier_key, self.TIER_PROBS[DifficultyTier.WARMUP])
        roll = self.rng.random()

        cumulative = 0.0
        for mode, probability in probs.items():
            cumulative += probability
            if roll < cumulative:
                return self._apply_fault(mode, dict(result)), True, mode

        return dict(result), False, None

    def _apply_fault(self, mode: FaultMode, result: dict[str, Any]) -> dict[str, Any]:
        if mode == FaultMode.TIMEOUT:
            return {"error": "Timeout Error: the request exceeded maximum allowed time."}
        if mode == FaultMode.RATE_LIMIT:
            return {"error": "Rate Limit Exceeded: HTTP 429 Too Many Requests."}
        if mode == FaultMode.STALE_DATA:
            stale = copy.deepcopy(result)
            stale["warning"] = "stale_data"
            stale["message"] = "Response came from an older cached snapshot."
            return stale
        if mode == FaultMode.SILENT_FAIL:
            return {"results": []}
        if mode == FaultMode.PARTIAL_RESPONSE:
            return self._partial_response(result)
        if mode == FaultMode.CORRUPT_FIELD:
            return self._corrupt_first_scalar(result)
        return result

    @staticmethod
    def _partial_response(result: dict[str, Any]) -> dict[str, Any]:
        partial = copy.deepcopy(result)
        for key in ("result", "results"):
            value = partial.get(key)
            if isinstance(value, list) and len(value) > 1:
                partial[key] = value[:1]
                partial["warning"] = "partial_response"
                return partial
        partial["warning"] = "partial_response"
        return partial

    @classmethod
    def _corrupt_first_scalar(cls, result: dict[str, Any]) -> dict[str, Any]:
        corrupted = copy.deepcopy(result)
        changed = cls._mutate_first_scalar(corrupted)
        if changed:
            corrupted["warning"] = "corrupt_field"
        return corrupted

    @classmethod
    def _mutate_first_scalar(cls, value: Any) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, bool):
                    value[key] = not item
                    return True
                if isinstance(item, (int, float)):
                    value[key] = round(float(item) * 1.37, 4)
                    return True
                if isinstance(item, str):
                    value[key] = f"{item} [CORRUPTED]"
                    return True
                if cls._mutate_first_scalar(item):
                    return True
        if isinstance(value, list):
            for item in value:
                if cls._mutate_first_scalar(item):
                    return True
        return False
