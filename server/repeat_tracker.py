from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def _stable_arguments(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return repr(sorted(arguments.items()))


@dataclass
class RepeatTracker:
    """Tracks repeated identical tool calls within one episode."""

    counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    total_repeats: int = 0

    def log_call(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        signature = (tool_name, _stable_arguments(arguments))
        self.counts[signature] += 1
        if self.counts[signature] == 1:
            return None

        self.total_repeats += 1
        if self.counts[signature] >= 3:
            return (
                f"Repeated identical call to {tool_name}; consider using a different "
                "tool or cross-checking another source."
            )
        return None
