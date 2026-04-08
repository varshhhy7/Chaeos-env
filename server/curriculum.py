from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from models import DifficultyTier


@dataclass
class CurriculumController:
    """Simple per-environment curriculum over fault difficulty tiers."""

    _tiers: tuple[DifficultyTier, ...] = (
        DifficultyTier.WARMUP,
        DifficultyTier.BEGINNER,
        DifficultyTier.INTERMEDIATE,
        DifficultyTier.EXPERT,
    )
    _tier_index: int = 0
    episodes_in_tier: int = 0
    recent_scores: deque[float] = field(default_factory=lambda: deque(maxlen=15))

    _advancement: dict[DifficultyTier, tuple[int, float] | None] = field(
        default_factory=lambda: {
            DifficultyTier.WARMUP: (5, 0.60),
            DifficultyTier.BEGINNER: (8, 0.50),
            DifficultyTier.INTERMEDIATE: (10, 0.40),
            DifficultyTier.EXPERT: None,
        }
    )

    @property
    def current_tier(self) -> DifficultyTier:
        return self._tiers[self._tier_index]

    @property
    def success_rate(self) -> float:
        if not self.recent_scores:
            return 0.0
        successes = sum(score > 0.5 for score in self.recent_scores)
        return successes / len(self.recent_scores)

    def record_episode(self, correctness: float) -> None:
        self.episodes_in_tier += 1
        self.recent_scores.append(correctness)

        requirement = self._advancement[self.current_tier]
        if requirement is None:
            return

        min_episodes, min_success_rate = requirement
        if self.episodes_in_tier >= min_episodes and self.success_rate >= min_success_rate:
            self._tier_index = min(self._tier_index + 1, len(self._tiers) - 1)
            self.episodes_in_tier = 0
