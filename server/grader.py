from __future__ import annotations

import math
import re
from contextlib import suppress
from datetime import date
from typing import Any

from models import ChaosAgentState, Fact, FactType, Scenario
from server.tasks import BenchmarkTask


class Grader:
    """Programmatic answer grader. No LLM calls and no runtime network access."""

    SCORE_MIN = 0.001
    SCORE_MAX = 0.990

    _NUMBER_RE = re.compile(
        r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*"
        r"(?:billion|million|thousand|bn|b|m|k)?",
        re.IGNORECASE,
    )
    _NEGATIVE_BOOL = {
        "false",
        "no",
        "not accurate",
        "inaccurate",
        "incorrect",
        "does not match",
        "doesn't match",
        "overstated",
        "understated",
    }
    _POSITIVE_BOOL = {
        "true",
        "yes",
        "accurate",
        "correct",
        "matches",
        "verified",
    }

    def grade(self, submitted_answer: str, scenario: Scenario) -> float:
        """Return a 0.0 to 1.0 correctness score."""
        if not scenario.required_facts:
            return 1.0

        earned_weight = 0.0
        total_weight = sum(fact.weight for fact in scenario.required_facts)
        for fact in scenario.required_facts:
            earned_weight += self._grade_fact(submitted_answer, fact) * fact.weight
        return earned_weight / max(total_weight, 1e-9)

    def grade_task(
        self,
        *,
        task: BenchmarkTask,
        scenario: Scenario,
        state: ChaosAgentState,
        correctness: float,
        answered: bool,
    ) -> float:
        """Return the benchmark task score in the open interval (0, 1)."""
        unique_tools = len(set(state.tools_called))
        diversity = min(unique_tools / max(task.target_tool_diversity, 1), 1.0)
        repeat_discipline = 1.0 - (state.repeat_calls / max(max(state.step_count - 1, 0), 1))
        repeat_discipline = max(0.0, min(1.0, repeat_discipline))
        efficiency = self._efficiency_score(
            step_count=state.step_count,
            max_steps=state.max_steps,
            ideal_steps=task.ideal_steps,
        )
        cross_validation = self.cross_validation_score(scenario, state)
        failure_recovery = self._failure_recovery_score(
            tool_failures=state.tool_failures_observed,
            correctness=correctness,
            diversity=diversity,
            repeat_discipline=repeat_discipline,
            answered=answered,
        )
        recovery_switch_score = self._target_score(
            actual=state.recovery_switches,
            target=task.target_recovery_switches,
        )
        verification_score = self._target_score(
            actual=state.verification_calls,
            target=task.target_verification_calls,
        )
        compute_score = self._target_score(
            actual=state.compute_calls,
            target=task.target_compute_calls,
        )
        artifact_score = self._target_score(
            actual=state.artifact_actions,
            target=task.target_artifact_actions,
        )
        retrieval_score = self._target_score(
            actual=state.retrieval_successes,
            target=max(task.target_tool_diversity - 1, 1),
        )

        if task.id == "task1":
            if state.tool_failures_observed > 0 or state.warning_events_observed > 0:
                resilience = 0.70 * recovery_switch_score + 0.30 * repeat_discipline
            else:
                resilience = 0.60 * diversity + 0.40 * repeat_discipline
            score = (
                0.55 * correctness + 0.20 * resilience + 0.15 * retrieval_score + 0.10 * efficiency
            )
        elif task.id == "task2":
            evidence = (
                0.40 * cross_validation
                + 0.25 * verification_score
                + 0.20 * compute_score
                + 0.15 * diversity
            )
            score = 0.55 * correctness + 0.30 * evidence + 0.15 * efficiency
        else:
            resilience = (
                0.25 * failure_recovery
                + 0.25 * recovery_switch_score
                + 0.20 * verification_score
                + 0.15 * artifact_score
                + 0.15 * repeat_discipline
            )
            score = (
                0.45 * correctness
                + 0.25 * resilience
                + 0.20 * cross_validation
                + 0.10 * compute_score
                + 0.10 * efficiency
            )

        if not answered:
            score *= 0.35

        return self._clamp_open_interval(score)

    def _grade_fact(self, submitted_answer: str, fact: Fact) -> float:
        if fact.type == FactType.NUMERIC:
            return self._grade_numeric(submitted_answer, fact)
        if fact.type == FactType.TEXT:
            return self._grade_text(submitted_answer, fact)
        if fact.type == FactType.BOOLEAN:
            return 1.0 if self._boolean_match(submitted_answer, bool(fact.value)) else 0.0
        if fact.type == FactType.DATE:
            return self._grade_date(submitted_answer, fact)
        return 0.0

    def _grade_numeric(self, submitted_answer: str, fact: Fact) -> float:
        expected = self._as_float(fact.value)
        if expected is None:
            return 0.0

        candidates = self._extract_numbers(submitted_answer)
        if not candidates:
            return 0.0

        best_relative_error = min(
            abs(candidate - expected) / max(abs(expected), 1e-9)
            for candidate in candidates
            if math.isfinite(candidate)
        )
        if best_relative_error <= fact.tolerance:
            return 1.0
        if fact.tolerance > 0 and best_relative_error <= fact.tolerance * 3:
            return 0.5
        return 0.0

    def _grade_text(self, submitted_answer: str, fact: Fact) -> float:
        normalized_answer = self._normalize_text(submitted_answer)
        expected_values = [str(fact.value), *fact.alternatives]
        if any(self._normalize_text(value) in normalized_answer for value in expected_values):
            return 1.0

        expected_tokens = self._tokens(str(fact.value))
        answer_tokens = self._tokens(submitted_answer)
        if expected_tokens and len(expected_tokens & answer_tokens) / len(expected_tokens) >= 0.5:
            return 0.5
        return 0.0

    def _grade_date(self, submitted_answer: str, fact: Fact) -> float:
        expected = str(fact.value)
        if expected in submitted_answer:
            return 1.0
        with suppress(ValueError):
            expected_date = date.fromisoformat(expected)
            long_date = f"{expected_date.strftime('%B')} {expected_date.day}, {expected_date.year}"
            if long_date in submitted_answer:
                return 1.0
        return 0.0

    def _boolean_match(self, submitted_answer: str, expected: bool) -> bool:
        answer = self._normalize_text(submitted_answer)
        if expected:
            return any(token in answer for token in self._POSITIVE_BOOL) and not any(
                token in answer for token in self._NEGATIVE_BOOL
            )
        return any(token in answer for token in self._NEGATIVE_BOOL)

    @classmethod
    def _extract_numbers(cls, text: str) -> list[float]:
        numbers: list[float] = []
        for match in cls._NUMBER_RE.finditer(text):
            raw = match.group(0).strip()
            if not raw:
                continue
            numbers.extend(cls._parse_number_variants(raw))
        return numbers

    @staticmethod
    def _parse_number_variants(raw: str) -> list[float]:
        lower = raw.lower().strip()
        cleaned = re.sub(r"[^\d.+-]", "", lower)
        base = float(cleaned.replace(",", ""))
        variants = [base]
        if lower.endswith(("billion", "bn", "b")):
            variants.append(base * 1_000_000_000.0)
        elif lower.endswith(("million", "m")):
            variants.append(base * 1_000_000.0)
        elif lower.endswith(("thousand", "k")):
            variants.append(base * 1_000.0)
        return variants

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(",", ""))
        except ValueError:
            return None

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9.]+", value.lower()))

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", value.lower()))

    @classmethod
    def _clamp_open_interval(cls, value: float) -> float:
        if not math.isfinite(value):
            return cls.SCORE_MIN
        return round(max(cls.SCORE_MIN, min(cls.SCORE_MAX, value)), 4)

    @staticmethod
    def _efficiency_score(*, step_count: int, max_steps: int, ideal_steps: int) -> float:
        if step_count <= 0:
            return 0.0
        if step_count <= ideal_steps:
            return 1.0
        penalty_window = max(max_steps - ideal_steps, 1)
        overflow = step_count - ideal_steps
        return max(0.0, 1.0 - (overflow / penalty_window))

    @staticmethod
    def cross_validation_score(scenario: Scenario, state: ChaosAgentState) -> float:
        if not scenario.cross_validation_tools:
            return 1.0
        unique_tools = set(state.tools_called)
        group_scores: list[float] = []
        for group in scenario.cross_validation_tools:
            if not group:
                group_scores.append(1.0)
                continue
            overlap = len(unique_tools & set(group))
            required = 2 if len(group) >= 2 else 1
            group_scores.append(min(overlap / required, 1.0))
        return sum(group_scores) / len(group_scores)

    @staticmethod
    def _target_score(*, actual: int, target: int) -> float:
        if target <= 0:
            return 1.0
        return min(actual / target, 1.0)

    @staticmethod
    def _failure_recovery_score(
        *,
        tool_failures: int,
        correctness: float,
        diversity: float,
        repeat_discipline: float,
        answered: bool,
    ) -> float:
        if not answered:
            return 0.0
        if tool_failures <= 0:
            return min(1.0, 0.55 * diversity + 0.45 * repeat_discipline)
        return min(
            1.0,
            0.45 * correctness + 0.30 * diversity + 0.25 * repeat_discipline,
        )
