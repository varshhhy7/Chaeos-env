from __future__ import annotations

import math
import re
from contextlib import suppress
from datetime import date
from typing import Any

from models import Fact, FactType, Scenario


class Grader:
    """Programmatic answer grader. No LLM calls and no runtime network access."""

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
