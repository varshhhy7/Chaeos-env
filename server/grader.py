import re
from typing import Any
import sys
import os

# Adjust path to import models since this will be in a subdirectory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Scenario, Fact

class Grader:
    """100% programmatic grading. No LLM judge."""
    
    def _extract_number(self, text: str, key: str) -> float | None:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if numbers:
            return float(numbers[0])
        return None

    def _text_match(self, text: str, expected: str, alternatives: list[str] | None) -> bool:
        text = text.lower()
        if expected.lower() in text:
            return True
        if alternatives:
            for alt in alternatives:
                if alt.lower() in text:
                    return True
        return False

    def _partial_text_match(self, text: str, expected: str) -> bool:
        return len(set(text.lower().split()) & set(expected.lower().split())) > 0

    def _boolean_match(self, text: str, expected: bool) -> bool:
        text = text.lower()
        if expected and ("true" in text or "yes" in text):
            return True
        if not expected and ("false" in text or "no" in text):
            return True
        return False
        
    def grade(self, submitted_answer: str, scenario: Scenario) -> float:
        """Returns 0.0 to 1.0 correctness score."""
        total_weight = 0.0
        earned_weight = 0.0
        
        if not scenario.required_facts:
            return 1.0 # If nothing to grade, then it's perfect by definition
            
        for fact in scenario.required_facts:
            weight = 1.0 
            total_weight += weight
            
            if fact.type == "numeric":
                extracted = self._extract_number(submitted_answer, fact.key)
                if extracted is not None:
                    relative_error = abs(extracted - fact.value) / max(abs(fact.value), 1e-9)
                    if relative_error <= fact.tolerance:
                        earned_weight += weight  
                    elif relative_error <= fact.tolerance * 3:
                        earned_weight += weight * 0.5  
                        
            elif fact.type == "text":
                if self._text_match(submitted_answer, fact.value, fact.alternatives):
                    earned_weight += weight
                elif self._partial_text_match(submitted_answer, str(fact.value)):
                    earned_weight += weight * 0.5
                    
            elif fact.type == "boolean":
                if self._boolean_match(submitted_answer, bool(fact.value)):
                    earned_weight += weight
        
        return earned_weight / max(total_weight, 1e-9)
