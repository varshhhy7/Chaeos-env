import random

class FaultInjector:
    """Injects faults based on scenario tier probability."""
    
    TIER_PROBS = {
        "warmup":       {"TIMEOUT": 0.10, "RATE_LIMIT": 0.05, "STALE_DATA": 0.0, "SILENT_FAIL": 0.0, "PARTIAL_RESPONSE": 0.0, "CORRUPT_FIELD": 0.0},
        "beginner":     {"TIMEOUT": 0.15, "RATE_LIMIT": 0.10, "STALE_DATA": 0.10, "SILENT_FAIL": 0.0, "PARTIAL_RESPONSE": 0.0, "CORRUPT_FIELD": 0.0},
        "intermediate": {"TIMEOUT": 0.20, "RATE_LIMIT": 0.15, "STALE_DATA": 0.15, "SILENT_FAIL": 0.10, "PARTIAL_RESPONSE": 0.10, "CORRUPT_FIELD": 0.0},
        "expert":       {"TIMEOUT": 0.25, "RATE_LIMIT": 0.15, "STALE_DATA": 0.20, "SILENT_FAIL": 0.15, "PARTIAL_RESPONSE": 0.15, "CORRUPT_FIELD": 0.10},
    }

    NEVER_FAIL_TOOLS = {"submit_answer", "calculator", "python_execute", "scratchpad_write", "scratchpad_read"}

    def inject_if_needed(self, tool_name: str, result: dict, tier: str) -> tuple[dict, bool]:
        """Returns (modified_result, was_injected)"""
        if tool_name in self.NEVER_FAIL_TOOLS:
            return result, False
            
        probs = self.TIER_PROBS.get(tier, self.TIER_PROBS["warmup"])
        rand_val = random.random()
        
        cumulative = 0.0
        for mode, prob in probs.items():
            cumulative += prob
            if rand_val < cumulative:
                return self._apply_fault(mode, result), True
                
        return result, False

    def _apply_fault(self, mode: str, result: dict) -> dict:
        if mode == "TIMEOUT":
            return {"error": "Timeout Error: the request exceeded maximum allowed time."}
        elif mode == "RATE_LIMIT":
            return {"error": "Rate Limit Exceeded: HTTP 429 Too Many Requests."}
        elif mode == "STALE_DATA":
            return {"error": "Connection returned stale/cached response from an earlier period."} 
        elif mode == "SILENT_FAIL":
            return {"results": []}
        elif mode == "PARTIAL_RESPONSE":
            if isinstance(result, list) and len(result) > 1:
                return {"results": [result[0]]}
            elif isinstance(result, dict) and "result" in result and isinstance(result["result"], list) and len(result["result"]) > 1:
                return {"result": [result["result"][0]]}
            return result
        elif mode == "CORRUPT_FIELD":
            if isinstance(result, dict):
                corrupted = dict(result)
                if "result" in corrupted:
                    if isinstance(corrupted["result"], dict):
                        k = list(corrupted["result"].keys())[0] if corrupted["result"] else None
                        if k and isinstance(corrupted["result"][k], (int, float)):
                            corrupted["result"][k] = corrupted["result"][k] * 1.5
                        elif k and isinstance(corrupted["result"][k], str):
                            corrupted["result"][k] = corrupted["result"][k] + " [CORRUPTED]"
                return corrupted
            return result
