import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Scenario

class ToolRouter:
    """Routes tool calls to pre-computed scenario data."""
    
    def route(self, tool_name: str, arguments: dict, scenario: Scenario) -> dict | None:
        """
        Look up the pre-computed tool output for this call.
        Returns None if the tool has no relevant data for this scenario.
        """
        tool_entries = scenario.tool_data.get(tool_name, [])
        
        if tool_entries == "always_live":
            return {"_directive": "always_live"}
            
        if not tool_entries:
            return {"results": [], "message": "No relevant data found"}
        
        if "query" in arguments or "sql" in arguments:
            query = arguments.get("query") or arguments.get("sql", "")
            best_match = self._fuzzy_match(query, tool_entries)
            if best_match:
                return best_match.get("result", {}) if "result" in best_match else best_match
                
        return tool_entries[0].get("result") if isinstance(tool_entries, list) and "result" in tool_entries[0] else tool_entries
    
    def _fuzzy_match(self, query: str, entries: list) -> dict | None:
        if not isinstance(entries, list):
            return None
            
        query_words = set(query.lower().split())
        best_score = 0.0
        best_entry = None
        for entry in entries:
            entry_words = set(entry.get("query", "").lower().split())
            if not query_words and not entry_words:
                continue
            overlap = len(query_words & entry_words) / max(len(query_words | entry_words), 1)
            if overlap > best_score:
                best_score = overlap
                best_entry = entry
        return best_entry if best_score > 0.3 else None
