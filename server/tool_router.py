from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from models import Scenario
from server.task_workspace import TaskWorkspace
from server.tools.live_tools import LiveTools
from server.tools.registry import is_known_tool


class ToolRouter:
    """Routes tool calls to workspace-backed systems with scenario fallbacks."""

    QUERY_ARGUMENT_KEYS = (
        "query",
        "sql",
        "entity",
        "url",
        "claim",
        "path",
        "table",
        "ticket_id",
    )
    ENTRY_MATCH_KEYS = (
        "query",
        "sql",
        "entity",
        "url",
        "claim",
        "path",
        "table",
        "ticket_id",
    )

    def route(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        scenario: Scenario,
        workspace: TaskWorkspace | None = None,
    ) -> dict[str, Any]:
        if not is_known_tool(tool_name):
            return {"error": f"Unknown tool: {tool_name}"}

        workspace_result = self._route_workspace_tool(tool_name, arguments, workspace)
        if workspace_result is not None:
            return workspace_result

        tool_entries = scenario.tool_data.get(tool_name)
        if tool_entries == "always_live" or (
            tool_entries is None and LiveTools.is_live_tool(tool_name)
        ):
            return {"_directive": "always_live"}

        if not tool_entries:
            return self._empty_result(tool_name)

        if isinstance(tool_entries, Sequence) and not isinstance(
            tool_entries, (str, bytes, bytearray)
        ):
            entries = [entry for entry in tool_entries if isinstance(entry, Mapping)]
            if not entries:
                return self._empty_result(tool_name)

            query_text = self._argument_text(arguments)
            if query_text:
                best_match = self._fuzzy_match(query_text, entries)
                if best_match is not None:
                    return self._extract_result(best_match)
                return self._empty_result(tool_name)
            return self._extract_result(entries[0])

        if isinstance(tool_entries, Mapping):
            return self._extract_result(tool_entries)

        return {"result": tool_entries}

    def _route_workspace_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        workspace: TaskWorkspace | None,
    ) -> dict[str, Any] | None:
        if workspace is None:
            return None

        if tool_name == "knowledge_base_lookup":
            return workspace.lookup_entity(str(arguments.get("entity", "")))
        if tool_name == "database_query":
            return workspace.run_sql(str(arguments.get("sql", "")))
        if tool_name == "document_search":
            return workspace.search_documents(str(arguments.get("query", "")))
        if tool_name == "web_search":
            return workspace.web_search(str(arguments.get("query", "")))
        if tool_name == "fetch_url":
            return workspace.fetch_url(str(arguments.get("url", "")))
        if tool_name == "fact_check":
            return workspace.fact_check(str(arguments.get("claim", "")))
        if tool_name == "check_consistency":
            return workspace.check_consistency(
                str(arguments.get("source1", "")),
                str(arguments.get("source2", "")),
            )
        if tool_name == "api_call":
            return workspace.api_call(
                str(arguments.get("url", "")),
                str(arguments.get("method", "GET")),
            )
        return None

    def _argument_text(self, arguments: Mapping[str, Any]) -> str:
        return " ".join(
            str(arguments[key])
            for key in self.QUERY_ARGUMENT_KEYS
            if key in arguments and arguments[key] is not None
        )

    def _fuzzy_match(
        self, query: str, entries: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any] | None:
        query_tokens = self._tokens(query)
        best_score = 0.0
        best_entry: Mapping[str, Any] | None = None

        for entry in entries:
            candidate_text = " ".join(
                str(entry[key])
                for key in self.ENTRY_MATCH_KEYS
                if key in entry and entry[key] is not None
            )
            candidate_tokens = self._tokens(candidate_text)
            if not query_tokens or not candidate_tokens:
                continue
            overlap = len(query_tokens & candidate_tokens)
            score = overlap / max(len(query_tokens | candidate_tokens), 1)
            if score > best_score:
                best_score = score
                best_entry = entry

        return best_entry if best_score >= 0.20 else None

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_]+", value.lower()))

    @staticmethod
    def _extract_result(entry: Mapping[str, Any]) -> dict[str, Any]:
        result = entry.get("result", entry)
        if isinstance(result, Mapping):
            return dict(result)
        return {"result": result}

    @staticmethod
    def _empty_result(tool_name: str) -> dict[str, Any]:
        if tool_name == "database_query":
            return {"error": "No matching table or row found for this query."}
        return {"results": [], "message": "No relevant data found"}
