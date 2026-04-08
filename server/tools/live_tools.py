from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import operator
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse


class LiveTools:
    """Deterministic per-episode implementations for pure utility tools."""

    LIVE_TOOL_NAMES = {
        "calculator",
        "python_execute",
        "scratchpad_write",
        "scratchpad_read",
        "text_extract",
        "data_transform",
        "compare_values",
        "json_query",
        "validate_data",
        "hash_verify",
        "create_report",
        "get_current_datetime",
        "validate_url",
        "write_file",
        "read_file",
        "database_insert",
        "send_notification",
        "schedule_task",
        "update_ticket",
        "request_human_review",
    }

    _BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    _MATH_NAMES: dict[str, Any] = {
        name: value for name, value in vars(math).items() if not name.startswith("_")
    }

    def __init__(self, *, now: datetime | None = None, python_timeout_s: float = 2.0):
        self.scratchpad: dict[str, Any] = {}
        self.virtual_files: dict[str, str] = {}
        self.virtual_tables: dict[str, list[dict[str, Any]]] = {}
        self.notifications: list[dict[str, Any]] = []
        self.scheduled_tasks: list[dict[str, Any]] = []
        self.tickets: dict[str, str] = {}
        self.human_review_requests: list[dict[str, Any]] = []
        self.now = now or datetime(2026, 4, 8, tzinfo=UTC)
        self.python_timeout_s = python_timeout_s

    @classmethod
    def is_live_tool(cls, tool_name: str) -> bool:
        return tool_name in cls.LIVE_TOOL_NAMES

    def handle(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown live tool: {tool_name}"}
        try:
            return handler(arguments)
        except Exception as exc:  # pragma: no cover - defensive boundary
            return {"error": f"{tool_name} failed: {exc}"}

    def _calculator(self, args: dict[str, Any]) -> dict[str, Any]:
        expression = str(args.get("expression", "")).strip()
        if not expression:
            return {"error": "Calculator expression is required"}

        try:
            tree = ast.parse(expression, mode="eval")
            result = self._eval_math_node(tree.body)
            return {"result": result}
        except Exception as exc:
            return {"error": f"Calculator Error: {exc}"}

    def _eval_math_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._BIN_OPS:
            return self._BIN_OPS[type(node.op)](
                self._eval_math_node(node.left), self._eval_math_node(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._UNARY_OPS:
            return self._UNARY_OPS[type(node.op)](self._eval_math_node(node.operand))
        if isinstance(node, ast.Name) and node.id in self._MATH_NAMES:
            value = self._MATH_NAMES[node.id]
            if isinstance(value, (int, float)):
                return value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = self._MATH_NAMES.get(node.func.id)
            if callable(fn):
                args = [self._eval_math_node(arg) for arg in node.args]
                return fn(*args)
        raise ValueError(f"unsupported expression: {ast.dump(node, include_attributes=False)}")

    def _python_execute(self, args: dict[str, Any]) -> dict[str, Any]:
        code = str(args.get("code", ""))
        if not code.strip():
            return {"error": "Python code is required", "status": "failed"}
        if len(code) > 4000:
            return {"error": "Python code exceeds 4000 characters", "status": "failed"}

        with TemporaryDirectory(prefix="chaosagent-python-") as temp_dir:
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", code],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.python_timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {
                    "error": f"Python execution exceeded {self.python_timeout_s:.1f}s",
                    "status": "timeout",
                }

        return {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "exit_code": completed.returncode,
            "status": "success" if completed.returncode == 0 else "failed",
        }

    def _scratchpad_write(self, args: dict[str, Any]) -> dict[str, Any]:
        key = str(args.get("key", "")).strip()
        if not key:
            return {"error": "scratchpad key is required"}
        self.scratchpad[key] = args.get("value")
        return {"status": "success", "key": key}

    def _scratchpad_read(self, args: dict[str, Any]) -> dict[str, Any]:
        key = str(args.get("key", "")).strip()
        if key not in self.scratchpad:
            return {"error": "Key not found"}
        return {"result": self.scratchpad[key], "key": key}

    def _text_extract(self, args: dict[str, Any]) -> dict[str, Any]:
        text = str(args.get("text", ""))
        pattern = str(args.get("pattern", ""))
        if not pattern:
            return {"error": "pattern is required"}
        try:
            matches = re.findall(pattern, text)
        except re.error as exc:
            return {"error": f"Invalid regex pattern: {exc}"}
        return {"matches": matches, "count": len(matches)}

    def _data_transform(self, args: dict[str, Any]) -> dict[str, Any]:
        data = self._coerce_json(args.get("data"))
        operation = str(args.get("operation", "")).strip().lower()
        if not operation:
            return {"error": "operation is required"}
        if not isinstance(data, list):
            return {"error": "data_transform expects a JSON array"}

        rows = data
        if operation == "count":
            return {"result": len(rows)}

        if ":" not in operation:
            return {"error": "unsupported operation"}
        op_name, field = operation.split(":", 1)
        field = field.strip()

        if op_name == "sort":
            return {"result": sorted(rows, key=lambda row: row.get(field))}
        if op_name in {"sum", "avg", "min", "max"}:
            values = [row.get(field) for row in rows if isinstance(row, dict)]
            numbers = [value for value in values if isinstance(value, (int, float))]
            if not numbers:
                return {"error": f"no numeric values found for field {field!r}"}
            if op_name == "sum":
                return {"result": sum(numbers)}
            if op_name == "avg":
                return {"result": sum(numbers) / len(numbers)}
            if op_name == "min":
                return {"result": min(numbers)}
            return {"result": max(numbers)}

        if op_name == "filter" and "=" in field:
            key, expected = [part.strip() for part in field.split("=", 1)]
            return {
                "result": [
                    row for row in rows if isinstance(row, dict) and str(row.get(key)) == expected
                ]
            }
        return {"error": "unsupported operation"}

    def _compare_values(self, args: dict[str, Any]) -> dict[str, Any]:
        left = args.get("value1", args.get("left"))
        right = args.get("value2", args.get("right"))
        tolerance = float(args.get("tolerance", 0.0) or 0.0)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            diff = abs(float(left) - float(right))
            relative_error = diff / max(abs(float(right)), 1e-9)
            return {
                "match": relative_error <= tolerance,
                "difference": diff,
                "relative_error": relative_error,
            }
        return {"match": left == right, "difference": None}

    def _json_query(self, args: dict[str, Any]) -> dict[str, Any]:
        data = self._coerce_json(args.get("data"))
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "query is required"}
        try:
            return {"result": self._query_path(data, query)}
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return {"error": f"json_query failed: {exc}"}

    def _validate_data(self, args: dict[str, Any]) -> dict[str, Any]:
        data = self._coerce_json(args.get("data"))
        schema = self._coerce_json(args.get("schema"))
        if not isinstance(schema, dict):
            return {"error": "schema must be a JSON object"}

        errors: list[str] = []
        required = schema.get("required", [])
        if isinstance(required, list) and isinstance(data, dict):
            for field_name in required:
                if field_name not in data:
                    errors.append(f"missing required field: {field_name}")

        properties = schema.get("properties", {})
        if isinstance(properties, dict) and isinstance(data, dict):
            for field_name, field_schema in properties.items():
                if field_name in data and isinstance(field_schema, dict):
                    expected = field_schema.get("type")
                    if expected and not self._matches_json_type(data[field_name], expected):
                        errors.append(f"{field_name} expected {expected}")

        return {"valid": not errors, "errors": errors}

    def _hash_verify(self, args: dict[str, Any]) -> dict[str, Any]:
        algorithm = str(args.get("algorithm", "sha256")).lower()
        data = str(args.get("data", ""))
        expected_hash = args.get("expected_hash")
        try:
            digest = hashlib.new(algorithm, data.encode("utf-8")).hexdigest()
        except ValueError:
            return {"error": f"unsupported hash algorithm: {algorithm}"}
        return {
            "algorithm": algorithm,
            "hash": digest,
            "match": None if expected_hash is None else digest == str(expected_hash).lower(),
        }

    def _create_report(self, args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("title", "Agent Report")).strip() or "Agent Report"
        content = str(args.get("content", "")).strip()
        findings = args.get("findings")
        report = {"title": title, "content": content, "findings": findings or []}
        return {"status": "success", "report": report}

    def _get_current_datetime(self, args: dict[str, Any]) -> dict[str, Any]:
        timezone = str(args.get("timezone", "UTC")).strip() or "UTC"
        return {
            "datetime": self.now.isoformat(),
            "timezone": timezone,
            "timestamp": int(self.now.timestamp()),
        }

    def _validate_url(self, args: dict[str, Any]) -> dict[str, Any]:
        url = str(args.get("url", "")).strip()
        parsed = urlparse(url)
        valid = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        return {"valid": valid, "scheme": parsed.scheme, "host": parsed.netloc}

    def _write_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._normalize_virtual_path(args.get("path"))
        if path is None:
            return {"error": "path is required"}
        content = str(args.get("content", ""))
        self.virtual_files[path] = content
        return {"status": "success", "path": path, "bytes": len(content.encode("utf-8"))}

    def _read_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self._normalize_virtual_path(args.get("path"))
        if path is None or path not in self.virtual_files:
            return {"error": "File not found"}
        return {"path": path, "content": self.virtual_files[path]}

    def _database_insert(self, args: dict[str, Any]) -> dict[str, Any]:
        table = str(args.get("table", "")).strip()
        row = self._coerce_json(args.get("data"))
        if not table:
            return {"error": "table is required"}
        if not isinstance(row, dict):
            return {"error": "data must be a JSON object"}
        self.virtual_tables.setdefault(table, []).append(row)
        return {"status": "success", "table": table, "rows": len(self.virtual_tables[table])}

    def _send_notification(self, args: dict[str, Any]) -> dict[str, Any]:
        message = str(args.get("message", "")).strip()
        if not message:
            return {"error": "message is required"}
        notification = {"message": message, "created_at": self.now.isoformat()}
        self.notifications.append(notification)
        return {"status": "queued", "notification": notification}

    def _schedule_task(self, args: dict[str, Any]) -> dict[str, Any]:
        task = str(args.get("task", "")).strip()
        scheduled_time = str(args.get("time", "")).strip()
        if not task or not scheduled_time:
            return {"error": "task and time are required"}
        record = {"task": task, "time": scheduled_time}
        self.scheduled_tasks.append(record)
        return {"status": "scheduled", "task": record}

    def _update_ticket(self, args: dict[str, Any]) -> dict[str, Any]:
        ticket_id = str(args.get("ticket_id", "")).strip()
        update = str(args.get("update", "")).strip()
        if not ticket_id or not update:
            return {"error": "ticket_id and update are required"}
        self.tickets[ticket_id] = update
        return {"status": "updated", "ticket_id": ticket_id}

    def _request_human_review(self, args: dict[str, Any]) -> dict[str, Any]:
        reason = str(args.get("reason", "")).strip()
        if not reason:
            return {"error": "reason is required"}
        request = {"reason": reason, "created_at": self.now.isoformat()}
        self.human_review_requests.append(request)
        return {"status": "review_requested", "request": request}

    @staticmethod
    def _coerce_json(value: Any) -> Any:
        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(value)
        return value

    @staticmethod
    def _query_path(data: Any, query: str) -> Any:
        current = data
        for raw_part in query.replace("[", ".").replace("]", "").split("."):
            part = raw_part.strip()
            if not part:
                continue
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                raise TypeError(f"cannot query through {type(current).__name__}")
        return current

    @staticmethod
    def _matches_json_type(value: Any, expected: str) -> bool:
        type_map: dict[str, type[Any] | tuple[type[Any], ...]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        expected_type = type_map.get(expected)
        return isinstance(value, expected_type) if expected_type is not None else True

    @staticmethod
    def _normalize_virtual_path(value: Any) -> str | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        path = PurePosixPath(raw.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            return None
        return str(path)
