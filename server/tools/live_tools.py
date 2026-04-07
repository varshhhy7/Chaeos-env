import ast
from typing import Any, Dict
import io
import contextlib

class LiveTools:
    """Implementations for tools that always execute live (calculator, python, memory)."""
    
    def __init__(self):
        self.scratchpad: Dict[str, Any] = {}

    def handle(self, tool_name: str, arguments: dict) -> dict:
        if tool_name == "calculator":
            return self._calculator(arguments)
        elif tool_name == "python_execute":
            return self._python_execute(arguments)
        elif tool_name == "scratchpad_write":
            return self._scratchpad_write(arguments)
        elif tool_name == "scratchpad_read":
            return self._scratchpad_read(arguments)
        else:
            return {"error": f"Unknown live tool: {tool_name}"}

    def _calculator(self, args: dict) -> dict:
        expr = args.get("expression", "")
        try:
            # Using ast.literal_eval for utmost safety is not enough for arithmetic.
            # For a basic mock, eval() on simple arithmetic is done, but restricted.
            allowed_names = {"__builtins__": None}
            # Only allow basic math
            import math
            allowed_names.update({k: v for k, v in math.__dict__.items() if not k.startswith("_")})
            
            result = eval(expr, allowed_names, {})
            return {"result": result}
        except Exception as e:
            return {"error": f"Calculator Error: {str(e)}"}

    def _python_execute(self, args: dict) -> dict:
        code = args.get("code", "")
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                # Basic exec wrapper for Day 1 mock phase. Real agent env would use subprocess or docker exec.
                exec_globals = {}
                exec(code, exec_globals)
            return {"result": output.getvalue(), "status": "success"}
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def _scratchpad_write(self, args: dict) -> dict:
        k = args.get("key")
        v = args.get("value")
        self.scratchpad[k] = v
        return {"status": "success"}

    def _scratchpad_read(self, args: dict) -> dict:
        k = args.get("key")
        if k in self.scratchpad:
            return {"result": self.scratchpad[k]}
        return {"error": "Key not found"}
