from __future__ import annotations

import ast
import math
import operator
from datetime import datetime, timezone
from typing import Any

from agentic_rag.errors import ToolValidationError
from agentic_rag.models import ToolResult

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_LEN = 120
_MAX_EXP = 12


class CalculatorTool:
    name = "calculator"
    description = "Safely evaluate a numeric arithmetic expression."
    args_schema = {
        "type": "object",
        "required": ["expression"],
        "properties": {"expression": {"type": "string"}},
    }
    requires_credentials = False

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expr = str(arguments.get("expression", "")).strip()
        if not expr or len(expr) > _MAX_LEN:
            raise ToolValidationError("expression is required and must be short")
        forbidden = ("__", "import", "os.", "sys.", "eval", "exec", "open(")
        lowered = expr.lower()
        if any(token in lowered for token in forbidden):
            raise ToolValidationError("expression contains disallowed tokens")
        return {"expression": expr}

    def run(self, arguments: dict[str, Any], *, timeout_s: float = 5.0) -> ToolResult:
        args = self.validate(arguments)
        try:
            value = _eval_expr(args["expression"])
        except ZeroDivisionError:
            return ToolResult(
                tool_name=self.name,
                ok=False,
                data=None,
                error="Division by zero",
                retrieved_at=datetime.now(timezone.utc),
                source="ast-calculator",
            )
        except (SyntaxError, TypeError, ValueError, OverflowError) as exc:
            raise ToolValidationError(f"Invalid expression: {exc}") from exc
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data={"expression": args["expression"], "value": value},
            error=None,
            retrieved_at=datetime.now(timezone.utc),
            source="ast-calculator",
        )


def _eval_expr(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXP:
            raise ValueError("exponent too large")
        result = _BIN_OPS[type(node.op)](left, right)
        if not math.isfinite(result):
            raise OverflowError("non-finite result")
        return float(result)
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    raise ValueError("only numeric arithmetic is allowed")
