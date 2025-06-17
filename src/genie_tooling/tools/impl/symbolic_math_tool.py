###src/genie_tooling/tools/impl/symbolic_math_tool.py###
import asyncio
import logging
from typing import Any, Dict

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

try:
    from sympy import Eq, expand, simplify, sympify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    simplify, expand, sympify, Eq = None, None, None, None
    logger.warning(
        "SymbolicMathTool: 'sympy' library not installed. "
        "This tool will not be functional. Please install it: poetry add sympy"
    )

class SymbolicMathTool(Tool):
    identifier: str = "symbolic_math_tool"
    plugin_id: str = "symbolic_math_tool"

    def _run_sympy_op(self, func, *args, **kwargs):
        """Helper to run CPU-bound sympy operations and handle errors."""
        if not SYMPY_AVAILABLE:
            return {"result": None, "error": "SymPy library not installed."}
        try:
            # Safely convert string inputs to sympy objects
            safe_args = [sympify(arg) for arg in args]
            result = func(*safe_args, **kwargs)
            return {"result": str(result), "error": None}
        except Exception as e:
            logger.error(f"SymPy error in {func.__name__}: {e}", exc_info=True)
            return {"result": None, "error": str(e)}

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Symbolic Math Engine (SymPy)",
            "description_human": "Performs symbolic math operations like simplifying, expanding, and checking expression equivalence using the SymPy library.",
            "description_llm": "Performs symbolic math operations. Args: operation (enum['simplify', 'expand', 'check_equivalence']), expression (str), expression2 (str, for equivalence check). Returns a dict with 'result' and 'error'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The symbolic operation to perform.",
                        "enum": ["simplify", "expand", "check_equivalence"],
                    },
                    "expression": {
                        "type": "string",
                        "description": "The primary mathematical expression as a string (e.g., 'x**2 + 2*x + 1').",
                    },
                    "expression2": {
                        "type": ["string", "null"],
                        "description": "The second expression, required only for 'check_equivalence'.",
                    },
                },
                "required": ["operation", "expression"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "result": {"type": ["string", "null"]},
                    "is_equivalent": {"type": ["boolean", "null"]},
                    "error": {"type": ["string", "null"]},
                },
            },
            "key_requirements": [],
            "tags": ["math", "symbolic", "algebra", "proof_assistant"],
            "version": "1.0.0",
            "cacheable": True,
        }

    async def execute(self, params: Dict[str, Any], key_provider: KeyProvider, context: Dict[str, Any]) -> Dict[str, Any]:
        operation = params.get("operation")
        expression = params.get("expression")
        if not operation or not expression:
            return {"result": None, "error": "Missing 'operation' or 'expression' parameter."}

        loop = asyncio.get_running_loop()

        if operation == "simplify":
            return await loop.run_in_executor(None, self._run_sympy_op, simplify, expression)

        elif operation == "expand":
            return await loop.run_in_executor(None, self._run_sympy_op, expand, expression)

        elif operation == "check_equivalence":
            expression2 = params.get("expression2")
            if not expression2:
                return {"result": None, "error": "'expression2' is required for check_equivalence."}

            def check_eq_op():
                if not SYMPY_AVAILABLE:
                    return {"result": None, "error": "SymPy library not installed."}
                try:
                    eq = Eq(sympify(expression), sympify(expression2))
                    simplified_result = simplify(eq)

                    # not the Python boolean `True`. Use `bool()` to cast it correctly.
                    is_equiv = bool(simplified_result)
                    return {"result": str(simplified_result), "is_equivalent": is_equiv, "error": None}
                except Exception as e:
                    logger.error(f"SymPy error in check_equivalence: {e}", exc_info=True)
                    return {"result": None, "is_equivalent": None, "error": str(e)}

            return await loop.run_in_executor(None, check_eq_op)

        else:
            return {"result": None, "error": f"Unknown operation: {operation}"}
