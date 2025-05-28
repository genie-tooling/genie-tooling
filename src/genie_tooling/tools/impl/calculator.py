"""CalculatorTool: A simple tool for performing arithmetic calculations."""
import logging
import operator
from typing import Any, Dict, Optional, Union

from genie_tooling.security.key_provider import (
    KeyProvider,  # Unused but part of interface
)
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

class CalculatorTool(Tool):
    identifier: str = "calculator_tool"
    plugin_id: str = "calculator_tool" # Matches identifier for consistency

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Calculator",
            "description_human": "Performs basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/).",
            "description_llm": "Calculator: Solves math expressions. Args: num1 (float), num2 (float), operation (str: 'add', 'subtract', 'multiply', 'divide'). Example: num1=5, num2=3, operation='add' results in 8.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "num1": {"type": "number", "description": "The first number in the operation."},
                    "num2": {"type": "number", "description": "The second number in the operation."},
                    "operation": {
                        "type": "string",
                        "description": "The arithmetic operation to perform.",
                        "enum": ["add", "subtract", "multiply", "divide", "+", "-", "*", "/"]
                    }
                },
                "required": ["num1", "num2", "operation"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "result": {"type": ["number", "null"], "description": "The numerical result of the calculation. Null if an error occurred."},
                    "error_message": {"type": ["string", "null"], "description": "An error message if the calculation failed, otherwise null."}
                },
                "required": ["result", "error_message"]
            },
            "key_requirements": [], # No API keys needed
            "tags": ["math", "calculator", "arithmetic", "computation"],
            "version": "1.1.0", # Indicate functional version
            "cacheable": True, # Math operations are deterministic
            "cache_ttl_seconds": 3600 * 24 # Cache for a day (long TTL for deterministic results)
        }

    async def execute(
        self,
        params: Dict[str, Any],
        key_provider: KeyProvider, # Unused but part of the Tool interface
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[float, str, None]]:
        try:
            num1 = float(params["num1"])
            num2 = float(params["num2"])
            operation_str = params["operation"].lower().strip()

            op_map = {
                "add": operator.add, "+": operator.add,
                "subtract": operator.sub, "-": operator.sub,
                "multiply": operator.mul, "*": operator.mul,
                "divide": operator.truediv, "/": operator.truediv,
            }

            if operation_str not in op_map:
                logger.warning(f"CalculatorTool: Unknown operation '{operation_str}'.")
                return {"result": None, "error_message": f"Unknown operation: {params['operation']}. Supported operations are: add (+), subtract (-), multiply (*), divide (/)."}

            selected_op = op_map[operation_str]

            if selected_op == operator.truediv and num2 == 0:
                logger.warning("CalculatorTool: Division by zero attempt.")
                return {"result": None, "error_message": "Cannot divide by zero."}

            result = selected_op(num1, num2)
            logger.info(f"CalculatorTool: Executed {num1} {operation_str} {num2} = {result}")
            return {"result": result, "error_message": None}

        except ValueError as ve: # Handle issues with float conversion
            logger.error(f"CalculatorTool: Invalid number input. Params: {params}. Error: {ve}", exc_info=True)
            return {"result": None, "error_message": f"Invalid number input: {ve}. Both num1 and num2 must be valid numbers."}
        except KeyError as ke: # Handle missing required parameters
            logger.error(f"CalculatorTool: Missing required parameter. Params: {params}. Error: {ke}", exc_info=True)
            return {"result": None, "error_message": f"Missing required parameter: {ke}. Required: num1, num2, operation."}
        except Exception as e:
            logger.error(f"CalculatorTool: Unexpected error during calculation. Params: {params}. Error: {e}", exc_info=True)
            return {"result": None, "error_message": f"An unexpected error occurred: {str(e)}"}

    # setup and teardown are inherited from Plugin protocol and can be pass if not needed
    # async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    # async def teardown(self) -> None: pass
