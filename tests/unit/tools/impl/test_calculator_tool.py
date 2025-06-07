"""Unit tests for the CalculatorTool."""
from typing import Optional

import pytest
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.impl.calculator import CalculatorTool


class MockCalcKeyProvider(KeyProvider):
    async def get_key(self, key_name: str) -> Optional[str]: return None
    async def setup(self,c=None): pass
    async def teardown(self): pass

@pytest.fixture
async def calculator_tool() -> CalculatorTool:
    tool = CalculatorTool()
    return tool

@pytest.fixture
async def mock_calc_key_provider() -> MockCalcKeyProvider:
    return MockCalcKeyProvider()

@pytest.mark.asyncio
async def test_calculator_tool_get_metadata(calculator_tool: CalculatorTool):
    actual_tool = await calculator_tool
    metadata = await actual_tool.get_metadata()
    assert metadata["identifier"] == "calculator_tool"

@pytest.mark.parametrize("num1, num2, operation, expected_result, expected_error", [
    (5, 3, "add", 8.0, None), (10, 0, "divide", None, "Cannot divide by zero."),
    (10, 5, "unknown_op", None, "Unknown operation: unknown_op. Supported operations are: add (+), subtract (-), multiply (*), divide (/)."),
    ("abc", 5, "add", None, "Invalid number input: could not convert string to float: 'abc'"),
])
@pytest.mark.asyncio
async def test_calculator_tool_execute(
    calculator_tool: CalculatorTool, mock_calc_key_provider: MockCalcKeyProvider,
    num1, num2, operation, expected_result, expected_error
):
    actual_tool = await calculator_tool
    actual_kp = await mock_calc_key_provider
    params = {"num1": num1, "num2": num2, "operation": operation}
    result_dict = await actual_tool.execute(params, actual_kp)
    assert result_dict["result"] == expected_result
    if expected_error:
        assert expected_error in result_dict["error_message"]
    else:
        assert result_dict["error_message"] is None

@pytest.mark.asyncio
async def test_calculator_tool_execute_missing_param(calculator_tool: CalculatorTool, mock_calc_key_provider: MockCalcKeyProvider):
    actual_tool = await calculator_tool
    actual_kp = await mock_calc_key_provider
    params = {"num1": 5, "operation": "add"}
    result_dict = await actual_tool.execute(params, actual_kp)
    assert result_dict["result"] is None
    assert "Missing required parameter" in result_dict["error_message"]
