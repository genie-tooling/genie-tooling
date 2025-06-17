###tests/unit/tools/impl/test_symbolic_math_tool.py###

from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.tools.impl.symbolic_math_tool import (
    SYMPY_AVAILABLE,
    SymbolicMathTool,
)


@pytest.fixture()
def mock_key_provider():
    """Provides a mock KeyProvider for the tool's execute signature."""
    return MagicMock()

@pytest.fixture()
async def symbolic_tool() -> SymbolicMathTool:
    """Provides a setup instance of the SymbolicMathTool."""
    tool = SymbolicMathTool()
    await tool.setup()
    return tool

@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy library not installed")
@pytest.mark.asyncio()
class TestSymbolicMathToolExecution:
    async def test_simplify_success(self, symbolic_tool, mock_key_provider):
        tool = await symbolic_tool  # Await the fixture
        params = {"operation": "simplify", "expression": "(x + 1)**2 - (x**2 + 2*x)"}
        result = await tool.execute(params, mock_key_provider, {})
        assert result["error"] is None
        assert result["result"] == "1"

    async def test_expand_success(self, symbolic_tool, mock_key_provider):
        tool = await symbolic_tool  # Await the fixture
        params = {"operation": "expand", "expression": "(x + y)**2"}
        result = await tool.execute(params, mock_key_provider, {})
        assert result["error"] is None
        assert result["result"] == "x**2 + 2*x*y + y**2"

    async def test_check_equivalence_true(self, symbolic_tool, mock_key_provider):
        tool = await symbolic_tool  # Await the fixture
        params = {
            "operation": "check_equivalence",
            "expression": "cos(x)**2 + sin(x)**2",
            "expression2": "1",
        }
        result = await tool.execute(params, mock_key_provider, {})
        assert result["error"] is None
        assert result["is_equivalent"] is True

    async def test_check_equivalence_false(self, symbolic_tool, mock_key_provider):
        tool = await symbolic_tool  # Await the fixture
        params = {
            "operation": "check_equivalence",
            "expression": "x + 1",
            "expression2": "x + 2",
        }
        result = await tool.execute(params, mock_key_provider, {})
        assert result["error"] is None
        assert result["is_equivalent"] is False

    async def test_unknown_operation(self, symbolic_tool, mock_key_provider):
        tool = await symbolic_tool  # Await the fixture
        params = {"operation": "integrate", "expression": "x**2"}
        result = await tool.execute(params, mock_key_provider, {})
        assert "Unknown operation: integrate" in result["error"]

    async def test_missing_expression2_for_equivalence(self, symbolic_tool, mock_key_provider):
        tool = await symbolic_tool  # Await the fixture
        params = {"operation": "check_equivalence", "expression": "x+1"}
        result = await tool.execute(params, mock_key_provider, {})
        assert "'expression2' is required for check_equivalence" in result["error"]

    async def test_invalid_expression_syntax(self, symbolic_tool, mock_key_provider):
        tool = await symbolic_tool  # Await the fixture
        params = {"operation": "simplify", "expression": "(x + 1"}  # Missing parenthesis
        result = await tool.execute(params, mock_key_provider, {})
        assert "error" in result
        assert result["error"] is not None

@pytest.mark.asyncio()
async def test_sympy_not_available():
    """Test behavior when sympy library is not installed."""
    with patch("genie_tooling.tools.impl.symbolic_math_tool.SYMPY_AVAILABLE", False):
        tool = SymbolicMathTool()
        params = {"operation": "simplify", "expression": "x"}
        result = await tool.execute(params, MagicMock(), {})
        assert "SymPy library not installed" in result["error"]
