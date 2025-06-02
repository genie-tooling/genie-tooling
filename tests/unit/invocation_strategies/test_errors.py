"""Unit tests for invocation.errors module."""
import logging
from typing import Any, Dict, Optional

import pytest

# Updated import paths
from genie_tooling.core.types import StructuredError
from genie_tooling.error_formatters.impl.json_formatter import JSONErrorFormatter
from genie_tooling.error_formatters.impl.llm_formatter import LLMErrorFormatter
from genie_tooling.error_handlers.impl.default_handler import DefaultErrorHandler
from genie_tooling.tools.abc import Tool


class MockToolForErrors(Tool):
    identifier: str = "mock_error_tool_v1"
    plugin_id: str = "mock_error_tool_v1"

    async def get_metadata(self) -> Dict[str, Any]:
        return {"identifier": self.identifier, "name": "Mock Error Tool"}
    async def execute(self, params: Dict[str, Any], key_provider: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        raise NotImplementedError("Execute not relevant for error handling tests of Tool properties")

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

@pytest.fixture
def default_error_handler() -> DefaultErrorHandler:
    return DefaultErrorHandler()

@pytest.fixture
def llm_error_formatter() -> LLMErrorFormatter:
    return LLMErrorFormatter()

@pytest.fixture
def json_error_formatter() -> JSONErrorFormatter:
    return JSONErrorFormatter()

@pytest.fixture
def mock_tool_instance() -> MockToolForErrors:
    return MockToolForErrors()

def test_default_error_handler_generic_exception(default_error_handler: DefaultErrorHandler, mock_tool_instance: MockToolForErrors):
    exc = Exception("Generic error")
    structured_error = default_error_handler.handle(exc, mock_tool_instance, None)

    assert structured_error["type"] == "Exception"
    assert structured_error["message"] == "Generic error"
    assert structured_error["details"]["tool_id"] == mock_tool_instance.identifier
    assert "builtins.Exception" in structured_error["details"]["exception_type"]

def test_default_error_handler_value_error(default_error_handler: DefaultErrorHandler, mock_tool_instance: MockToolForErrors):
    exc = ValueError("Specific value error")
    structured_error = default_error_handler.handle(exc, mock_tool_instance, {"ctx_key": "ctx_val"})

    assert structured_error["type"] == "ValueError"
    assert structured_error["message"] == "Specific value error"
    assert structured_error["details"]["error_category"] == "UsageError"

def test_default_error_handler_connection_error(default_error_handler: DefaultErrorHandler, mock_tool_instance: MockToolForErrors):
    exc = ConnectionError("Network issue")
    structured_error = default_error_handler.handle(exc, mock_tool_instance, None)

    assert structured_error["type"] == "ConnectionError"
    assert structured_error["details"]["error_category"] == "NetworkError"

def test_default_error_handler_tool_missing_identifier(default_error_handler: DefaultErrorHandler):
    class ToolWithoutIdentifier:
        pass
    exc = Exception("Test")
    tool_no_id = ToolWithoutIdentifier()
    structured_error = default_error_handler.handle(exc, tool_no_id, None)
    assert structured_error["details"]["tool_id"] == "unknown_tool_instance"

def test_llm_error_formatter_basic(llm_error_formatter: LLMErrorFormatter):
    err: StructuredError = {"type": "TestError", "message": "Something broke.", "details": {"tool_id": "tool123"}}
    formatted = llm_error_formatter.format(err)
    assert formatted == "Error executing tool 'tool123' (TestError): Something broke."

def test_llm_error_formatter_no_tool_id(llm_error_formatter: LLMErrorFormatter):
    err: StructuredError = {"type": "GlobalError", "message": "System issue."}
    formatted = llm_error_formatter.format(err)
    assert formatted == "Error (GlobalError): System issue."

def test_llm_error_formatter_with_suggestion(llm_error_formatter: LLMErrorFormatter):
    err: StructuredError = {
        "type": "InputError",
        "message": "Param missing.",
        "details": {"tool_id": "tool_abc"},
        "suggestion": "Provide the 'foo' parameter."
    }
    formatted = llm_error_formatter.format(err)
    assert formatted == "Error executing tool 'tool_abc' (InputError): Param missing. Suggestion: Provide the 'foo' parameter."

def test_llm_error_formatter_wrong_target_format(llm_error_formatter: LLMErrorFormatter, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    err: StructuredError = {"type": "TestError", "message": "Msg", "details": {"tool_id": "x"}}
    llm_error_formatter.format(err, target_format="json")
    assert "LLMErrorFormatter received target_format 'json', but only supports 'llm'." in caplog.text

def test_json_error_formatter_returns_dict(json_error_formatter: JSONErrorFormatter):
    err: StructuredError = {"type": "JSONError", "message": "For JSON.", "details": {"code": 500}}
    formatted = json_error_formatter.format(err)
    assert formatted == err

def test_json_error_formatter_wrong_target_format(json_error_formatter: JSONErrorFormatter, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    err: StructuredError = {"type": "TestError", "message": "Msg"}
    json_error_formatter.format(err, target_format="llm")
    assert "JSONErrorFormatter received target_format 'llm', but primarily returns JSON dict." in caplog.text

@pytest.mark.asyncio
async def test_default_error_handler_plugin_methods(default_error_handler: DefaultErrorHandler):
    assert default_error_handler.plugin_id == "default_error_handler_v1"
    await default_error_handler.setup()
    await default_error_handler.teardown()

@pytest.mark.asyncio
async def test_llm_error_formatter_plugin_methods(llm_error_formatter: LLMErrorFormatter):
    assert llm_error_formatter.plugin_id == "llm_error_formatter_v1"
    await llm_error_formatter.setup()
    await llm_error_formatter.teardown()

@pytest.mark.asyncio
async def test_json_error_formatter_plugin_methods(json_error_formatter: JSONErrorFormatter):
    assert json_error_formatter.plugin_id == "json_error_formatter_v1"
    await json_error_formatter.setup()
    await json_error_formatter.teardown()
