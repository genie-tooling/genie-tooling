### tests/unit/observability/test_decorators.py
import asyncio
import inspect
from typing import Any, Dict, Optional
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from genie_tooling.observability.decorators import traceable

# Mock the OTel trace API for testing without a real backend
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE_FOR_TEST = True
except ImportError:
    # Create mock objects if opentelemetry is not installed
    trace = MagicMock()
    Status = MagicMock()
    StatusCode = MagicMock()
    StatusCode.OK = "OK"
    StatusCode.ERROR = "ERROR"
    OTEL_AVAILABLE_FOR_TEST = False


@pytest.fixture
def mock_otel_tracer_components():
    """Mocks the core OpenTelemetry trace components used by the decorator."""
    if not OTEL_AVAILABLE_FOR_TEST:
        pytest.skip("opentelemetry-api is not installed, skipping decorator tests.")

    # We will patch the functions individually where they are looked up.
    with patch("genie_tooling.observability.decorators.trace.get_tracer") as mock_get_tracer, \
         patch("genie_tooling.observability.decorators.trace.set_span_in_context") as mock_set_span_in_context, \
         patch("genie_tooling.observability.decorators.trace.NonRecordingSpan") as mock_non_recording_span, \
         patch("genie_tooling.observability.decorators.Status", Status), \
         patch("genie_tooling.observability.decorators.StatusCode", StatusCode):

        # 1. Setup the mock span and its context manager
        mock_span_instance = MagicMock(spec=trace.Span)
        mock_span_instance.set_attribute = MagicMock()
        mock_span_instance.set_status = MagicMock()
        mock_span_instance.record_exception = MagicMock()

        mock_span_context_manager = MagicMock()
        mock_span_context_manager.__enter__.return_value = mock_span_instance
        # CORRECTED: __exit__ must return a falsy value (like None) to propagate exceptions.
        # Returning True would suppress the exception, causing tests to pass incorrectly.
        mock_span_context_manager.__exit__.return_value = None

        # 2. Setup the mock tracer
        mock_tracer_instance = MagicMock(spec=trace.Tracer)
        mock_tracer_instance.start_as_current_span.return_value = mock_span_context_manager

        # 3. Configure the patched get_tracer to return our mock tracer
        mock_get_tracer.return_value = mock_tracer_instance

        yield {
            "get_tracer": mock_get_tracer,
            "tracer": mock_tracer_instance,
            "span": mock_span_instance,
            "set_span_in_context": mock_set_span_in_context,
            "non_recording_span": mock_non_recording_span,
        }


@pytest.mark.asyncio
class TestTraceableDecorator:
    async def test_traceable_async_function_success(self, mock_otel_tracer_components):
        """Test that a successful async function is traced correctly."""
        mock_tracer = mock_otel_tracer_components["tracer"]
        mock_span = mock_otel_tracer_components["span"]

        @traceable
        async def my_async_func(arg1: str, arg2: int, context: Optional[Dict[str, Any]] = None):
            return f"{arg1}-{arg2}"

        result = await my_async_func("hello", 123)

        assert result == "hello-123"
        mock_tracer.start_as_current_span.assert_called_once_with("traceable.my_async_func", context=None)
        mock_span.set_attribute.assert_any_call("arg.arg1", "hello")
        mock_span.set_attribute.assert_any_call("arg.arg2", "123")
        
        # CORRECTED: Assert call count first, then inspect the call arguments.
        mock_span.set_status.assert_called_once()
        status_call_args = mock_span.set_status.call_args[0]
        assert isinstance(status_call_args[0], Status)
        assert status_call_args[0].status_code == StatusCode.OK
        
        mock_span.record_exception.assert_not_called()

    async def test_traceable_sync_function_success(self, mock_otel_tracer_components):
        """Test that a successful sync function is traced correctly."""
        mock_tracer = mock_otel_tracer_components["tracer"]
        mock_span = mock_otel_tracer_components["span"]

        @traceable
        def my_sync_func(param: bool, context: Optional[Dict[str, Any]] = None):
            return not param

        # The decorated function is now always async
        result = await my_sync_func(True)

        assert result is False
        mock_tracer.start_as_current_span.assert_called_once_with("traceable.my_sync_func", context=None)
        mock_span.set_attribute.assert_any_call("arg.param", "True")

        # CORRECTED: Assert call count first, then inspect the call arguments.
        mock_span.set_status.assert_called_once()
        status_call_args = mock_span.set_status.call_args[0]
        assert isinstance(status_call_args[0], Status)
        assert status_call_args[0].status_code == StatusCode.OK

    async def test_traceable_async_function_exception(self, mock_otel_tracer_components):
        """Test that an exception in an async function is recorded."""
        mock_tracer = mock_otel_tracer_components["tracer"]
        mock_span = mock_otel_tracer_components["span"]
        test_exception = ValueError("Test error")

        @traceable
        async def my_failing_func(context: Optional[Dict[str, Any]] = None):
            raise test_exception

        with pytest.raises(ValueError, match="Test error"):
            await my_failing_func()

        mock_tracer.start_as_current_span.assert_called_once()

        # CORRECTED: Assert call count first, then inspect the call arguments.
        mock_span.set_status.assert_called_once()
        status_call_args = mock_span.set_status.call_args[0]
        assert isinstance(status_call_args[0], Status)
        assert status_call_args[0].status_code == StatusCode.ERROR
        assert status_call_args[0].description == "Test error"

        mock_span.record_exception.assert_called_once_with(test_exception, attributes=ANY)

    async def test_traceable_with_parent_context(self, mock_otel_tracer_components):
        """Test that a parent OTel context is correctly used."""
        mock_tracer = mock_otel_tracer_components["tracer"]
        mock_set_span_in_context = mock_otel_tracer_components["set_span_in_context"]
        mock_parent_context_obj = MagicMock(name="MockParentOtelContextObject")

        @traceable
        async def my_child_func(context: Dict[str, Any]):
            return "done"

        await my_child_func(context={"otel_context": mock_parent_context_obj})

        # Verify that set_span_in_context was called with the parent context
        mock_set_span_in_context.assert_called_once()
        # Verify that start_as_current_span was called with the result of set_span_in_context
        mock_tracer.start_as_current_span.assert_called_once_with(
            "traceable.my_child_func", context=mock_set_span_in_context.return_value
        )

    async def test_traceable_unserializable_arg(self, mock_otel_tracer_components):
        """Test that unserializable arguments are handled gracefully."""
        mock_span = mock_otel_tracer_components["span"]
        class Unserializable:
            def __str__(self): raise TypeError("Cannot stringify")

        @traceable
        async def func_with_bad_arg(bad_arg: Any, context: Optional[Dict[str, Any]] = None):
            pass

        await func_with_bad_arg(Unserializable())
        mock_span.set_attribute.assert_called_once_with("arg.bad_arg", "[Unserializable]")

    async def test_traceable_no_context_arg_in_signature(self, mock_otel_tracer_components):
        """Test that the decorator works on functions without a 'context' argument."""
        mock_tracer = mock_otel_tracer_components["tracer"]
        mock_span = mock_otel_tracer_components["span"]

        @traceable
        async def func_no_context(x: int):
            return x * 2

        result = await func_no_context(5)

        assert result == 10
        mock_tracer.start_as_current_span.assert_called_once_with("traceable.func_no_context", context=None)
        mock_span.set_attribute.assert_called_once_with("arg.x", "5")
        
        # CORRECTED: Assert call count first, then inspect the call arguments.
        mock_span.set_status.assert_called_once()
        status_call_args = mock_span.set_status.call_args[0]
        assert isinstance(status_call_args[0], Status)
        assert status_call_args[0].status_code == StatusCode.OK