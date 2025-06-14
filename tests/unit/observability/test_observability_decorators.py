### tests/unit/observability/test_decorators.py
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.observability.decorators import traceable


@pytest.fixture()
def mock_otel_trace_module():
    """Fixture to mock the opentelemetry trace and status modules."""
    # FIX: Patch the specific objects where they are imported and used.
    with patch("genie_tooling.observability.decorators.trace") as mock_trace, \
         patch("genie_tooling.observability.decorators.Status") as mock_status_class, \
         patch("genie_tooling.observability.decorators.StatusCode") as mock_status_code_class:

        mock_span = MagicMock(name="MockSpan")
        mock_span.set_attribute = MagicMock()
        mock_span.set_status = MagicMock()
        mock_span.record_exception = MagicMock()

        mock_span_context_manager = MagicMock()
        mock_span_context_manager.__enter__.return_value = mock_span
        mock_span_context_manager.__exit__ = MagicMock(return_value=None)  # No exception suppression

        mock_tracer = MagicMock(name="MockTracer")
        mock_tracer.start_as_current_span.return_value = mock_span_context_manager

        mock_trace.get_tracer.return_value = mock_tracer
        mock_trace.set_span_in_context = MagicMock(side_effect=lambda span, context=None: "new_context_with_parent")
        mock_trace.NonRecordingSpan = MagicMock() # Mock the NonRecordingSpan class

        mock_status_class.return_value = "MockStatusInstance"

        yield {
            "trace": mock_trace,
            "tracer": mock_tracer,
            "span": mock_span,
            "status_class": mock_status_class,
            "status_code_class": mock_status_code_class,
        }


@pytest.mark.asyncio()
class TestTraceableDecoratorAsync:
    async def test_async_function_success(self, mock_otel_trace_module):
        """Test decorator on a successful async function."""

        @traceable
        async def my_async_func(a: int, b: str = "default"):
            return f"{b}-{a}"

        result = await my_async_func(1, b="test")
        assert result == "test-1"

        tracer = mock_otel_trace_module["tracer"]
        span = mock_otel_trace_module["span"]
        status_class = mock_otel_trace_module["status_class"]
        status_code_class = mock_otel_trace_module["status_code_class"]

        tracer.start_as_current_span.assert_called_once_with("traceable.my_async_func", context=None)
        span.set_attribute.assert_any_call("arg.a", "1")
        span.set_attribute.assert_any_call("arg.b", "test")
        status_class.assert_called_once_with(status_code_class.OK)
        span.set_status.assert_called_once_with("MockStatusInstance")
        span.record_exception.assert_not_called()

    async def test_async_function_exception(self, mock_otel_trace_module):
        """Test decorator when async function raises an exception."""

        @traceable
        async def my_failing_async_func():
            raise ValueError("Async fail")

        with pytest.raises(ValueError, match="Async fail"):
            await my_failing_async_func()

        span = mock_otel_trace_module["span"]
        status_class = mock_otel_trace_module["status_class"]
        status_code_class = mock_otel_trace_module["status_code_class"]

        status_class.assert_called_once_with(status_code_class.ERROR, description="Async fail")
        span.set_status.assert_called_once_with("MockStatusInstance")
        span.record_exception.assert_called_once()
        # Check that the first argument to record_exception is a ValueError instance
        assert isinstance(span.record_exception.call_args[0][0], ValueError)

    async def test_async_context_propagation(self, mock_otel_trace_module):
        """Test that OTel context is correctly extracted and used."""

        @traceable
        async def func_with_context(arg: str, context: dict):
            return arg

        mock_otel_context_obj = MagicMock(name="MockOtelContextObject")
        context_arg = {"otel_context": mock_otel_context_obj, "other_data": "test"}

        await func_with_context("test", context=context_arg)

        trace_module = mock_otel_trace_module["trace"]
        tracer = mock_otel_trace_module["tracer"]
        span = mock_otel_trace_module["span"]

        trace_module.set_span_in_context.assert_called_once()
        tracer.start_as_current_span.assert_called_once_with(
            "traceable.func_with_context", context="new_context_with_parent"
        )
        # Ensure the context argument itself is not added as a span attribute
        for call in span.set_attribute.call_args_list:
            assert call.args[0] != "arg.context"
        span.set_attribute.assert_called_with("arg.arg", "test")

    async def test_async_unserializable_arg(self, mock_otel_trace_module):
        """Test that unserializable arguments are handled gracefully."""
        class Unserializable:
            def __str__(self):
                raise TypeError("I cannot be a string")

        @traceable
        async def func_with_bad_arg(bad_arg: Unserializable):
            pass

        await func_with_bad_arg(Unserializable())

        span = mock_otel_trace_module["span"]
        span.set_attribute.assert_called_once_with("arg.bad_arg", "[Unserializable]")
