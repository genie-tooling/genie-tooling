"""Observability-related decorators."""
import asyncio
import functools
import inspect
import logging
import traceback
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)

def traceable(func: Callable) -> Callable:
    """
    A decorator to automatically create OpenTelemetry spans for function calls.

    This decorator wraps both sync and async functions in an awaitable wrapper.
    It inspects the decorated function's signature to find a 'context'
    dictionary and extracts the parent OTel span context if available. It records
    function arguments as span attributes and handles exceptions automatically.

    Args:
        func: The async or sync function to be decorated.

    Returns:
        An awaitable wrapped function.
    """
    is_original_func_async = inspect.iscoroutinefunction(func)

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        # Use the function's own module for better tracer naming
        tracer = trace.get_tracer(func.__module__ or __name__)
        span_name = f"traceable.{func.__name__}"

        # Find the context dictionary from args/kwargs
        sig = inspect.signature(func)
        try:
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            context_dict: Optional[Dict[str, Any]] = bound_args.arguments.get("context")
        except TypeError:
            bound_args = None
            context_dict = None

        parent_context = None
        if isinstance(context_dict, dict):
            parent_otel_context_obj = context_dict.get("otel_context")
            if parent_otel_context_obj:
                parent_context = trace.set_span_in_context(trace.NonRecordingSpan(parent_otel_context_obj))

        with tracer.start_as_current_span(span_name, context=parent_context) as span:
            try:
                if bound_args:
                    for name, value in bound_args.arguments.items():
                        if name != "context":
                            try:
                                span.set_attribute(f"arg.{name}", str(value))
                            except Exception:
                                span.set_attribute(f"arg.{name}", "[Unserializable]")

                # CORRECTED: Handle sync and async functions differently
                if is_original_func_async:
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, description=str(e)))
                span.record_exception(e, attributes={"exception.stacktrace": traceback.format_exc()})
                raise

    return async_wrapper