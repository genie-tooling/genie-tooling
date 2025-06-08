"""Observability-related decorators."""
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

    This decorator inspects the decorated function's signature to find a 'context'
    dictionary. It extracts the parent OTel span context from `context['otel_context']`
    if available, creating a nested span. It records function arguments as span
    attributes and handles exceptions automatically.

    Args:
        func: The async or sync function to be decorated.

    Returns:
        The wrapped function.
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        tracer = trace.get_tracer(__name__)
        span_name = f"traceable.{func.__name__}"

        # Find the context dictionary from args/kwargs
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        context_dict: Optional[Dict[str, Any]] = bound_args.arguments.get("context")

        parent_context = None
        if isinstance(context_dict, dict):
            parent_otel_context_obj = context_dict.get("otel_context")
            if parent_otel_context_obj:
                parent_context = trace.set_span_in_context(trace.NonRecordingSpan(parent_otel_context_obj))

        with tracer.start_as_current_span(span_name, context=parent_context) as span:
            try:
                # Record arguments as attributes
                for name, value in bound_args.arguments.items():
                    if name != "context": # Don't record the whole context object
                        try:
                            span.set_attribute(f"arg.{name}", str(value))
                        except Exception:
                            span.set_attribute(f"arg.{name}", "[Unserializable]")

                result = await func(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, description=str(e)))
                span.record_exception(e, attributes={"exception.stacktrace": traceback.format_exc()})
                raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # This is a simplified sync wrapper. For production, you might want
        # to ensure it works correctly outside an async context if needed.
        # For Genie Tooling, most traceable functions will be async.
        tracer = trace.get_tracer(__name__)
        span_name = f"traceable.{func.__name__}"

        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        context_dict: Optional[Dict[str, Any]] = bound_args.arguments.get("context")

        parent_context = None
        if isinstance(context_dict, dict):
            parent_otel_context_obj = context_dict.get("otel_context")
            if parent_otel_context_obj:
                parent_context = trace.set_span_in_context(trace.NonRecordingSpan(parent_otel_context_obj))

        with tracer.start_as_current_span(span_name, context=parent_context) as span:
            try:
                for name, value in bound_args.arguments.items():
                    if name != "context":
                        span.set_attribute(f"arg.{name}", str(value))

                result = func(*args, **kwargs)
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, description=str(e)))
                span.record_exception(e, attributes={"exception.stacktrace": traceback.format_exc()})
                raise

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
