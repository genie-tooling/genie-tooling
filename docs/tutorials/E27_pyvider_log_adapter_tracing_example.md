# Tutorial: Pyvider Log Adapter for Tracing Example (E27)

This tutorial corresponds to the example file `examples/E27_pyvider_log_adapter_tracing_example.py`.

It demonstrates how to:
- Configure Genie Tooling to use the `PyviderTelemetryLogAdapter`.
- Use the `ConsoleTracerPlugin` which will, in turn, use the configured `PyviderTelemetryLogAdapter` to process and output trace events.
- Observe how trace events are formatted by Pyvider (e.g., key-value or JSON, potentially with emojis).

## Prerequisites
- Genie Tooling installed with `pyvider-telemetry` support: `poetry install --extras pyvider_adapter`
- An LLM provider like Ollama running (if you want to see LLM operation traces).

## Example Code

```python
# examples/E27_pyvider_log_adapter_tracing_example.py
import asyncio
import logging
import traceback # For stacktrace example
import uuid
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def run_pyvider_observability_demo():
    print("--- Pyvider Log Adapter with Observability Tracing Example ---")

    logging.basicConfig(level=logging.INFO)

    app_config_pyvider = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", 
            llm_ollama_model_name="mistral:latest",
            
            logging_adapter="pyvider_log_adapter", 
            logging_pyvider_service_name="GeniePyviderDemo",

            observability_tracer="console_tracer" 
        ),
        log_adapter_configurations={
            "pyvider_telemetry_log_adapter_v1": { 
                "service_name": "GeniePyviderServiceExplicit", 
                "default_level": "DEBUG", 
                "console_formatter": "key_value", 
                "logger_name_emoji_prefix_enabled": True,
                "das_emoji_prefix_enabled": True,
                "omit_timestamp": False,
            }
        },
        observability_tracer_configurations={
            "console_tracer_plugin_v1": {}
        }
    )

    app_config = app_config_pyvider

    genie: Optional[Genie] = None
    try:
        print(f"\nInitializing Genie with LogAdapter: {app_config.features.logging_adapter} and Tracer: {app_config.features.observability_tracer}...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized! Automatic traces will now be generated and processed by PyviderTelemetryLogAdapter.")

        print("\n--- Performing operations that trigger automatic traces ---")
        try:
            chat_response = await genie.llm.chat([{"role": "user", "content": "Hello Pyvider Tracer! Tell me a short story."}])
            print(f"LLM Response (first 60 chars): {chat_response['message']['content'][:60]}...")
        except Exception as e_llm:
            print(f"LLM call failed (expected if Ollama not running): {e_llm}")
            await genie.observability.trace_event(
                "app.llm_call.error",
                {"error": str(e_llm), "type": type(e_llm).__name__},
                "PyviderDemoApp"
            )

        print("\n--- Emitting a custom trace event (will also go through Pyvider) ---")
        custom_correlation_id = str(uuid.uuid4())
        custom_event_data = {
            "user_id": "pyvider_user",
            "input_param": "example_value_for_pyvider",
            "status_override": "attempt" 
        }
        await genie.observability.trace_event(
            event_name="my_app.custom_pyvider_op.start",
            data=custom_event_data,
            component="MyApplicationLogic", 
            correlation_id=custom_correlation_id
        )
        await asyncio.sleep(0.1) 
        try:
            raise ValueError("Something went wrong in custom Pyvider op!")
        except ValueError as e_custom:
            await genie.observability.trace_event(
                event_name="my_app.custom_pyvider_op.error",
                data={
                    "status": "failed", 
                    "error_message": str(e_custom),
                    "error_type": type(e_custom).__name__,
                    "error_stacktrace": traceback.format_exc()
                    },
                component="MyApplicationLogic",
                correlation_id=custom_correlation_id
            )
        print("Custom trace events (including an error) emitted and processed by Pyvider.")

    except Exception as e:
        print(f"\nAn error occurred in the demo: {e}")
        logging.exception("Pyvider observability demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    asyncio.run(run_pyvider_observability_demo())
```

**Key Takeaways:**
- By setting `features.logging_adapter="pyvider_log_adapter"`, trace events captured by `ConsoleTracerPlugin` (or other future tracers designed to use the central `LogAdapter`) are processed and formatted by `PyviderTelemetryLogAdapter`.
- You can configure Pyvider's behavior (like `service_name`, `console_formatter`, emoji settings) through the `log_adapter_configurations` in `MiddlewareConfig`.
- This demonstrates how the `LogAdapterPlugin` system allows for different logging backends/formatters to be used for Genie's internal eventing.
