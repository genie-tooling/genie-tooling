# examples/E27_pyvider_log_adapter_tracing_example.py
"""
Example: Using PyviderTelemetryLogAdapter for Observability Tracing
--------------------------------------------------------------------
This example demonstrates how to configure Genie Tooling to use the
PyviderTelemetryLogAdapter for processing and outputting interaction traces.
It builds upon the concepts shown in E17 (Observability & Tracing).

When PyviderTelemetryLogAdapter is active and ConsoleTracerPlugin is used,
the trace events will be formatted by Pyvider (e.g., key-value or JSON,
potentially with emojis) instead of the ConsoleTracerPlugin's default format.

To Run:
1. Ensure Genie Tooling is installed with Pyvider support:
   `poetry install --all-extras` or `poetry install --extras pyvider_adapter`
2. Ensure `pyvider-telemetry` library is installed (it's an optional dependency).
3. Run from the root of the project:
   `poetry run python examples/E27_pyvider_log_adapter_tracing_example.py`
   Observe the console output; it should be formatted by Pyvider.
"""
import asyncio
import logging
import traceback  # For stacktrace example
import uuid
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def run_pyvider_observability_demo():
    print("--- Pyvider Log Adapter with Observability Tracing Example ---")

    # Configure logging for the application itself (not Pyvider's internal config here)
    # Pyvider will configure its own handlers based on its setup.
    logging.basicConfig(level=logging.INFO)
    # To see Genie's internal debug logs (not Pyvider formatted unless Pyvider is set to DEBUG for genie_tooling):
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)


    # Configuration using PyviderTelemetryLogAdapter
    app_config_pyvider = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", # For an operation that generates traces
            llm_ollama_model_name="mistral:latest",

            logging_adapter="pyvider_log_adapter", # Select Pyvider
            logging_pyvider_service_name="GeniePyviderDemo", # Service name for Pyvider

            observability_tracer="console_tracer" # ConsoleTracer will use the configured LogAdapter
        ),
        # Configuration for PyviderTelemetryLogAdapter
        log_adapter_configurations={
            "pyvider_telemetry_log_adapter_v1": { # Canonical ID
                "service_name": "GeniePyviderServiceExplicit", # Overrides feature setting
                "default_level": "DEBUG", # Pyvider's default log level for its logger
                "console_formatter": "key_value", # "key_value" or "json"
                "logger_name_emoji_prefix_enabled": True,
                "das_emoji_prefix_enabled": True,
                "omit_timestamp": False,
                # Example: Configure redactor for Pyvider adapter
                # "redactor_plugin_id": "schema_aware_redactor_v1",
                # "redactor_config": {"redact_matching_key_names": True}
            }
        },
        # ConsoleTracerPlugin itself doesn't need much config when delegating
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
            # This LLM call will generate trace events (e.g., llm.chat.start, llm.chat.success)
            # which ConsoleTracerPlugin will pass to PyviderTelemetryLogAdapter.
            chat_response = await genie.llm.chat([{"role": "user", "content": "Hello Pyvider Tracer! Tell me a short story."}])
            print(f"LLM Response (first 60 chars): {chat_response['message']['content'][:60]}...")
        except Exception as e_llm:
            # This error will also be traced.
            print(f"LLM call failed (expected if Ollama not running): {e_llm}")
            # Manually trace the error if needed, though Genie's LLMInterface might do it.
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
            "status_override": "attempt" # For DAS mapping in Pyvider adapter
        }
        await genie.observability.trace_event(
            event_name="my_app.custom_pyvider_op.start",
            data=custom_event_data,
            component="MyApplicationLogic", # This might become 'domain' in Pyvider
            correlation_id=custom_correlation_id
        )
        await asyncio.sleep(0.1) # Simulate work
        try:
            raise ValueError("Something went wrong in custom Pyvider op!")
        except ValueError as e_custom:
            await genie.observability.trace_event(
                event_name="my_app.custom_pyvider_op.error",
                data={
                    "status": "failed", # Will be picked up by Pyvider adapter for DAS
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
    # Note: Pyvider's setup_telemetry might configure the root logger.
    # If you want to control formatting for non-Genie logs, configure Python's
    # root logger before Genie/Pyvider setup, or adjust Pyvider's config.
    asyncio.run(run_pyvider_observability_demo())
