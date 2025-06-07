# examples/E17_observability_tracing_example.py
"""
Example: Using Observability and Interaction Tracing (`genie.observability`)
---------------------------------------------------------------------------
This example demonstrates how to enable and use the interaction tracing
feature of Genie Tooling. It shows configuration for both ConsoleTracerPlugin
and the new OpenTelemetryTracerPlugin.

To Run with Console Tracer:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Run from the root of the project:
   `poetry run python examples/E17_observability_tracing_example.py`
   You should see trace logs in your console output.

To Run with OpenTelemetry Tracer (e.g., to Jaeger):
1. Ensure OTel dependencies are installed: `poetry install --extras observability`
2. Start an OTel collector (e.g., Jaeger all-in-one):
   `docker run -d --name jaeger -e COLLECTOR_OTLP_ENABLED=true -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/all-in-one:latest`
3. Modify the `app_config` below to use `otel_tracer` and configure the OTLP endpoint.
4. Run the script. Traces should appear in Jaeger UI (http://localhost:16686).
"""
import asyncio
import logging
import traceback  # For stacktrace example
import uuid
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def run_observability_demo():
    print("--- Observability and Tracing Example ---")

    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling.observability.impl.console_tracer").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling.observability.impl.otel_tracer").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling").setLevel(logging.INFO)

    app_config_console = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="mistral:latest",
            observability_tracer="console_tracer"
        ),
        observability_tracer_configurations={
            "console_tracer_plugin_v1": {
                "log_level": "INFO"
            }
        }
    )

    app_config_otel = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="mistral:latest",
            observability_tracer="otel_tracer",
            token_usage_recorder="otel_metrics_recorder" # Enable OTel metrics for token usage
        ),
        observability_tracer_configurations={
            "otel_tracer_plugin_v1": {
                "otel_service_name": "genie-e17-demo-app",
                "otel_service_version": "0.1.0",
                "exporter_type": "otlp_http",
                "otlp_http_endpoint": "http://localhost:4318/v1/traces",
                # "otlp_http_headers": "Authorization=Bearer mytoken,X-Custom-Header=value",
                # "otlp_http_timeout": 20,
                # For OTLP gRPC:
                # "exporter_type": "otlp_grpc",
                # "otlp_grpc_endpoint": "localhost:4317",
                # "otlp_grpc_insecure": True,
                # "otlp_grpc_timeout": 15,
                "resource_attributes": {"deployment.environment": "development_e17"}
            }
        }
    )

    app_config = app_config_console
    # app_config = app_config_otel # Uncomment to use OTel

    genie: Optional[Genie] = None
    try:
        print(f"\nInitializing Genie with tracer: {app_config.features.observability_tracer}...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized! Automatic traces will now be generated.")

        print("\n--- Performing operations that trigger automatic traces ---")
        try:
            chat_response = await genie.llm.chat([{"role": "user", "content": "Hello Tracer! Tell me a short story."}])
            print(f"LLM Response: {chat_response['message']['content'][:60]}...")
        except Exception as e_llm:
            print(f"LLM call failed (expected if Ollama not running): {e_llm}")

        print("\n--- Emitting a custom trace event ---")
        custom_correlation_id = str(uuid.uuid4())
        await genie.observability.trace_event(
            event_name="my_app.custom_operation.start",
            data={"user_id": "test_user", "input_param": "example_value"},
            component="MyApplicationLogic",
            correlation_id=custom_correlation_id
        )
        await asyncio.sleep(0.1)
        try:
            raise ValueError("Something went wrong in custom op!")
        except ValueError as e_custom:
            await genie.observability.trace_event(
                event_name="my_app.custom_operation.error",
                data={
                    "status": "failed",
                    "error_message": str(e_custom),
                    "error_type": type(e_custom).__name__,
                    "error_stacktrace": traceback.format_exc()
                    },
                component="MyApplicationLogic",
                correlation_id=custom_correlation_id
            )
        print("Custom trace events (including an error) emitted.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Observability demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    asyncio.run(run_observability_demo())
