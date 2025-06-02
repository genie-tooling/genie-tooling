# examples/E20_token_usage_example.py
"""
Example: Using Token Usage Tracking (`genie.usage`)
--------------------------------------------------
This example demonstrates how to enable and use the token usage tracking
feature of Genie Tooling. It shows configuration for both the
`InMemoryTokenUsageRecorderPlugin` and the `OpenTelemetryMetricsTokenRecorderPlugin`.

To Run with InMemory Recorder:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Ensure Ollama is running and 'mistral:latest' is pulled (or configure a different LLM).
3. Run from the root of the project:
   `poetry run python examples/E20_token_usage_example.py`

To Run with OpenTelemetry Metrics Recorder (e.g., to Prometheus via OTel Collector):
1. Ensure OTel dependencies: `poetry install --extras observability`
2. Set up an OTel Collector configured to scrape Prometheus metrics and export them
   (e.g., to a Prometheus instance).
3. Modify `app_config` below to use `token_usage_recorder="otel_metrics_recorder"`
   and ensure `observability_tracer="otel_tracer"` is also configured for the OTel SDK to be initialized.
4. Run the script. Metrics should be available in your Prometheus/Grafana setup.
"""
import asyncio
import json
import logging
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def run_token_usage_demo():
    print("--- Token Usage Tracking Example ---")
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling.token_usage").setLevel(logging.DEBUG)

    # --- Config for InMemoryTokenUsageRecorder ---
    app_config_in_memory = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="mistral:latest",
            token_usage_recorder="in_memory_token_recorder"
        )
    )

    # --- Config for OpenTelemetryMetricsTokenRecorder ---
    # app_config_otel_metrics = MiddlewareConfig(
    #     features=FeatureSettings(
    #         llm="ollama",
    #         llm_ollama_model_name="mistral:latest",
    #         token_usage_recorder="otel_metrics_recorder",
    #         # OTel SDK needs to be initialized, typically by the tracer
    #         observability_tracer="otel_tracer" 
    #     ),
    #     observability_tracer_configurations={
    #         "otel_tracer_plugin_v1": {
    #             "otel_service_name": "genie-e20-token-metrics-app",
    #             "exporter_type": "console" # Or "otlp_http" to send to a collector
    #         }
    #     }
    # )
    
    # Select the configuration to use:
    app_config = app_config_in_memory
    # app_config = app_config_otel_metrics # Uncomment to use OTel Metrics

    genie: Optional[Genie] = None
    try:
        print(f"\nInitializing Genie with token usage recorder: {app_config.features.token_usage_recorder}...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        # --- Perform some LLM calls to generate usage data ---
        print("\n--- Making LLM calls ---")
        try:
            await genie.llm.chat([{"role": "user", "content": "Tell me a short story about a robot."}])
            print("First LLM chat call complete.")
            await genie.llm.generate("What is the capital of France?")
            print("Second LLM generate call complete.")
            await genie.llm.chat([{"role": "user", "content": "Another question for the same model."}])
            print("Third LLM chat call complete.")

        except Exception as e_llm:
            print(f"LLM call failed (expected if Ollama not running): {e_llm}")
            print("Token usage summary might be empty or incomplete.")

        # --- Get and print the token usage summary ---
        print("\n--- Token Usage Summary ---")
        # If using otel_metrics_recorder, get_summary will return a message saying it's not applicable.
        # Metrics would be viewed in your OTel backend (e.g., Prometheus/Grafana).
        usage_summary = await genie.usage.get_summary()
        
        print(json.dumps(usage_summary, indent=2))

        if app_config.features.token_usage_recorder == "in_memory_token_recorder" and usage_summary and usage_summary.get("by_model"):
            print("\nBreakdown by model (for in_memory_token_recorder):")
            for model_name, data in usage_summary["by_model"].items():
                print(f"  Model: {model_name}")
                print(f"    Calls: {data.get('count')}")
                print(f"    Prompt Tokens: {data.get('prompt')}")
                print(f"    Completion Tokens: {data.get('completion')}")
                print(f"    Total Tokens: {data.get('total')}")
        elif app_config.features.token_usage_recorder == "otel_metrics_recorder":
            print("\nFor 'otel_metrics_recorder', view metrics in your OpenTelemetry backend (e.g., Prometheus/Grafana).")
            print("Example Prometheus queries:")
            print("  - sum(rate(llm_request_tokens_prompt_total[5m])) by (llm_model_name)")
            print("  - sum(rate(llm_request_tokens_completion_total[5m])) by (llm_model_name)")


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Token usage demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    asyncio.run(run_token_usage_demo())
