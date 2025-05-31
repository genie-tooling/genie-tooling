# examples/E17_observability_tracing_example.py
"""
Example: Using Observability and Interaction Tracing (`genie.observability`)
---------------------------------------------------------------------------
This example demonstrates how to enable and use the interaction tracing
feature of Genie Tooling. It uses the `ConsoleTracerPlugin` by default,
which prints trace events to the console.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Run from the root of the project:
   `poetry run python examples/E17_observability_tracing_example.py`
   You should see trace logs in your console output.
"""
import asyncio
import logging
import uuid
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie

async def run_observability_demo():
    print("--- Observability and Tracing Example ---")
    
    # Configure basic logging to see standard logs AND trace logs from ConsoleTracerPlugin
    # The ConsoleTracerPlugin logs at the level specified in its config (default INFO)
    # via the "genie_tooling.observability.impl.console_tracer" logger.
    logging.basicConfig(level=logging.DEBUG) # Set root logger to DEBUG to catch all
    # For more fine-grained control:
    # logging.getLogger("genie_tooling.observability.impl.console_tracer").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling").setLevel(logging.INFO) # Genie's own operational logs

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", # To generate some LLM-related traces
            llm_ollama_model_name="mistral:latest",
            observability_tracer="console_tracer" # Enable console tracer
        ),
        observability_tracer_configurations={
            "console_tracer_plugin_v1": {
                "log_level": "INFO" # Traces will be logged at INFO level
            }
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie with ConsoleTracer...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized! Automatic traces will now be generated.")

        # --- Operations that generate automatic traces ---
        print("\n--- Performing operations that trigger automatic traces ---")
        
        # 1. LLM Chat (will generate llm.chat.start, llm.chat.success/error)
        try:
            await genie.llm.chat([{"role": "user", "content": "Hello Tracer!"}])
        except Exception as e_llm:
            print(f"LLM call failed (expected if Ollama not running): {e_llm}")


        # 2. Custom trace event
        print("\n--- Emitting a custom trace event ---")
        custom_correlation_id = str(uuid.uuid4())
        await genie.observability.trace_event(
            event_name="my_app.custom_operation.start",
            data={"user_id": "test_user", "input_param": "example_value"},
            component="MyApplicationLogic",
            correlation_id=custom_correlation_id
        )
        # Simulate some work
        await asyncio.sleep(0.1) 
        await genie.observability.trace_event(
            event_name="my_app.custom_operation.end",
            data={"status": "completed", "result_code": 0},
            component="MyApplicationLogic",
            correlation_id=custom_correlation_id
        )
        print("Custom trace events emitted.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Observability demo error details:")
    finally:
        if genie:
            # Teardown also generates a trace event
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    asyncio.run(run_observability_demo())
