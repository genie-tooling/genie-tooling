# examples/E20_token_usage_example.py
"""
Example: Using Token Usage Tracking (`genie.usage`)
--------------------------------------------------
This example demonstrates how to enable and use the token usage tracking
feature of Genie Tooling. It uses the `InMemoryTokenUsageRecorderPlugin`
by default.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Ensure Ollama is running and 'mistral:latest' is pulled (or configure a different LLM).
3. Run from the root of the project:
   `poetry run python examples/E20_token_usage_example.py`
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

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", 
            llm_ollama_model_name="mistral:latest",
            token_usage_recorder="in_memory_token_recorder" # Enable in-memory recorder
        )
        # No specific config needed for InMemoryTokenUsageRecorderPlugin by default
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie with token usage tracking...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        # --- Perform some LLM calls to generate usage data ---
        print("\n--- Making LLM calls ---")
        try:
            await genie.llm.chat([{"role": "user", "content": "Tell me a short story about a robot."}])
            print("First LLM chat call complete.")
            await genie.llm.generate("What is the capital of France?")
            print("Second LLM generate call complete.")
            # Make another call with a different (mock) model to see breakdown
            # Note: Ollama provider might not easily allow model switching per call like this
            # without specific provider options. This is illustrative.
            # For a real test with different models, you might use different provider_ids
            # or ensure your default provider supports model switching in options.
            # We'll assume the same model for simplicity here with Ollama.
            await genie.llm.chat([{"role": "user", "content": "Another question for the same model."}])
            print("Third LLM chat call complete.")

        except Exception as e_llm:
            print(f"LLM call failed (expected if Ollama not running): {e_llm}")
            print("Token usage summary might be empty or incomplete.")

        # --- Get and print the token usage summary ---
        print("\n--- Token Usage Summary ---")
        usage_summary = await genie.usage.get_summary()
        # For InMemoryTokenUsageRecorderPlugin, the summary is a dict with keys like:
        # 'total_records', 'total_prompt_tokens', 'total_completion_tokens', 
        # 'total_tokens_overall', 'by_model'
        print(json.dumps(usage_summary, indent=2))

        if usage_summary and usage_summary.get("by_model"):
            print("\nBreakdown by model:")
            for model_name, data in usage_summary["by_model"].items():
                print(f"  Model: {model_name}")
                print(f"    Calls: {data.get('count')}")
                print(f"    Prompt Tokens: {data.get('prompt')}")
                print(f"    Completion Tokens: {data.get('completion')}")
                print(f"    Total Tokens: {data.get('total')}")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Token usage demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    asyncio.run(run_token_usage_demo())
