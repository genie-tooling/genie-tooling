# examples/E02_ollama_chat_example.py
"""
Example: Ollama Chat with Genie Facade
--------------------------------------
This example demonstrates how to use the Genie facade to chat with an
Ollama-hosted LLM (e.g., Mistral).

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Ensure Ollama is running and the model 'mistral:latest' (or your chosen model) is pulled:
   `ollama serve`
   `ollama pull mistral`
3. Run from the root of the project:
   `poetry run python examples/E02_ollama_chat_example.py`
"""
import asyncio
import logging
from typing import List, Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import ChatMessage


async def run_ollama_chat_demo():
    print("--- Ollama Chat Example ---")

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="mistral:latest" # Or any other model you have pulled
        )
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie with Ollama...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        messages: List[ChatMessage] = [
            {"role": "user", "content": "Hello, Ollama! Tell me a fun fact about llamas."}
        ]
        print(f"\nSending to Ollama: {messages[0]['content']}")

        response = await genie.llm.chat(messages)
        # Default provider is Ollama as per FeatureSettings

        assistant_response = response.get("message", {}).get("content", "No content received.")
        print(f"\nOllama says: {assistant_response}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Ollama chat error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(run_ollama_chat_demo())
