# examples/E04_gemini_chat_example.py
"""
Example: Gemini Chat with Genie Facade
-------------------------------------
This example demonstrates how to use the Genie facade to chat with a
Google Gemini LLM (e.g., gemini-1.5-flash-latest).

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras --extras gemini`).
2. Set your Google API key as an environment variable:
   `export GOOGLE_API_KEY="your_google_api_key"`
3. Run from the root of the project:
   `poetry run python examples/E04_gemini_chat_example.py`
"""
import asyncio
import logging
import os
from typing import List, Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import ChatMessage


async def run_gemini_chat_demo():
    print("--- Gemini Chat Example ---")

    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set. Please set it to run this demo.")
        return

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="gemini",
            llm_gemini_model_name="gemini-1.5-flash-latest"
        )
        # KeyProvider defaults to EnvironmentKeyProvider
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie with Gemini...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        messages: List[ChatMessage] = [
            {"role": "user", "content": "Hello, Gemini! Can you write a short poem about AI?"}
        ]
        print(f"\nSending to Gemini: {messages[0]['content']}")

        response = await genie.llm.chat(messages) # Uses default provider (Gemini)

        assistant_response = response.get("message", {}).get("content", "No content received.")
        print(f"\nGemini says: {assistant_response}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Gemini chat error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(run_gemini_chat_demo())
