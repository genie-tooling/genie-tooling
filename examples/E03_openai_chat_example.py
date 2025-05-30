# examples/E03_openai_chat_example.py
"""
Example: OpenAI Chat with Genie Facade
-------------------------------------
This example demonstrates how to use the Genie facade to chat with an
OpenAI LLM (e.g., gpt-3.5-turbo).

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Set your OpenAI API key as an environment variable:
   `export OPENAI_API_KEY="your_openai_api_key"`
3. Run from the root of the project:
   `poetry run python examples/E03_openai_chat_example.py`
"""
import asyncio
import logging
import os
from typing import List, Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import ChatMessage


async def run_openai_chat_demo():
    print("--- OpenAI Chat Example ---")

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set. Please set it to run this demo.")
        return

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="openai",
            llm_openai_model_name="gpt-3.5-turbo"
        )
        # KeyProvider defaults to EnvironmentKeyProvider, which will pick up OPENAI_API_KEY
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie with OpenAI...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        messages: List[ChatMessage] = [
            {"role": "user", "content": "Hello, OpenAI! What is the capital of France?"}
        ]
        print(f"\nSending to OpenAI: {messages[0]['content']}")

        response = await genie.llm.chat(messages) # Uses default provider (OpenAI)

        assistant_response = response.get("message", {}).get("content", "No content received.")
        print(f"\nOpenAI says: {assistant_response}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("OpenAI chat error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(run_openai_chat_demo())
