# examples/E12_custom_key_provider_example.py
"""
Example: Using a Custom KeyProvider with Genie
---------------------------------------------
This example demonstrates how to implement a custom KeyProvider and
use it with the Genie facade, for instance, to fetch an OpenAI API key.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Set an environment variable for your custom key provider to read:
   `export MY_APP_OPENAI_KEY="your_actual_openai_api_key"`
3. Run from the root of the project:
   `poetry run python examples/E12_custom_key_provider_example.py`
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.types import (
    Plugin as CorePluginType,  # For KeyProvider inheritance
)
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

# 1. Implement your application's KeyProvider
class MyAppCustomKeyProvider(KeyProvider, CorePluginType):
    plugin_id = "my_app_custom_key_provider_v1" # Must be unique if registered via entry points

    async def get_key(self, key_name: str) -> str | None:
        # This example reads from a specific env var, but could use a vault, etc.
        if key_name == "OPENAI_API_KEY": # The key name OpenAI LLM Provider will request
            return os.environ.get("MY_APP_OPENAI_KEY")
        # Add other key mappings as needed
        # Example: if key_name == "GOOGLE_API_KEY": return os.environ.get("MY_GOOGLE_KEY")
        logger.debug(f"[{self.plugin_id}] Requested key '{key_name}', not specifically handled by this provider.")
        return None # Fallback

    async def setup(self, config: Optional[Dict[str, Any]] = None):
        logger.info(f"[{self.plugin_id}] Custom KeyProvider setup.")
        # You could load config from a file here, initialize vault clients, etc.

    async def teardown(self):
        logger.info(f"[{self.plugin_id}] Custom KeyProvider teardown.")

async def run_custom_key_provider_demo():
    print("--- Custom KeyProvider Demo ---")

    if not os.getenv("MY_APP_OPENAI_KEY"):
        print("ERROR: MY_APP_OPENAI_KEY environment variable not set. Please set it to run this demo.")
        return

    # 2. Instantiate your custom KeyProvider
    my_key_provider = MyAppCustomKeyProvider()

    # 3. Configure MiddlewareConfig
    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="openai", # We want to use OpenAI, which requires an API key
            llm_openai_model_name="gpt-3.5-turbo"
        )
        # No key_provider_id needed in MiddlewareConfig if passing an instance
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie with custom KeyProvider...")
        # 4. Pass your KeyProvider instance to Genie.create()
        genie = await Genie.create(
            config=app_config,
            key_provider_instance=my_key_provider
        )
        print("Genie initialized!")

        # 5. Use a feature that requires the key (e.g., OpenAI chat)
        messages: List[ChatMessage] = [
            {"role": "user", "content": "Hello via custom KeyProvider! What's a fun fact?"}
        ]
        print(f"\nSending to OpenAI: {messages[0]['content']}")

        response = await genie.llm.chat(messages) # Uses default provider (OpenAI)

        assistant_response = response.get("message", {}).get("content", "No content received.")
        print(f"\nOpenAI (via custom KeyProvider) says: {assistant_response}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Custom KeyProvider demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(run_custom_key_provider_demo())
