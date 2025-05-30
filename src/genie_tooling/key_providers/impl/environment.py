# src/genie_tooling/key_providers/impl/environment.py
import logging
import os
from typing import Any, Dict, Optional

from genie_tooling.core.types import Plugin
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

class EnvironmentKeyProvider(KeyProvider, Plugin):
    plugin_id: str = "environment_key_provider_v1"
    description: str = "Provides API keys by reading them from environment variables."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        logger.info(f"{self.plugin_id}: Initialized. Will read keys from environment variables.")

    async def get_key(self, key_name: str) -> Optional[str]:
        key_value = os.environ.get(key_name)
        if key_value:
            logger.debug(f"{self.plugin_id}: Retrieved key '{key_name}' from environment (exists).")
        else:
            logger.debug(f"{self.plugin_id}: Key '{key_name}' not found in environment variables.")
        return key_value

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Teardown complete.")
