# src/genie_tooling/key_providers/impl/environment.py
import logging
import os
from typing import Any, Dict, Optional

from genie_tooling.core.types import Plugin
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

class EnvironmentKeyProvider(KeyProvider, Plugin):
    plugin_id: str = "environment_key_provider_v1"
    description: str = (
        "Provides API keys by reading them from environment variables. "
        "Phase 6C.7: supports per-scope keys via uppercase-prefixed env vars "
        "(e.g., scope={'tenant':'team_a'} → checks TEAM_A_<KEY_NAME> first, "
        "falls back to plain <KEY_NAME>). Phase 6C.8: refresh() is a no-op "
        "because os.environ is read every call — already 'hot'."
    )

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        logger.info(f"{self.plugin_id}: Initialized. Will read keys from environment variables.")

    async def get_key(self, key_name: str, scope: Optional[Dict[str, Any]] = None) -> Optional[str]:
        # Phase 6C.7: try scoped lookup first.
        # scope={"tenant": "team_a"} → TEAM_A_OPENAI_API_KEY
        # scope={"env": "prod", "tenant": "team_a"} → PROD_TEAM_A_OPENAI_API_KEY
        # The convention: each scope value is upper-snake-cased and prepended.
        if scope:
            # Order matters for determinism: tenant > team > env.
            for skey in ("tenant", "team", "env"):
                if scope.get(skey):
                    prefix = str(scope[skey]).upper().replace("-", "_").replace(":", "_")
                    scoped_name = f"{prefix}_{key_name}"
                    val = os.environ.get(scoped_name)
                    if val:
                        logger.debug(
                            f"{self.plugin_id}: Retrieved scoped key '{scoped_name}' for "
                            f"scope={scope!r}."
                        )
                        return val
        key_value = os.environ.get(key_name)
        if key_value:
            logger.debug(f"{self.plugin_id}: Retrieved key '{key_name}' from environment (exists).")
        else:
            logger.debug(f"{self.plugin_id}: Key '{key_name}' not found in environment variables.")
        return key_value

    async def refresh(self) -> None:
        # os.environ is re-read on every get_key call — nothing to invalidate.
        logger.debug(f"{self.plugin_id}: refresh() called (no-op; env vars read on every get_key).")

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Teardown complete.")
