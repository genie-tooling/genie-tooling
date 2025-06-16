# src/genie_tooling/conversation/impl/redis_state_provider.py
import json
import logging
from typing import Any, Dict, Optional

from ..types import ConversationState

# Corrected relative imports
from .abc import ConversationStateProviderPlugin

logger = logging.getLogger(__name__)

try:
    from redis import asyncio as aioredis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None
    RedisConnectionError = Exception
    RedisError = Exception
    REDIS_AVAILABLE = False

class RedisStateProviderPlugin(ConversationStateProviderPlugin):
    plugin_id: str = "redis_conversation_state_v1"
    description: str = "Stores conversation state in a Redis instance."

    _redis_client: Optional[Any] = None
    _redis_url: Optional[str] = None
    _key_prefix: str = "genie_cs:"
    _default_ttl_seconds: Optional[int] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not REDIS_AVAILABLE:
            logger.error(f"{self.plugin_id}: 'redis' library (>=4.2) not installed. This plugin will not function.")
            return
        cfg = config or {}
        self._redis_url = cfg.get("redis_url")
        if not self._redis_url:
            logger.info(f"{self.plugin_id}: 'redis_url' not configured. Plugin will be disabled.")
            self._redis_client = None
            return
        self._key_prefix = cfg.get("key_prefix", self._key_prefix)
        self._default_ttl_seconds = cfg.get("default_ttl_seconds")
        if self._default_ttl_seconds is not None and self._default_ttl_seconds <= 0:
            self._default_ttl_seconds = None
        try:
            self._redis_client = aioredis.from_url(self._redis_url) # type: ignore
            await self._redis_client.ping() # type: ignore
            logger.info(f"{self.plugin_id}: Connected to Redis at {self._redis_url}. Key prefix: '{self._key_prefix}'.")
        except (RedisConnectionError, RedisError) as e: # type: ignore
            self._redis_client = None
            logger.error(f"{self.plugin_id}: Failed to connect to Redis: {e}", exc_info=True)
        except Exception as e_other:
            self._redis_client = None
            logger.error(f"{self.plugin_id}: Unexpected error connecting to Redis: {e_other}", exc_info=True)

    def _get_redis_key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    async def load_state(self, session_id: str) -> Optional[ConversationState]:
        if not self._redis_client:
            logger.error(f"{self.plugin_id}: Redis client not available.")
            return None
        redis_key = self._get_redis_key(session_id)
        try:
            json_data_bytes = await self._redis_client.get(redis_key) # type: ignore
            if json_data_bytes:
                json_data = json_data_bytes.decode("utf-8") if isinstance(json_data_bytes, bytes) else json_data_bytes
                state_dict = json.loads(json_data)
                if isinstance(state_dict, dict) and "session_id" in state_dict and "history" in state_dict:
                    # Ensure all required fields of TypedDict are present before casting
                    # This is a simplified check. Pydantic would be more robust for validation.
                    return ConversationState(session_id=state_dict["session_id"], history=state_dict["history"], metadata=state_dict.get("metadata"))
                else:
                    logger.error(f"{self.plugin_id}: Invalid state structure in Redis for key '{redis_key}'. Data: {state_dict}")
                    return None
            return None
        except RedisError as e: # type: ignore
            logger.error(f"{self.plugin_id}: Redis GET error for key '{redis_key}': {e}", exc_info=True)
            return None
        except json.JSONDecodeError as e_json:
            logger.error(f"{self.plugin_id}: Failed to JSON decode state from Redis for key '{redis_key}': {e_json}", exc_info=True)
            return None

    async def save_state(self, state: ConversationState) -> None:
        if not self._redis_client:
            logger.error(f"{self.plugin_id}: Redis client not available. Cannot save state.")
            return
        if not state or "session_id" not in state:
            logger.error(f"{self.plugin_id}: Attempted to save invalid state (missing session_id). State: {state}")
            return
        redis_key = self._get_redis_key(state["session_id"])
        try:
            state_dict_to_save = dict(state)
            json_data = json.dumps(state_dict_to_save)
            await self._redis_client.set(redis_key, json_data.encode("utf-8"), ex=self._default_ttl_seconds) # type: ignore
            logger.debug(f"{self.plugin_id}: Saved state for session_id '{state['session_id']}' to Redis key '{redis_key}'.")
        except (TypeError, RedisError) as e: # type: ignore
            logger.error(f"{self.plugin_id}: Redis SET error for key '{redis_key}': {e}", exc_info=True)

    async def delete_state(self, session_id: str) -> bool:
        if not self._redis_client:
            logger.error(f"{self.plugin_id}: Redis client not available. Cannot delete state.")
            return False
        redis_key = self._get_redis_key(session_id)
        try:
            deleted_count = await self._redis_client.delete(redis_key) # type: ignore
            if deleted_count > 0:
                logger.debug(f"{self.plugin_id}: Deleted state for session_id '{session_id}' from Redis key '{redis_key}'.")
                return True
            return False
        except RedisError as e: # type: ignore
            logger.error(f"{self.plugin_id}: Redis DELETE error for key '{redis_key}': {e}", exc_info=True)
            return False

    async def teardown(self) -> None:
        if self._redis_client:
            try:
                await self._redis_client.close() # type: ignore
            except RedisError as e: # type: ignore
                logger.error(f"{self.plugin_id}: Error closing Redis client: {e}", exc_info=True)
            finally:
                self._redis_client = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
