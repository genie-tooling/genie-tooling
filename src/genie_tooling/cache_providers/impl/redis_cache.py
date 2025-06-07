import json
import logging
from typing import Any, Dict, Optional

from genie_tooling.cache_providers.abc import CacheProvider

try:
    from redis import asyncio as aioredis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import RedisError
except ImportError:
    aioredis = None # type: ignore
    RedisError = Exception # type: ignore
    RedisConnectionError = Exception # type: ignore


logger = logging.getLogger(__name__)

class RedisCacheProvider(CacheProvider):
    plugin_id: str = "redis_cache_provider_v1"
    description: str = "Uses official Redis client (redis.asyncio) as a distributed cache backend."

    _redis_client: Optional["aioredis.Redis"] = None
    _redis_url: Optional[str] = None
    _default_ttl_seconds: Optional[int] = None
    _json_serialization: bool = True

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not aioredis:
            logger.error(f"{self.plugin_id} Error: 'redis' library (>=4.2) not installed.")
            return

        cfg = config or {}
        self._redis_url = cfg.get("redis_url")
        if not self._redis_url:
            logger.info(f"{self.plugin_id}: 'redis_url' not configured. Plugin will be disabled and will not attempt to connect.")
            self._redis_client = None
            return

        self._default_ttl_seconds = cfg.get("default_ttl_seconds")
        if self._default_ttl_seconds is not None and self._default_ttl_seconds <= 0:
            self._default_ttl_seconds = None
        self._json_serialization = bool(cfg.get("json_serialization", True))

        try:
            self._redis_client = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._redis_client.ping()
            logger.info(f"{self.plugin_id}: Connected to Redis at {self._redis_url}.")
        except (RedisConnectionError, RedisError) as e:
            self._redis_client = None
            logger.error(f"{self.plugin_id}: Failed to connect to Redis: {e}", exc_info=True)
        except Exception as e_other:
            self._redis_client = None
            logger.error(f"{self.plugin_id}: Unexpected error connecting to Redis: {e_other}", exc_info=True)

    async def get(self, key: str) -> Optional[Any]:
        if not self._redis_client:
            return None
        try:
            value_str = await self._redis_client.get(key)
            if value_str is None:
                return None
            if self._json_serialization:
                try:
                    return json.loads(value_str)
                except json.JSONDecodeError:
                    logger.error(f"{self.plugin_id}: Failed to JSON decode for key '{key}'. Returning raw string.", exc_info=True)
                    return value_str
            return value_str
        except RedisError as e:
            logger.error(f"{self.plugin_id}: Redis GET error for '{key}': {e}", exc_info=True)
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        if not self._redis_client:
            return
        final_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        value_to_store: str
        try:
            if self._json_serialization:
                if isinstance(value, (str, bytes)):
                    value_to_store = str(value)
                else:
                    value_to_store = json.dumps(value)
            else:
                if not isinstance(value, (str, bytes, int, float)):
                    logger.warning(f"{self.plugin_id}: Value for '{key}' is complex type ({type(value)}) "
                                   "but JSON serialization is off. Storing as string representation.")
                value_to_store = str(value)

            await self._redis_client.set(key, value_to_store, ex=final_ttl if final_ttl and final_ttl > 0 else None)
        except (TypeError, RedisError) as e:
            logger.error(f"{self.plugin_id}: Redis SET error for '{key}': {e}", exc_info=True)

    async def delete(self, key: str) -> bool:
        if not self._redis_client:
            return False
        try:
            deleted_count = await self._redis_client.delete(key)
            return deleted_count > 0
        except RedisError as e:
            logger.error(f"{self.plugin_id}: Redis DELETE error for '{key}': {e}", exc_info=True)
            return False

    async def exists(self, key: str) -> bool:
        if not self._redis_client:
            return False
        try:
            count = await self._redis_client.exists(key)
            return count > 0
        except RedisError as e:
            logger.error(f"{self.plugin_id}: Redis EXISTS error for '{key}': {e}", exc_info=True)
            return False

    async def clear_all(self) -> bool:
        if not self._redis_client:
            return False
        try:
            await self._redis_client.flushdb()
            return True
        except RedisError as e:
            logger.error(f"{self.plugin_id}: Redis FLUSHDB error: {e}", exc_info=True)
            return False

    async def teardown(self) -> None:
        if self._redis_client:
            try:
                await self._redis_client.close()
            except RedisError as e:
                logger.error(f"{self.plugin_id}: Error closing Redis client: {e}", exc_info=True)
            finally:
                self._redis_client = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
