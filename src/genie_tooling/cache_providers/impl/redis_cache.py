"""RedisCacheProvider: Uses Redis as a cache backend via official redis[hiredis] asyncio support."""
import json  # For serializing/deserializing complex Python objects
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Import from redis.asyncio
try:
    from redis import (
        asyncio as aioredis,  # Use redis.asyncio and alias it to aioredis for minimal code change
    )
    from redis.exceptions import ConnectionError as RedisConnectionError
    from redis.exceptions import RedisError  # Specific exceptions
except ImportError:
    aioredis = None # type: ignore
    RedisError = Exception # type: ignore # Fallback base exception
    RedisConnectionError = Exception # type: ignore

# Updated import path for CacheProvider
from genie_tooling.cache_providers.abc import CacheProvider


class RedisCacheProvider(CacheProvider):
    plugin_id: str = "redis_cache_provider_v1"
    description: str = "Uses official Redis client (redis.asyncio) as a distributed cache backend."

    _redis_client: Optional[aioredis.Redis] = None
    _redis_url: str
    _default_ttl_seconds: Optional[int] = None
    _json_serialization: bool = True

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not aioredis:
            logger.error(f"{self.plugin_id} Error: 'redis' library (version >=4.2 with asyncio) not installed correctly. "
                         "Please install it with: poetry add redis or poetry install --extras full")
            return

        cfg = config or {}
        self._redis_url = cfg.get("redis_url", "redis://localhost:6379/0")
        if not self._redis_url:
            logger.error(f"{self.plugin_id} Error: 'redis_url' not provided in config.")
            return

        self._default_ttl_seconds = cfg.get("default_ttl_seconds")
        if self._default_ttl_seconds is not None and self._default_ttl_seconds <= 0:
            logger.warning(f"{self.plugin_id}: Invalid default_ttl_seconds ({self._default_ttl_seconds}), disabling default TTL.")
            self._default_ttl_seconds = None

        self._json_serialization = bool(cfg.get("json_serialization", True))

        try:
            logger.info(f"{self.plugin_id}: Connecting to Redis at {self._redis_url} using redis.asyncio...")
            self._redis_client = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._redis_client.ping()
            logger.info(f"{self.plugin_id}: Successfully connected to Redis and pinged.")
        except RedisConnectionError as e_conn:
            self._redis_client = None
            logger.error(f"{self.plugin_id}: Failed to connect to Redis at '{self._redis_url}': {e_conn}", exc_info=True)
        except RedisError as e_redis_other:
            self._redis_client = None
            logger.error(f"{self.plugin_id}: Redis setup error for '{self._redis_url}': {e_redis_other}", exc_info=True)
        except Exception as e_other:
            self._redis_client = None
            logger.error(f"{self.plugin_id}: Unexpected error connecting to Redis: {e_other}", exc_info=True)


    async def get(self, key: str) -> Optional[Any]:
        if not self._redis_client:
            logger.warning(f"{self.plugin_id}: Redis client not available. Cannot GET '{key}'.")
            return None
        try:
            value_str = await self._redis_client.get(key)
            if value_str is None:
                logger.debug(f"{self.plugin_id}: Cache miss for key '{key}'.")
                return None

            logger.debug(f"{self.plugin_id}: Cache hit for key '{key}'.")
            if self._json_serialization and isinstance(value_str, str):
                try:
                    return json.loads(value_str)
                except json.JSONDecodeError as e_json:
                    logger.error(f"{self.plugin_id}: Failed to JSON decode cached value for key '{key}': {e_json}. Returning raw string.", exc_info=True)
                    return value_str
            else:
                return value_str
        except RedisError as e_redis_get:
            logger.error(f"{self.plugin_id}: Redis error during GET for key '{key}': {e_redis_get}", exc_info=True)
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        if not self._redis_client:
            logger.warning(f"{self.plugin_id}: Redis client not available. Cannot SET '{key}'.")
            return

        final_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds

        value_to_store: Any
        try:
            if self._json_serialization:
                if not isinstance(value, (str, bytes, int, float)):
                    value_to_store = json.dumps(value)
                else:
                    value_to_store = value
            else:
                if not isinstance(value, (str, bytes, int, float)):
                    logger.warning(f"{self.plugin_id}: Value for key '{key}' is complex type ({type(value)}) "
                                   "but JSON serialization is off. Storing as string representation.")
                    value_to_store = str(value)
                else:
                    value_to_store = value

            if final_ttl is not None and final_ttl > 0:
                await self._redis_client.set(key, value_to_store, ex=final_ttl)
                logger.debug(f"{self.plugin_id}: Item SET for key '{key}'. TTL: {final_ttl}s.")
            else:
                await self._redis_client.set(key, value_to_store)
                logger.debug(f"{self.plugin_id}: Item SET for key '{key}' with no specific TTL.")
        except TypeError as e_type_json:
            logger.error(f"{self.plugin_id}: Failed to JSON serialize value for key '{key}': {e_type_json}", exc_info=True)
        except RedisError as e_redis_set:
            logger.error(f"{self.plugin_id}: Redis error during SET for key '{key}': {e_redis_set}", exc_info=True)

    async def delete(self, key: str) -> bool:
        if not self._redis_client:
            logger.warning(f"{self.plugin_id}: Redis client not available. Cannot DELETE '{key}'.")
            return False
        try:
            deleted_count = await self._redis_client.delete(key)
            was_deleted = deleted_count > 0
            logger.debug(f"{self.plugin_id}: Item delete for key '{key}'. Deleted: {was_deleted} (count: {deleted_count}).")
            return was_deleted
        except RedisError as e_redis_del:
            logger.error(f"{self.plugin_id}: Redis error during DELETE for key '{key}': {e_redis_del}", exc_info=True)
            return False

    async def exists(self, key: str) -> bool:
        if not self._redis_client:
            logger.warning(f"{self.plugin_id}: Redis client not available. Cannot check EXISTS for '{key}'.")
            return False
        try:
            key_exists_count = await self._redis_client.exists(key)
            key_exists = key_exists_count > 0
            logger.debug(f"{self.plugin_id}: Key '{key}' exists check result: {key_exists}.")
            return key_exists
        except RedisError as e_redis_exists:
            logger.error(f"{self.plugin_id}: Redis error during EXISTS for key '{key}': {e_redis_exists}", exc_info=True)
            return False

    async def clear_all(self) -> bool:
        if not self._redis_client:
            logger.warning(f"{self.plugin_id}: Redis client not available. Cannot CLEARALL.")
            return False
        try:
            await self._redis_client.flushdb() # Corrected call
            logger.info(f"{self.plugin_id}: Cache (current Redis DB) cleared via FLUSHDB.")
            return True
        except RedisError as e_redis_flush:
            logger.error(f"{self.plugin_id}: Redis error during FLUSHDB: {e_redis_flush}", exc_info=True)
            return False

    async def teardown(self) -> None:
        """Closes the Redis client connection pool."""
        if self._redis_client:
            try:
                await self._redis_client.close()
                logger.info(f"{self.plugin_id}: Redis client connection pool closed.")
            except RedisError as e_redis_close:
                logger.error(f"{self.plugin_id}: Error closing Redis client: {e_redis_close}", exc_info=True)
            except Exception as e_close_other:
                 logger.error(f"{self.plugin_id}: Unexpected error closing Redis client: {e_close_other}", exc_info=True)
            finally:
                self._redis_client = None
