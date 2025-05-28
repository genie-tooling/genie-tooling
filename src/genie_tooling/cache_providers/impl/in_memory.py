### src/genie_tooling/caching/impl/in_memory.py
"""InMemoryCacheProvider: A simple, non-persistent in-memory cache with TTL support."""
import asyncio  # For lock and potential background cleanup task
import logging
import time
from typing import Any, Dict, Optional

# Updated import path for CacheProvider
from genie_tooling.cache_providers.abc import CacheProvider

logger = logging.getLogger(__name__)

class _CacheEntry:
    """Helper class to store cache value and its expiry time."""
    def __init__(self, value: Any, expiry_time: Optional[float]):
        self.value = value
        self.expiry_time = expiry_time

class InMemoryCacheProvider(CacheProvider):
    plugin_id: str = "in_memory_cache_provider_v1"
    description: str = "A simple, non-persistent, in-process in-memory cache with TTL support."

    _cache: Dict[str, _CacheEntry]
    _lock: asyncio.Lock
    _max_size: Optional[int] = None
    _cleanup_task: Optional[asyncio.Task] = None
    _default_ttl_seconds: Optional[int] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._cache = {}
        self._lock = asyncio.Lock()

        cfg = config or {}
        self._max_size = cfg.get("max_size")
        if self._max_size is not None and self._max_size <= 0:
            logger.warning(f"{self.plugin_id}: Invalid max_size ({self._max_size}), disabling size limit.")
            self._max_size = None

        self._default_ttl_seconds = cfg.get("default_ttl_seconds")
        if self._default_ttl_seconds is not None and self._default_ttl_seconds <= 0:
            logger.warning(f"{self.plugin_id}: Invalid default_ttl_seconds ({self._default_ttl_seconds}), disabling default TTL.")
            self._default_ttl_seconds = None

        cleanup_interval = cfg.get("cleanup_interval_seconds")
        if cleanup_interval is not None and cleanup_interval > 0:
            # Ensure that an existing task is cancelled before creating a new one if setup is called multiple times
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    logger.debug(f"{self.plugin_id}: Previous cleanup task cancelled during setup.")
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup(cleanup_interval))
            logger.info(f"{self.plugin_id}: Initialized with max_size={self._max_size}, default_ttl={self._default_ttl_seconds}s. Periodic cleanup every {cleanup_interval}s.")
        else:
            logger.info(f"{self.plugin_id}: Initialized with max_size={self._max_size}, default_ttl={self._default_ttl_seconds}s. No periodic cleanup.")


    async def _periodic_cleanup(self, interval_seconds: int) -> None:
        """Periodically removes expired items from the cache."""
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                async with self._lock:
                    current_time = time.monotonic()
                    keys_to_delete = [
                        key for key, entry in self._cache.items()
                        if entry.expiry_time is not None and entry.expiry_time <= current_time
                    ]
                    for key in keys_to_delete:
                        # Ensure key still exists before deleting, in case of concurrent modification
                        if key in self._cache:
                            del self._cache[key]
                    if keys_to_delete:
                        logger.debug(f"{self.plugin_id}: Cleaned up {len(keys_to_delete)} expired items.")
        except asyncio.CancelledError:
            logger.debug(f"{self.plugin_id}: Periodic cleanup task was cancelled.")
            # Optionally re-raise if the cancellation needs to propagate further,
            # but for a background task, usually just exiting is fine.
            # raise # Re-raise if needed
        except Exception as e:
            logger.error(f"{self.plugin_id}: Periodic cleanup task encountered an error: {e}", exc_info=True)
            # Decide if the task should terminate or try to continue. For now, it terminates.


    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._cache.get(key)
            if entry:
                if entry.expiry_time is None or entry.expiry_time > time.monotonic():
                    logger.debug(f"{self.plugin_id}: Cache hit for key '{key}'.")
                    return entry.value
                else:
                    logger.debug(f"{self.plugin_id}: Cache miss (expired) for key '{key}'. Removing.")
                    del self._cache[key]
            else:
                logger.debug(f"{self.plugin_id}: Cache miss for key '{key}'.")
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        final_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds

        async with self._lock:
            if self._max_size is not None and len(self._cache) >= self._max_size and key not in self._cache:
                try:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    logger.debug(f"{self.plugin_id}: Cache full (size {self._max_size}). Evicted '{oldest_key}' (basic LRU).")
                except StopIteration:
                    pass

            expiry_abs_time = (time.monotonic() + final_ttl) if final_ttl is not None and final_ttl > 0 else None
            self._cache[key] = _CacheEntry(value, expiry_abs_time)
            logger.debug(f"{self.plugin_id}: Item set for key '{key}'. TTL: {final_ttl}s. Expires at: {expiry_abs_time if expiry_abs_time else 'Never'}.")

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                logger.debug(f"{self.plugin_id}: Item deleted for key '{key}'.")
                return entry.expiry_time is None or entry.expiry_time > time.monotonic()
            logger.debug(f"{self.plugin_id}: Item to delete not found for key '{key}'.")
            return False

    async def exists(self, key: str) -> bool:
        async with self._lock:
            entry = self._cache.get(key)
            if entry:
                if entry.expiry_time is None or entry.expiry_time > time.monotonic():
                    return True
            return False

    async def clear_all(self) -> bool:
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
        logger.info(f"{self.plugin_id}: Cache cleared. Removed {count} items.")
        return True

    async def teardown(self) -> None:
        """Stops the periodic cleanup task and clears the cache."""
        logger.debug(f"{self.plugin_id}: Starting teardown...")
        if self._cleanup_task and not self._cleanup_task.done():
            logger.debug(f"{self.plugin_id}: Cancelling cleanup task.")
            self._cleanup_task.cancel()
            try:
                # Wait for the task to actually finish after cancellation
                # Add a timeout to prevent hanging indefinitely if the task doesn't handle cancellation well
                await asyncio.wait_for(self._cleanup_task, timeout=1.0)
            except asyncio.CancelledError:
                logger.debug(f"{self.plugin_id}: Cleanup task successfully cancelled and awaited.")
            except asyncio.TimeoutError:
                logger.warning(f"{self.plugin_id}: Timeout waiting for cleanup task to finish after cancellation.")
            except Exception as e: # Catch any other unexpected errors during task await
                logger.error(f"{self.plugin_id}: Error awaiting cleanup task during teardown: {e}", exc_info=True)

        self._cleanup_task = None # Clear the task reference
        await self.clear_all() # Clear cache data
        logger.info(f"{self.plugin_id}: Torn down.")
###<END-OF-FILE>###
