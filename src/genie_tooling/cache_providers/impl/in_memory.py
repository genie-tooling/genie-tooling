import asyncio  # Standard import
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from genie_tooling.cache_providers.abc import CacheProvider

logger = logging.getLogger(__name__)

class _CacheEntry:
    def __init__(self, value: Any, expiry_time: Optional[float]):
        self.value = value
        self.expiry_time = expiry_time

class InMemoryCacheProvider(CacheProvider):
    plugin_id: str = "in_memory_cache_provider_v1"
    description: str = "A simple, non-persistent, in-process in-memory cache with TTL and basic LRU eviction."

    _cache: OrderedDict[str, _CacheEntry]
    _lock: asyncio.Lock
    _max_size: Optional[int] = None
    _cleanup_task: Optional[asyncio.Task] = None
    _default_ttl_seconds: Optional[int] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._cache = OrderedDict()
        self._lock = asyncio.Lock()
        cfg = config or {}
        self._max_size = cfg.get("max_size")
        if self._max_size is not None and self._max_size <= 0:
            self._max_size = None

        self._default_ttl_seconds = cfg.get("default_ttl_seconds")
        if self._default_ttl_seconds is not None and self._default_ttl_seconds <= 0:
            self._default_ttl_seconds = None

        cleanup_interval = cfg.get("cleanup_interval_seconds")

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try: await self._cleanup_task
            except asyncio.CancelledError: logger.debug(f"{self.plugin_id}: Previous cleanup task successfully cancelled during setup.")
            except Exception as e_await: logger.warning(f"{self.plugin_id}: Error awaiting previous cleanup task cancellation: {e_await}")
            self._cleanup_task = None

        if cleanup_interval is not None and cleanup_interval > 0:
            try:
                loop = asyncio.get_running_loop()
                logger.debug(f"{self.plugin_id}: About to call asyncio.create_task for periodic cleanup.")
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup(cleanup_interval)) # Using asyncio.create_task directly
                logger.info(f"{self.plugin_id}: Initialized. Max size: {self._max_size or 'unlimited'}. Default TTL: {self._default_ttl_seconds or 'none'}. Cleanup interval: {cleanup_interval}s. Task created: {self._cleanup_task is not None}.")
            except RuntimeError as e_loop:
                 logger.error(f"{self.plugin_id}: Could not get running loop to create cleanup task: {e_loop}. Periodic cleanup disabled.")
                 self._cleanup_task = None
        else:
            logger.info(f"{self.plugin_id}: Initialized. Max size: {self._max_size or 'unlimited'}. Default TTL: {self._default_ttl_seconds or 'none'}. No periodic cleanup.")

    async def _periodic_cleanup(self, interval_seconds: int) -> None:
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                async with self._lock:
                    current_time = time.monotonic()
                    keys_to_delete = [k for k, entry in list(self._cache.items()) if entry.expiry_time and entry.expiry_time <= current_time]
                    for key in keys_to_delete:
                        if key in self._cache: del self._cache[key]
                    if keys_to_delete: logger.debug(f"{self.plugin_id}: Cleaned {len(keys_to_delete)} expired items.")
        except asyncio.CancelledError: logger.debug(f"{self.plugin_id}: Periodic cleanup task was cancelled.")
        except Exception as e: logger.error(f"{self.plugin_id}: Periodic cleanup task encountered an error: {e}", exc_info=True)

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            entry = self._cache.get(key)
            if entry:
                if entry.expiry_time is None or entry.expiry_time > time.monotonic():
                    if key in self._cache: self._cache.move_to_end(key)
                    return entry.value
                else:
                    if key in self._cache: del self._cache[key]
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        final_ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        async with self._lock:
            if self._max_size is not None and len(self._cache) >= self._max_size and key not in self._cache:
                try:
                    self._cache.popitem(last=False)
                except KeyError: pass

            expiry_abs_time = (time.monotonic() + final_ttl) if final_ttl and final_ttl > 0 else None
            self._cache[key] = _CacheEntry(value, expiry_abs_time)
            if key in self._cache: self._cache.move_to_end(key)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                return True
            return False

    async def exists(self, key: str) -> bool:
        async with self._lock:
            entry = self._cache.get(key)
            if entry and (entry.expiry_time is None or entry.expiry_time > time.monotonic()):
                return True
            return False

    async def clear_all(self) -> bool:
        async with self._lock: self._cache.clear()
        return True

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Starting teardown...")
        if self._cleanup_task and not self._cleanup_task.done():
            logger.debug(f"{self.plugin_id}: Cancelling cleanup task.")
            self._cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=1.0)
            except asyncio.CancelledError:
                logger.debug(f"{self.plugin_id}: Cleanup task successfully cancelled.")
            except asyncio.TimeoutError:
                logger.warning(f"{self.plugin_id}: Timeout waiting for cleanup task to finish after cancellation.")
            except ValueError as ve:
                logger.warning(f"{self.plugin_id}: Error awaiting cleanup task (likely loop mismatch): {ve}. Task was cancelled.")
            except Exception as e:
                logger.error(f"{self.plugin_id}: Error awaiting cleanup task during teardown: {e}", exc_info=True)
        self._cleanup_task = None
        await self.clear_all()
        logger.info(f"{self.plugin_id}: Torn down.")
