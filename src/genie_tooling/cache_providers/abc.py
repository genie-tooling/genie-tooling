"""Abstract Base Class/Protocol for CacheProvider Plugins."""
import logging
from typing import Any, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

logger = logging.getLogger(__name__)

@runtime_checkable
class CacheProvider(Plugin, Protocol):
    """Protocol for a cache provider, designed for async operations."""
    # plugin_id: str (from Plugin protocol)
    description: str # Human-readable description of this cache provider

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache.
        Args:
            key: The key of the item to retrieve.
        Returns:
            The cached item, or None if the key is not found or item is expired.
        """
        logger.warning(f"CacheProvider '{self.plugin_id}' get method not fully implemented.")
        return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Stores an item in the cache.
        Args:
            key: The key under which to store the item.
            value: The item to store. Should be serializable if cache is external.
            ttl_seconds: Optional time-to-live in seconds. If None, item may persist indefinitely
                         or use a default TTL defined by the cache provider.
        """
        logger.warning(f"CacheProvider '{self.plugin_id}' set method not fully implemented.")
        pass

    async def delete(self, key: str) -> bool:
        """
        Deletes an item from the cache.
        Args:
            key: The key of the item to delete.
        Returns:
            True if the key existed and was deleted, False otherwise.
        """
        logger.warning(f"CacheProvider '{self.plugin_id}' delete method not fully implemented.")
        return False

    async def exists(self, key: str) -> bool:
        """
        Checks if a key exists in the cache (and is not expired).
        Args:
            key: The key to check.
        Returns:
            True if the key exists and is valid, False otherwise.
        """
        # Default implementation using get()
        logger.debug(f"CacheProvider '{self.plugin_id}' exists method using default get() check.")
        return await self.get(key) is not None

    async def clear_all(self) -> bool:
        """
        Clears all items from the cache managed by this provider.
        Use with caution, especially for shared caches.
        Returns:
            True if the clear operation was successful or attempted, False on failure.
        """
        logger.warning(f"CacheProvider '{self.plugin_id}' clear_all method not fully implemented.")
        return False
