"""Protocol for KeyProvider: Securely provides API keys to tools."""
import logging
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

@runtime_checkable # Allows isinstance checks if needed
class KeyProvider(Protocol):
    """
    Protocol for a component that securely provides API keys.
    This protocol **must be implemented by the consuming application**.
    The middleware itself does not store or manage API keys directly.
    All methods must be async.
    """
    async def get_key(self, key_name: str) -> Optional[str]:
        """
        Asynchronously retrieves the API key value for the given key name.

        Args:
            key_name: The logical name of the API key required by a tool
                      (e.g., "OPENWEATHERMAP_API_KEY", "OPENAI_API_KEY").

        Returns:
            The API key string if found and accessible, otherwise None.
            Implementations should fetch keys securely (e.g., from environment
            variables, a secrets manager/vault, or application configuration).
            It should log (at debug level) if a key is requested but not found,
            but avoid logging the key value itself.
        """
        # Example of what an implementation might log (this is a protocol, so no actual logic here)
        # logger.debug(f"KeyProvider: Request received for key '{key_name}'.")
        # value = self._internal_secure_key_storage.get(key_name)
        # if not value:
        #     logger.warning(f"KeyProvider: Key '{key_name}' not found in secure storage.")
        # return value
        ... # Indicates abstract method in Protocol
