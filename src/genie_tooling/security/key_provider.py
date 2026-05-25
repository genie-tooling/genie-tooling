"""Protocol for KeyProvider: Securely provides API keys to tools.

Phase 6C.7 added optional scope-based lookup; Phase 6C.8 added an optional
``refresh()`` hook for hot-reloading rotated credentials without a process
restart. Both extensions are backwards-compatible — existing implementations
keep working without changes.
"""
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class KeyProvider(Protocol):
    """
    Protocol for a component that securely provides API keys.
    This protocol **must be implemented by the consuming application**.
    The middleware itself does not store or manage API keys directly.
    All methods must be async.
    """
    async def get_key(self, key_name: str, scope: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Asynchronously retrieves the API key value for the given key name.

        Args:
            key_name: The logical name of the API key required by a tool
                      (e.g., "OPENWEATHERMAP_API_KEY", "OPENAI_API_KEY").
            scope: Optional scoping context (Phase 6C.7). Common keys:

                   * ``tenant``: tenant / team identifier
                   * ``env``: ``"prod"`` / ``"staging"`` / ``"dev"``
                   * ``team``: team identifier

                   Implementations may use these to resolve the *right* key for
                   the current request — e.g., separate Slack tokens per team,
                   separate OpenAI keys per environment. Implementations that
                   don't need scoping should simply ignore the argument.

        Returns:
            The API key string if found and accessible, otherwise None.
            Implementations should fetch keys securely (e.g., from environment
            variables, a secrets manager/vault, or application configuration).
            It should log (at debug level) if a key is requested but not found,
            but avoid logging the key value itself.
        """
        ...

    async def refresh(self) -> None:
        """Phase 6C.8 — invalidate any cached keys and re-read from source.

        Useful when credentials have been rotated and the process is
        long-lived. Implementations that re-read every call may treat this
        as a no-op. The default ``EnvironmentKeyProvider`` re-reads
        ``os.environ`` on the next ``get_key`` call.
        """
        # Default implementations may safely ignore this method.
        return None
