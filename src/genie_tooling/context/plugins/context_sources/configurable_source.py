import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from genie_tooling.context.protocols import ContextSourcePlugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


class ConfigurableContextSourcePlugin(ContextSourcePlugin):
    """
    A context source that retrieves user profiles from a dictionary defined
    in the application's configuration. This allows for easy definition of
    multiple static profiles for testing or for applications with a fixed
    set of user personas.
    """

    plugin_id: str = "configurable_context_source_v1"
    description: str = "Provides user profiles from a predefined configuration dictionary."

    _profiles_by_id: Dict[str, Dict[str, Any]]
    _default_profile: Dict[str, Any]

    async def setup(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self._profiles_by_id = cfg.get("profiles", {})
        self._default_profile = cfg.get("default_profile", {"name": "Default User", "expertise": "layperson"})

        logger.info(
            f"[{self.plugin_id}] Initialized with {len(self._profiles_by_id)} configured profiles. "
            f"Default profile: {self._default_profile}"
        )

    async def get_profile(self, session_id: str, genie: "Genie") -> Dict[str, Any]:
        if not session_id:
            logger.debug(f"[{self.plugin_id}] No session_id provided, returning default profile.")
            return self._default_profile

        profile = self._profiles_by_id.get(session_id, self._default_profile)
        logger.debug(f"[{self.plugin_id}] Providing profile for session '{session_id}': {profile}")
        return profile
