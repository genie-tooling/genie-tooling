"""NoOpRedactorPlugin: A redactor plugin that performs no redaction."""
import logging
from typing import Any, Dict, Optional

from genie_tooling.redactors.abc import Redactor

logger = logging.getLogger(__name__)

class NoOpRedactorPlugin(Redactor):
    plugin_id: str = "noop_redactor_v1"
    description: str = "A pass-through redactor plugin that performs no actual redaction. Useful for disabling custom redaction or as a default."

    def sanitize(self, data: Any, schema_hints: Optional[Dict[str, Any]] = None) -> Any:
        """Returns the data as-is, without any modifications."""
        logger.debug(f"{self.plugin_id}: sanitize called, returning data untouched.")
        return data

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        logger.debug(f"{self.plugin_id}: Setup complete (no-op).")

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Teardown complete (no-op).")
