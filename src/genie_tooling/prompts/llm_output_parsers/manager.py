"""LLMOutputParserManager: Orchestrates LLMOutputParserPlugin."""
import logging
from typing import Any, Dict, Optional

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.observability.manager import InteractionTracingManager

from .abc import LLMOutputParserPlugin
from .types import ParsedOutput

logger = logging.getLogger(__name__)

class LLMOutputParserManager:
    def __init__(
        self,
        plugin_manager: PluginManager,
        default_parser_id: Optional[str] = None,
        parser_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
        tracing_manager: Optional[InteractionTracingManager] = None,
    ):
        self._plugin_manager = plugin_manager
        self._default_parser_id = default_parser_id
        self._parser_configurations = parser_configurations or {}
        self._tracing_manager = tracing_manager
        logger.info("LLMOutputParserManager initialized.")

    async def _trace(self, event_name: str, data: Dict, level: str = "info"):
        if self._tracing_manager:
            await self._tracing_manager.trace_event(f"log.{level}", {"message": event_name, **data}, "LLMOutputParserManager")

    async def parse(
        self, text_output: str, parser_id: Optional[str] = None, schema: Optional[Any] = None
    ) -> ParsedOutput:
        parser_id_to_use = parser_id or self._default_parser_id
        if not parser_id_to_use:
            await self._trace("log.error", {"message": "No LLM output parser ID specified and no default is set."})
            return text_output

        parser_plugin = await self._plugin_manager.get_plugin_instance(
            parser_id_to_use, config=self._parser_configurations.get(parser_id_to_use, {})
        )
        if not parser_plugin or not isinstance(parser_plugin, LLMOutputParserPlugin):
            await self._trace("log.error", {"message": f"LLMOutputParserPlugin '{parser_id_to_use}' not found or invalid."})
            return text_output

        try:
            return parser_plugin.parse(text_output, schema)
        except Exception as e:
            await self._trace("log.error", {"message": f"Error parsing output with parser '{parser_id_to_use}': {e}", "exc_info": True})
            raise

    async def teardown(self) -> None:
        await self._trace("log.info", {"message": "Tearing down..."})
        pass
