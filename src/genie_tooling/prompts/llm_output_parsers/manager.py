# src/genie_tooling/llm_output_parsers/manager.py
"""LLMOutputParserManager: Orchestrates LLMOutputParserPlugin."""
import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager

from .abc import LLMOutputParserPlugin
from .types import ParsedOutput

logger = logging.getLogger(__name__)

class LLMOutputParserManager:
    def __init__(
        self,
        plugin_manager: PluginManager,
        default_parser_id: Optional[str] = None,
        parser_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._plugin_manager = plugin_manager
        self._default_parser_id = default_parser_id
        self._parser_configurations = parser_configurations or {}
        logger.info("LLMOutputParserManager initialized.")

    async def parse(
        self, text_output: str, parser_id: Optional[str] = None, schema: Optional[Any] = None
    ) -> ParsedOutput:
        parser_id_to_use = parser_id or self._default_parser_id
        if not parser_id_to_use:
            logger.error("No LLM output parser ID specified and no default is set.")
            # Fallback to returning raw text if no parser can be identified
            return text_output

        parser_plugin = await self._plugin_manager.get_plugin_instance(
            parser_id_to_use, config=self._parser_configurations.get(parser_id_to_use, {})
        )
        if not parser_plugin or not isinstance(parser_plugin, LLMOutputParserPlugin):
            logger.error(f"LLMOutputParserPlugin '{parser_id_to_use}' not found or invalid.")
            return text_output # Fallback

        try:
            return parser_plugin.parse(text_output, schema)
        except Exception as e:
            logger.error(f"Error parsing output with parser '{parser_id_to_use}': {e}", exc_info=True)
            raise # Re-raise the parsing exception

    async def teardown(self) -> None:
        logger.info("LLMOutputParserManager tearing down (no specific resources to release here).")
        pass