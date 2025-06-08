"""PromptManager: Orchestrates PromptRegistryPlugin and PromptTemplatePlugin."""
import logging
from typing import Any, Dict, List, Optional

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.observability.manager import InteractionTracingManager

from .abc import PromptRegistryPlugin, PromptTemplatePlugin
from .types import FormattedPrompt, PromptData, PromptIdentifier

logger = logging.getLogger(__name__)

class PromptManager:
    def __init__(
        self,
        plugin_manager: PluginManager,
        default_registry_id: Optional[str] = None,
        default_template_engine_id: Optional[str] = None,
        registry_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
        template_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
        tracing_manager: Optional[InteractionTracingManager] = None,
    ):
        self._plugin_manager = plugin_manager
        self._default_registry_id = default_registry_id
        self._default_template_engine_id = default_template_engine_id
        self._registry_configurations = registry_configurations or {}
        self._template_configurations = template_configurations or {}
        self._tracing_manager = tracing_manager
        logger.info("PromptManager initialized.")

    async def _trace(self, event_name: str, data: Dict, level: str = "info"):
        if self._tracing_manager:
            await self._tracing_manager.trace_event(f"log.{level}", {"message": event_name, **data}, "PromptManager")

    async def get_raw_template(
        self, name: str, version: Optional[str] = None, registry_id: Optional[str] = None
    ) -> Optional[str]:
        reg_id_to_use = registry_id or self._default_registry_id
        if not reg_id_to_use:
            await self._trace("log.error", {"message": "No prompt registry ID specified and no default is set."})
            return None
        registry = await self._plugin_manager.get_plugin_instance(
            reg_id_to_use, config=self._registry_configurations.get(reg_id_to_use, {})
        )
        if not registry or not isinstance(registry, PromptRegistryPlugin):
            await self._trace("log.error", {"message": f"PromptRegistryPlugin '{reg_id_to_use}' not found or invalid."})
            return None
        return await registry.get_template_content(name, version)

    async def render_prompt(
        self, name: str, data: PromptData, version: Optional[str] = None,
        registry_id: Optional[str] = None, template_engine_id: Optional[str] = None
    ) -> Optional[FormattedPrompt]:
        template_content = await self.get_raw_template(name, version, registry_id)
        if template_content is None:
            return None

        engine_id_to_use = template_engine_id or self._default_template_engine_id
        if not engine_id_to_use:
            await self._trace("log.error", {"message": "No prompt template engine ID specified and no default is set."})
            return None

        engine = await self._plugin_manager.get_plugin_instance(
            engine_id_to_use, config=self._template_configurations.get(engine_id_to_use, {})
        )
        if not engine or not isinstance(engine, PromptTemplatePlugin):
            await self._trace("log.error", {"message": f"PromptTemplatePlugin '{engine_id_to_use}' not found or invalid."})
            return None
        return await engine.render(template_content, data)

    async def render_chat_prompt(
        self, name: str, data: PromptData, version: Optional[str] = None,
        registry_id: Optional[str] = None, template_engine_id: Optional[str] = None
    ) -> Optional[List[ChatMessage]]:
        template_content = await self.get_raw_template(name, version, registry_id)
        if template_content is None:
            return None
        engine_id_to_use = template_engine_id or self._default_template_engine_id
        if not engine_id_to_use:
            await self._trace("log.error", {"message": "No prompt template engine ID specified for chat and no default is set."})
            return None
        engine = await self._plugin_manager.get_plugin_instance(
            engine_id_to_use, config=self._template_configurations.get(engine_id_to_use, {})
        )
        if not engine or not isinstance(engine, PromptTemplatePlugin):
            await self._trace("log.error", {"message": f"PromptTemplatePlugin '{engine_id_to_use}' for chat not found or invalid."})
            return None
        return await engine.render_chat_messages(template_content, data)


    async def list_available_templates(self, registry_id: Optional[str] = None) -> List[PromptIdentifier]:
        reg_id_to_use = registry_id or self._default_registry_id
        if not reg_id_to_use:
            await self._trace("log.error", {"message": "No prompt registry ID specified for listing and no default is set."})
            return []
        registry = await self._plugin_manager.get_plugin_instance(
            reg_id_to_use, config=self._registry_configurations.get(reg_id_to_use, {})
        )
        if not registry or not isinstance(registry, PromptRegistryPlugin):
            await self._trace("log.error", {"message": f"PromptRegistryPlugin '{reg_id_to_use}' for listing not found or invalid."})
            return []
        return await registry.list_available_templates()

    async def teardown(self) -> None:
        await self._trace("log.info", {"message": "Tearing down..."})
        pass
