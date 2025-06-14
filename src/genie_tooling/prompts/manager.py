### src/genie_tooling/prompts/manager.py
# src/genie_tooling/prompts/manager.py
"""PromptManager: Orchestrates PromptRegistryPlugin and PromptTemplatePlugin."""
import logging
from typing import Any, Dict, List, Optional, cast

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
        default_template_engine_id: Optional[str] = None,  # This should be the CANONICAL ID
        registry_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
        template_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
        tracing_manager: Optional[InteractionTracingManager] = None,
    ):
        self._plugin_manager = plugin_manager
        self._default_registry_id = default_registry_id
        self._default_template_engine_id = default_template_engine_id  # EXPECTS CANONICAL
        self._registry_configurations = registry_configurations or {}
        self._template_configurations = template_configurations or {}
        self._tracing_manager = tracing_manager
        logger.info(f"PromptManager initialized. Default template engine ID: {self._default_template_engine_id}")

    async def _trace(self, event_name: str, data: Dict, level: str = "info"):
        if self._tracing_manager:
            event_data_with_msg = data.copy()
            if "message" not in event_data_with_msg:
                if "error" in data:
                    event_data_with_msg["message"] = str(data["error"])
                else:
                    event_data_with_msg["message"] = event_name.split(".")[-1].replace("_", " ").capitalize()
            final_event_name = event_name
            if not event_name.startswith("log.") and level in ["debug", "info", "warning", "error", "critical"]:
                final_event_name = f"log.{level}"
            await self._tracing_manager.trace_event(final_event_name, event_data_with_msg, "PromptManager")

    async def get_raw_template(
        self, name: str, version: Optional[str] = None, registry_id: Optional[str] = None
    ) -> Optional[str]:
        reg_id_to_use = registry_id or self._default_registry_id  # Assumes this is canonical
        if not reg_id_to_use:
            await self._trace("log.error", {"message": "No prompt registry ID specified and no default is set."})
            return None
        registry_config = self._registry_configurations.get(reg_id_to_use, {})
        registry_any = await self._plugin_manager.get_plugin_instance(reg_id_to_use, config=registry_config)
        if not registry_any or not isinstance(registry_any, PromptRegistryPlugin):
            await self._trace("log.error", {"message": f"PromptRegistryPlugin '{reg_id_to_use}' not found or invalid."})
            return None
        registry = cast(PromptRegistryPlugin, registry_any)
        try:
            return await registry.get_template_content(name, version)
        except Exception as e:
            await self._trace("log.error", {"message": f"Error getting raw template '{name}' from registry '{reg_id_to_use}': {e}", "exc_info": True})
            return None

    async def render_prompt(
        self, name: Optional[str] = None, data: "PromptData" = None, template_content: Optional[str] = None,
        version: Optional[str] = None, registry_id: Optional[str] = None, template_engine_id: Optional[str] = None
    ) -> Optional[FormattedPrompt]:
        content_to_render: Optional[str] = template_content
        if content_to_render is None:
            if name is None:
                await self._trace("log.error", {"message": "render_prompt requires 'name' or 'template_content'.", "error": "MissingNameOrContent"})
                raise ValueError("Either 'name' or 'template_content' must be provided to render_prompt.")
            content_to_render = await self.get_raw_template(name, version, registry_id)
            if content_to_render is None:
                await self._trace("log.warning", {"message": f"Template '{name}' (version: {version}) not found in registry '{registry_id or self._default_registry_id}'. Cannot render."})
                return None

        # Use the provided engine_id (expected to be canonical) or the default (also expected to be canonical)
        engine_id_to_use = template_engine_id or self._default_template_engine_id

        if not engine_id_to_use:
            await self._trace("log.warning", {"message": "No prompt template engine configured. Attempting basic string.format() for render_prompt."})
            try:
                return content_to_render.format(**(data or {}))  # type: ignore
            except Exception as e_format_fallback:
                await self._trace("log.error", {"message": f"Basic string.format() fallback failed for render_prompt: {e_format_fallback}", "exc_info": True})
                return content_to_render  # Return unrendered template as last resort

        engine_config = self._template_configurations.get(engine_id_to_use, {})
        engine_any = await self._plugin_manager.get_plugin_instance(engine_id_to_use, config=engine_config)

        if not engine_any or not isinstance(engine_any, PromptTemplatePlugin):
            await self._trace("log.error", {"message": f"PromptTemplatePlugin '{engine_id_to_use}' not found or invalid for render_prompt.", "error": "InvalidTemplateEngine"})
            logger.error(f"Failed to load PromptTemplatePlugin '{engine_id_to_use}' for render_prompt. Cannot render.")
            return None

        engine = cast(PromptTemplatePlugin, engine_any)
        try:
            if content_to_render is None:
                await self._trace("log.error", {"message": "Internal error: content_to_render became None unexpectedly for render_prompt."})
                return None
            return await engine.render(content_to_render, data or {})
        except Exception as e_render:
            await self._trace("log.error", {"message": f"Error rendering prompt with engine '{engine_id_to_use}': {e_render}", "exc_info": True})
            return None

    async def render_chat_prompt(
        self, name: Optional[str] = None, data: "PromptData" = None, template_content: Optional[str] = None,
        version: Optional[str] = None, registry_id: Optional[str] = None, template_engine_id: Optional[str] = None
    ) -> Optional[List[ChatMessage]]:
        content_to_render: Optional[str] = template_content
        if content_to_render is None:
            if name is None:
                await self._trace("log.error", {"message": "render_chat_prompt requires 'name' or 'template_content'.", "error": "MissingNameOrContent"})
                raise ValueError("Either 'name' or 'template_content' must be provided to render_chat_prompt.")
            content_to_render = await self.get_raw_template(name, version, registry_id)
            if content_to_render is None:
                await self._trace("log.warning", {"message": f"Chat template '{name}' (version: {version}) not found in registry '{registry_id or self._default_registry_id}'. Cannot render."})
                return None

        # Use the provided engine_id (expected to be canonical) or the default (also expected to be canonical)
        engine_id_to_use = template_engine_id or self._default_template_engine_id

        if not engine_id_to_use:
            await self._trace("log.warning", {"message": "No prompt template engine configured for chat. Attempting basic string.format() fallback."})
            try:
                rendered_text = content_to_render.format(**(data or {})) # type: ignore
                return [{"role": "user", "content": rendered_text}]
            except Exception as e_format_fallback:
                await self._trace("log.error", {"message": f"Basic string.format() fallback failed for chat prompt: {e_format_fallback}", "exc_info": True})
                return [{"role": "user", "content": content_to_render}] # Return unrendered in user message

        engine_config = self._template_configurations.get(engine_id_to_use, {})
        engine_any = await self._plugin_manager.get_plugin_instance(engine_id_to_use, config=engine_config)

        if not engine_any or not isinstance(engine_any, PromptTemplatePlugin):
            await self._trace("log.error", {"message": f"PromptTemplatePlugin '{engine_id_to_use}' for chat not found or invalid.", "error": "InvalidTemplateEngine"})
            logger.error(f"Failed to load PromptTemplatePlugin '{engine_id_to_use}' for chat. Cannot render chat messages properly.")
            return [{"role": "system", "content": f"Error: Template engine '{engine_id_to_use}' for chat not found. Template content: {content_to_render[:100]}..."}]

        engine = cast(PromptTemplatePlugin, engine_any)
        try:
            if content_to_render is None:
                await self._trace("log.error", {"message": "Internal error: content_to_render became None unexpectedly for chat."})
                return None
            return await engine.render_chat_messages(content_to_render, data or {})
        except Exception as e_render_chat:
            await self._trace("log.error", {"message": f"Error rendering chat prompt with engine '{engine_id_to_use}': {e_render_chat}", "exc_info": True})
            return None

    async def list_available_templates(self, registry_id: Optional[str] = None) -> List[PromptIdentifier]:
        reg_id_to_use = registry_id or self._default_registry_id  # Assumes this is canonical
        if not reg_id_to_use:
            await self._trace("log.error", {"message": "No prompt registry ID specified for listing and no default is set."})
            return []
        registry_config = self._registry_configurations.get(reg_id_to_use, {})
        registry_any = await self._plugin_manager.get_plugin_instance(reg_id_to_use, config=registry_config)
        if not registry_any or not isinstance(registry_any, PromptRegistryPlugin):
            await self._trace("log.error", {"message": f"PromptRegistryPlugin '{reg_id_to_use}' for listing not found or invalid."})
            return []
        registry = cast(PromptRegistryPlugin, registry_any)
        try:
            return await registry.list_available_templates()
        except Exception as e:
            await self._trace("log.error", {"message": f"Error listing templates from registry '{reg_id_to_use}': {e}", "exc_info": True})
            return []


    async def teardown(self) -> None:
        await self._trace("log.info", {"message": "Tearing down..."})
        pass