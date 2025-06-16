# src/genie_tooling/prompts/abc.py
"""Abstract Base Classes for Prompt System Plugins."""
import logging
from typing import List, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin
from genie_tooling.llm_providers.types import ChatMessage

from .types import FormattedPrompt, PromptData, PromptIdentifier

logger = logging.getLogger(__name__)

@runtime_checkable
class PromptRegistryPlugin(Plugin, Protocol):
    plugin_id: str
    async def get_template_content(self, name: str, version: Optional[str] = None) -> Optional[str]:
        logger.warning(f"PromptRegistryPlugin '{self.plugin_id}' get_template_content not implemented.")
        return None
    async def list_available_templates(self) -> List[PromptIdentifier]:
        logger.warning(f"PromptRegistryPlugin '{self.plugin_id}' list_available_templates not implemented.")
        return []

@runtime_checkable
class PromptTemplatePlugin(Plugin, Protocol):
    plugin_id: str
    async def render(self, template_content: str, data: PromptData) -> FormattedPrompt:
        logger.warning(f"PromptTemplatePlugin '{self.plugin_id}' render not implemented.")
        return template_content.format(**data) if isinstance(data, dict) else template_content # type: ignore
    async def render_chat_messages(self, template_content: str, data: PromptData) -> List[ChatMessage]:
        logger.warning(f"PromptTemplatePlugin '{self.plugin_id}' render_chat_messages not implemented.")
        rendered_text_any = await self.render(template_content, data)
        rendered_text = str(rendered_text_any)
        return [{"role": "user", "content": rendered_text}]
