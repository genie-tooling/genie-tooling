# src/genie_tooling/prompts/impl/basic_string_format_template.py
import logging
from typing import Any, Dict, List, Optional

from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.prompts.abc import PromptTemplatePlugin
from genie_tooling.prompts.types import FormattedPrompt, PromptData

logger = logging.getLogger(__name__)

class BasicStringFormatTemplatePlugin(PromptTemplatePlugin):
    plugin_id: str = "basic_string_format_template_v1"
    description: str = "Renders prompt templates using Python's basic string.format(**kwargs) method."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        logger.info(f"{self.plugin_id}: Initialized.")

    async def render(self, template_content: str, data: PromptData) -> FormattedPrompt:
        if not isinstance(data, dict):
            logger.warning(f"{self.plugin_id}: Data for rendering is not a dictionary (type: {type(data)}). Using empty dict.")
            data_dict = {}
        else:
            data_dict = data

        try:
            rendered_text = template_content.format(**data_dict)
            return rendered_text
        except KeyError as e:
            logger.error(f"{self.plugin_id}: Missing key '{e}' in data for template. Template: '{template_content[:100]}...', Data: {data_dict}")
            # Fallback: return template with placeholders visible
            return template_content 
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error rendering template: {e}. Template: '{template_content[:100]}...', Data: {data_dict}", exc_info=True)
            return template_content # Fallback

    async def render_chat_messages(self, template_content: str, data: PromptData) -> List[ChatMessage]:
        # This basic formatter assumes the template_content is a single user message.
        # More complex chat structures would need a more sophisticated template format (e.g., Jinja2 or JSON).
        rendered_text = await self.render(template_content, data)
        if isinstance(rendered_text, str):
            return [{"role": "user", "content": rendered_text}]
        else: # Should not happen if render returns string
            logger.error(f"{self.plugin_id}: render method did not return a string for chat message content. Got: {type(rendered_text)}")
            return [{"role": "user", "content": str(rendered_text)}]


    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Teardown complete.")
