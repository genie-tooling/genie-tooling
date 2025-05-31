# src/genie_tooling/prompts/impl/jinja2_chat_template.py
import json
import logging
from typing import Any, Dict, List, Optional

from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.prompts.abc import PromptTemplatePlugin
from genie_tooling.prompts.types import FormattedPrompt, PromptData

logger = logging.getLogger(__name__)

try:
    from jinja2 import Environment, select_autoescape, FileSystemLoader as JinjaFileSystemLoader, TemplateSyntaxError, UndefinedError
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Environment = None
    select_autoescape = None
    JinjaFileSystemLoader = None
    TemplateSyntaxError = Exception
    UndefinedError = Exception


class Jinja2ChatTemplatePlugin(PromptTemplatePlugin):
    plugin_id: str = "jinja2_chat_template_v1"
    description: str = "Renders complex chat prompt templates using the Jinja2 templating engine."

    _env: Optional[Any] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not JINJA2_AVAILABLE:
            logger.error(f"{self.plugin_id}: Jinja2 library not installed. This plugin will not function.")
            return
        
        cfg = config or {}
        # Jinja2 Environment can be configured further (e.g., custom filters, extensions)
        # For now, a basic environment.
        # If templates are loaded from files by a FileSystemPromptRegistry, the content is passed directly.
        # If this plugin were to load its own templates, FileSystemLoader would be configured here.
        self._env = Environment(autoescape=select_autoescape()) # Basic env
        logger.info(f"{self.plugin_id}: Initialized with Jinja2.")

    async def render(self, template_content: str, data: PromptData) -> FormattedPrompt:
        if not self._env:
            logger.error(f"{self.plugin_id}: Jinja2 environment not initialized.")
            return template_content # Fallback

        if not isinstance(data, dict):
            logger.warning(f"{self.plugin_id}: Data for rendering is not a dictionary (type: {type(data)}). Using empty dict.")
            data_dict = {}
        else:
            data_dict = data
        
        try:
            template = self._env.from_string(template_content)
            rendered_text = template.render(**data_dict)
            return rendered_text
        except (TemplateSyntaxError, UndefinedError) as e:
            logger.error(f"{self.plugin_id}: Error rendering Jinja2 template (for string output): {e}. Template: '{template_content[:100]}...', Data: {data_dict}", exc_info=True)
            return f"Error rendering template: {e}" # Fallback
        except Exception as e:
            logger.error(f"{self.plugin_id}: Unexpected error rendering Jinja2 template: {e}", exc_info=True)
            return f"Unexpected error rendering template: {e}"


    async def render_chat_messages(self, template_content: str, data: PromptData) -> List[ChatMessage]:
        if not self._env:
            logger.error(f"{self.plugin_id}: Jinja2 environment not initialized for chat messages.")
            return [{"role": "user", "content": "Error: Jinja2 environment not ready."}]

        if not isinstance(data, dict):
            logger.warning(f"{self.plugin_id}: Data for chat rendering is not a dictionary (type: {type(data)}). Using empty dict.")
            data_dict = {}
        else:
            data_dict = data

        try:
            # First, render the template content as a string.
            # This string is expected to be a JSON representation of chat messages.
            template = self._env.from_string(template_content)
            rendered_json_str = template.render(**data_dict)

            # Then, parse the rendered string as JSON into a list of ChatMessage.
            try:
                chat_messages = json.loads(rendered_json_str)
                if not isinstance(chat_messages, list):
                    logger.error(f"{self.plugin_id}: Rendered Jinja2 template for chat did not produce a JSON list. Output: {rendered_json_str[:200]}...")
                    return [{"role": "user", "content": f"Error: Template did not produce a list of messages. Output: {rendered_json_str[:100]}..."}]
                
                # Basic validation of message structure (can be more thorough)
                validated_messages: List[ChatMessage] = []
                for msg in chat_messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                         validated_messages.append(ChatMessage(role=msg["role"], content=msg["content"])) # type: ignore
                    elif isinstance(msg, dict) and "role" in msg and msg.get("tool_calls"): # For tool calls
                         validated_messages.append(ChatMessage(role=msg["role"], content=msg.get("content"), tool_calls=msg["tool_calls"])) # type: ignore
                    else:
                        logger.warning(f"{self.plugin_id}: Invalid message structure in rendered chat JSON: {msg}. Skipping.")
                return validated_messages

            except json.JSONDecodeError as e_json:
                logger.error(f"{self.plugin_id}: Failed to parse rendered Jinja2 output as JSON for chat messages: {e_json}. Output: {rendered_json_str[:200]}...", exc_info=True)
                return [{"role": "user", "content": f"Error: Template output is not valid JSON. Output: {rendered_json_str[:100]}..."}]

        except (TemplateSyntaxError, UndefinedError) as e_render:
            logger.error(f"{self.plugin_id}: Error rendering Jinja2 template (for chat output): {e_render}. Template: '{template_content[:100]}...', Data: {data_dict}", exc_info=True)
            return [{"role": "user", "content": f"Error rendering chat template: {e_render}"}]
        except Exception as e:
            logger.error(f"{self.plugin_id}: Unexpected error rendering Jinja2 chat template: {e}", exc_info=True)
            return [{"role": "user", "content": f"Unexpected error rendering chat template: {e}"}]

    async def teardown(self) -> None:
        self._env = None
        logger.debug(f"{self.plugin_id}: Teardown complete.")
