"""KeywordBlocklistGuardrailPlugin: Blocks input/output containing specific keywords."""
import json
import logging
from typing import Any, Dict, Optional, Set

from genie_tooling.guardrails.abc import InputGuardrailPlugin, OutputGuardrailPlugin
from genie_tooling.guardrails.types import GuardrailViolation

logger = logging.getLogger(__name__)

class KeywordBlocklistGuardrailPlugin(InputGuardrailPlugin, OutputGuardrailPlugin):
    plugin_id: str = "keyword_blocklist_guardrail_v1"
    description: str = "Checks input or output text against a configurable blocklist of keywords."
    default_action: str = "block" # Default action if a keyword is found

    _blocklist: Set[str]
    _case_sensitive: bool
    _action_on_match: str

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        raw_blocklist = cfg.get("blocklist", [])
        self._case_sensitive = bool(cfg.get("case_sensitive", False))
        self._action_on_match = cfg.get("action_on_match", self.default_action)
        if self._action_on_match not in ["allow", "block", "warn"]:
            logger.warning(f"{self.plugin_id}: Invalid action_on_match '{self._action_on_match}'. Defaulting to 'block'.")
            self._action_on_match = "block"

        if not self._case_sensitive:
            self._blocklist = {keyword.lower() for keyword in raw_blocklist if isinstance(keyword, str)}
        else:
            self._blocklist = {keyword for keyword in raw_blocklist if isinstance(keyword, str)}

        logger.info(f"{self.plugin_id}: Initialized with {len(self._blocklist)} keywords. Case sensitive: {self._case_sensitive}. Action on match: {self._action_on_match}.")

    def _check_text(self, text: str) -> Optional[str]:
        """Checks if any keyword from the blocklist is in the text."""
        if not self._blocklist:
            return None

        text_to_check = text if self._case_sensitive else text.lower()

        # Simple substring check for each keyword
        for keyword in self._blocklist:
            if keyword in text_to_check:
                return keyword # Return the matched keyword
        return None

    async def check_input(self, data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailViolation:
        text_to_check = ""
        if isinstance(data, str):
            text_to_check = data
        elif isinstance(data, dict) and "content" in data and isinstance(data["content"], str): # Common for ChatMessage
            text_to_check = data["content"]
        elif isinstance(data, list): # Check content of ChatMessages in a list
            for item in data:
                if isinstance(item, dict) and "content" in item and isinstance(item["content"], str):
                    matched_keyword = self._check_text(item["content"])
                    if matched_keyword:
                        return GuardrailViolation(action=self._action_on_match, reason=f"Blocked input keyword: '{matched_keyword}'", guardrail_id=self.plugin_id)
            return GuardrailViolation(action="allow", reason="All input items passed keyword check.")
        else: # Cannot determine text to check
            return GuardrailViolation(action="allow", reason="Input data format not recognized for keyword check.")

        matched_keyword = self._check_text(text_to_check)
        if matched_keyword:
            return GuardrailViolation(action=self._action_on_match, reason=f"Blocked input keyword: '{matched_keyword}'", guardrail_id=self.plugin_id)
        return GuardrailViolation(action="allow", reason="Input passed keyword check.")

    async def check_output(self, data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailViolation:
        text_to_check = ""
        if isinstance(data, str):
            text_to_check = data
        elif isinstance(data, dict): # Common for LLM responses or tool results
            # Try to find a 'text' or 'content' field, or serialize the whole dict
            if "text" in data and isinstance(data["text"], str):
                text_to_check = data["text"]
            elif "content" in data and isinstance(data["content"], str):
                text_to_check = data["content"]
            elif "message" in data and isinstance(data["message"], dict) and "content" in data["message"] and isinstance(data["message"]["content"], str): # LLMChatResponse
                 text_to_check = data["message"]["content"]
            else:
                try:
                    text_to_check = json.dumps(data) # Fallback to checking serialized dict
                except Exception as e:
                    logger.warning(f"Got an exception with {e}")
                    text_to_check = str(data)
        else: # Cannot determine text to check
            return GuardrailViolation(action="allow", reason="Output data format not recognized for keyword check.")

        matched_keyword = self._check_text(text_to_check)
        if matched_keyword:
            return GuardrailViolation(action=self._action_on_match, reason=f"Blocked output keyword: '{matched_keyword}'", guardrail_id=self.plugin_id)
        return GuardrailViolation(action="allow", reason="Output passed keyword check.")

    async def teardown(self) -> None:
        self._blocklist.clear()
        logger.debug(f"{self.plugin_id}: Teardown complete, blocklist cleared.")
