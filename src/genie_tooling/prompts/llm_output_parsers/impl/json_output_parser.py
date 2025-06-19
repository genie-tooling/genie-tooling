# src/genie_tooling/prompts/llm_output_parsers/impl/json_output_parser.py
import json
import logging
from typing import Any, Dict, Optional

from genie_tooling.prompts.llm_output_parsers.abc import LLMOutputParserPlugin
from genie_tooling.prompts.llm_output_parsers.types import ParsedOutput
from genie_tooling.utils.json_parser_utils import extract_json_block

logger = logging.getLogger(__name__)

class JSONOutputParserPlugin(LLMOutputParserPlugin):
    plugin_id: str = "json_output_parser_v1"
    description: str = "Parses LLM text output as JSON, attempting to extract a valid JSON object or array."

    _strict: bool = False

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._strict = bool(cfg.get("strict_parsing", False))
        logger.info(f"{self.plugin_id}: Initialized. Strict parsing: {self._strict}.")

    def parse(self, text_output: str, schema: Optional[Any] = None) -> ParsedOutput:
        if not text_output or not text_output.strip():
            raise ValueError("Input text_output is empty or whitespace.")

        if self._strict:
            try:
                return json.loads(text_output) # type: ignore
            except json.JSONDecodeError as e:
                logger.warning(f"{self.plugin_id}: Strict parsing failed. Invalid JSON: {e.msg}. Input: '{text_output[:100]}...'")
                raise ValueError(f"Strict JSON parsing failed: {e.msg}") from e
        else:
            extracted_json_str = extract_json_block(text_output)

            if extracted_json_str:
                try:
                    return json.loads(extracted_json_str) # type: ignore
                except json.JSONDecodeError as e:
                    logger.warning(f"{self.plugin_id}: Extracted block is not valid JSON: {e.msg}. Extracted: '{extracted_json_str[:100]}...'")
                    raise ValueError(f"Extracted JSON block is invalid: {e.msg}") from e
            else:
                logger.warning(f"{self.plugin_id}: No JSON block found in text_output: '{text_output[:100]}...'")
                raise ValueError("No parsable JSON block found in the input text.")

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Teardown complete.")
