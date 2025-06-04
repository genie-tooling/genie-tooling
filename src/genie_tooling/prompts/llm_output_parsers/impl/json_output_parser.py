# src/genie_tooling/prompts/llm_output_parsers/impl/json_output_parser.py
import json
import logging
import re
from typing import Any, Dict, Optional

from genie_tooling.prompts.llm_output_parsers.abc import LLMOutputParserPlugin
from genie_tooling.prompts.llm_output_parsers.types import ParsedOutput

logger = logging.getLogger(__name__)

class JSONOutputParserPlugin(LLMOutputParserPlugin):
    plugin_id: str = "json_output_parser_v1"
    description: str = "Parses LLM text output as JSON, attempting to extract a valid JSON object or array."

    _strict: bool = False

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._strict = bool(cfg.get("strict_parsing", False))
        logger.info(f"{self.plugin_id}: Initialized. Strict parsing: {self._strict}.")

    def _extract_json_block(self, text: str) -> Optional[str]:
        # 1. Try to find JSON within ```json ... ```
        code_block_match_json = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
        if code_block_match_json:
            potential_json = code_block_match_json.group(1).strip()
            try:
                json.loads(potential_json) # Validate
                logger.debug(f"{self.plugin_id}: Extracted JSON from ```json ... ``` block.")
                return potential_json
            except json.JSONDecodeError:
                logger.debug(f"{self.plugin_id}: Found ```json``` block, but content is not valid JSON: {potential_json[:100]}...")

        # 2. Try to find JSON within generic ``` ... ```
        code_block_match_generic = re.search(r"```\s*([\s\S]*?)\s*```", text, re.DOTALL)
        if code_block_match_generic:
            potential_json = code_block_match_generic.group(1).strip()
            if potential_json.startswith(("{", "[")): # Heuristic
                try:
                    json.loads(potential_json) # Validate
                    logger.debug(f"{self.plugin_id}: Extracted JSON from generic ``` ... ``` block.")
                    return potential_json
                except json.JSONDecodeError:
                    logger.debug(f"{self.plugin_id}: Found generic ``` ``` block, but content is not valid JSON: {potential_json[:100]}...")

        # 3. If no code block, try to find the first JSON object or array in the possibly "dirty" string
        stripped_text = text.strip() # Strip the whole text once for the general search
        decoder = json.JSONDecoder()

        first_obj_idx = stripped_text.find("{")
        first_arr_idx = stripped_text.find("[")

        start_indices = []
        if first_obj_idx != -1:
            start_indices.append(first_obj_idx)
        if first_arr_idx != -1:
            start_indices.append(first_arr_idx)

        if not start_indices:
            logger.debug(f"{self.plugin_id}: No '{'{'}' or '[' found in stripped text for general extraction.")
            return None

        start_indices.sort()

        for start_idx in start_indices:
            try:
                obj, end_idx = decoder.raw_decode(stripped_text[start_idx:])
                found_json_str = stripped_text[start_idx : start_idx + end_idx]
                logger.debug(f"{self.plugin_id}: Extracted JSON by raw_decode from stripped text: {found_json_str[:100]}...")
                return found_json_str
            except json.JSONDecodeError:
                logger.debug(f"{self.plugin_id}: No valid JSON found by raw_decode starting at index {start_idx} of stripped text. Text slice: {stripped_text[start_idx:start_idx+100]}...")
                continue

        logger.debug(f"{self.plugin_id}: Could not extract any valid JSON block from text by general search: {text[:200]}...")
        return None


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
            extracted_json_str = self._extract_json_block(text_output)

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
