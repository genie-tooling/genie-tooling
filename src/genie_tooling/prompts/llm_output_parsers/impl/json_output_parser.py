### src/genie_tooling/prompts/llm_output_parsers/impl/json_output_parser.py
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

    _strict: bool = False # If true, requires the entire string to be valid JSON.

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._strict = bool(cfg.get("strict_parsing", False))
        logger.info(f"{self.plugin_id}: Initialized. Strict parsing: {self._strict}.")

    def _extract_json_block(self, text: str) -> Optional[str]:
        """
        Extracts the first valid JSON object or array string from text.
        Prioritizes JSON within ```json ... ```, then ``` ... ```,
        then looks for the first complete JSON object or array.
        """
        # 1. Try to find JSON within ```json ... ``` (DOTALL for multiline JSON)
        code_block_match_json = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
        if code_block_match_json:
            potential_json = code_block_match_json.group(1).strip()
            try:
                json.loads(potential_json)
                logger.debug(f"{self.plugin_id}: Extracted JSON from ```json ... ``` block.")
                return potential_json
            except json.JSONDecodeError:
                logger.debug(f"{self.plugin_id}: Found ```json``` block, but content is not valid JSON: {potential_json[:100]}...")

        # 2. Try to find JSON within generic ``` ... ```
        code_block_match_generic = re.search(r"```\s*([\s\S]*?)\s*```", text, re.DOTALL)
        if code_block_match_generic:
            potential_json = code_block_match_generic.group(1).strip()
            # Check if it starts with { or [ as a heuristic for JSON
            if potential_json.startswith(("{", "[")):
                try:
                    json.loads(potential_json)
                    logger.debug(f"{self.plugin_id}: Extracted JSON from generic ``` ... ``` block.")
                    return potential_json
                except json.JSONDecodeError:
                    logger.debug(f"{self.plugin_id}: Found generic ``` ``` block, but content is not valid JSON: {potential_json[:100]}...")

        # 3. Try to find the first complete JSON object or array using a robust approach.
        # Find the first '{' or '[' and try to parse progressively larger substrings.
        first_obj_brace_idx = text.find("{")
        first_arr_brace_idx = text.find("[")

        start_indices = []
        if first_obj_brace_idx != -1:
            start_indices.append(first_obj_brace_idx)
        if first_arr_brace_idx != -1:
            start_indices.append(first_arr_brace_idx)

        if not start_indices:
            logger.debug(f"{self.plugin_id}: No '{'{'}' or '[' found in text for general extraction.")
            return None

        start_indices.sort() # Process in order of appearance

        for start_idx in start_indices:
            open_chars = 0
            # Determine if we are looking for an object or array based on the starting char
            start_char = text[start_idx]
            expected_close_char = "}" if start_char == "{" else "]"

            for i in range(start_idx, len(text)):
                current_char = text[i]
                if current_char == start_char: # Counts opening character ({ or [)
                    open_chars += 1
                elif current_char == expected_close_char:
                    open_chars -= 1
                    if open_chars == 0: # Found a potentially balanced block
                        potential_json_block = text[start_idx : i + 1]
                        try:
                            json.loads(potential_json_block)
                            logger.debug(f"{self.plugin_id}: Extracted JSON by general block search: {potential_json_block[:100]}...")
                            return potential_json_block
                        except json.JSONDecodeError:
                            # This balanced block wasn't valid JSON.
                            # If we started with '{', and this failed, we don't want to accidentally
                            # match a later '[' that might be part of this malformed object.
                            # So, we break from this specific start_idx attempt.
                            # The outer loop will try the next start_idx (e.g., if an array started later).
                            break
            # If the loop finishes for a start_idx and no balanced valid JSON was found

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
