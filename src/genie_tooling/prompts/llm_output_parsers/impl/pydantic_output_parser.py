# src/genie_tooling/prompts/llm_output_parsers/impl/pydantic_output_parser.py
import json
import logging
import re
from typing import Any, Dict, Optional, Type

from genie_tooling.prompts.llm_output_parsers.abc import LLMOutputParserPlugin
from genie_tooling.prompts.llm_output_parsers.types import ParsedOutput

logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None # type: ignore
    ValidationError = None # type: ignore
    PYDANTIC_AVAILABLE = False

class PydanticOutputParserPlugin(LLMOutputParserPlugin):
    plugin_id: str = "pydantic_output_parser_v1"
    description: str = "Parses LLM text output (expected to be JSON) into a Pydantic model instance."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not PYDANTIC_AVAILABLE:
            logger.error(f"{self.plugin_id}: Pydantic library not installed. This plugin will not function.")
        logger.info(f"{self.plugin_id}: Initialized.")

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
        if not PYDANTIC_AVAILABLE or not BaseModel or not ValidationError:
            raise RuntimeError(f"{self.plugin_id}: Pydantic library not available at runtime.")

        if schema is None or not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise ValueError(f"{self.plugin_id}: A Pydantic model class must be provided as the 'schema' argument.")

        pydantic_model_cls: Type[BaseModel] = schema

        if not text_output or not text_output.strip():
            raise ValueError("Input text_output is empty or whitespace.")

        json_str_to_parse = self._extract_json_block(text_output)

        if not json_str_to_parse:
            logger.warning(f"{self.plugin_id}: No valid JSON block found in text_output: '{text_output[:200]}...'")
            raise ValueError("No parsable JSON block found in the input text for Pydantic parsing.")

        try:
            # Use model_validate_json for Pydantic v2+
            parsed_model = pydantic_model_cls.model_validate_json(json_str_to_parse)
            return parsed_model
        except ValidationError as e_pydantic:
            logger.warning(f"{self.plugin_id}: Pydantic validation failed for model '{pydantic_model_cls.__name__}'. Errors: {e_pydantic.errors()}. Input JSON: '{json_str_to_parse[:200]}...'")
            raise ValueError(f"Pydantic validation failed: {e_pydantic.errors()}") from e_pydantic
        except json.JSONDecodeError as e_json:
            logger.warning(f"{self.plugin_id}: Extracted block is not valid JSON: {e_json.msg}. Extracted: '{json_str_to_parse[:100]}...'")
            raise ValueError(f"Extracted JSON block is invalid: {e_json.msg}") from e_json
        except Exception as e:
            logger.error(f"{self.plugin_id}: Unexpected error during Pydantic parsing: {e}", exc_info=True)
            raise ValueError(f"Unexpected Pydantic parsing error: {e!s}") from e

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Teardown complete.")
