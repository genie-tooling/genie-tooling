# src/genie_tooling/prompts/llm_output_parsers/impl/pydantic_output_parser.py
import json
import logging
import re
from typing import Any, Dict, Optional, Type, Union

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
        # (Same as JSONOutputParserPlugin's _extract_json_block for consistency)
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        arr_match = re.search(r"\[.*\]", text, re.DOTALL)
        potential_json_str: Optional[str] = None
        if obj_match:
            potential_json_str = obj_match.group(0)
            try: json.loads(potential_json_str); return potential_json_str
            except json.JSONDecodeError: potential_json_str = None
        if arr_match:
            array_candidate = arr_match.group(0)
            try:
                json.loads(array_candidate)
                if potential_json_str is None or (obj_match and arr_match.start() < obj_match.start()):
                    return array_candidate
                elif potential_json_str is None: return array_candidate
            except json.JSONDecodeError: pass
        return potential_json_str if potential_json_str else None


    def parse(self, text_output: str, schema: Optional[Any] = None) -> ParsedOutput:
        if not PYDANTIC_AVAILABLE or not BaseModel or not ValidationError:
            raise RuntimeError(f"{self.plugin_id}: Pydantic library not available at runtime.")
        
        if schema is None or not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise ValueError(f"{self.plugin_id}: A Pydantic model class must be provided as the 'schema' argument.")
        
        pydantic_model_cls: Type[BaseModel] = schema

        if not text_output or not text_output.strip():
            raise ValueError("Input text_output is empty or whitespace.")

        # Attempt to extract JSON from markdown or general text
        code_block_patterns = [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]
        json_str_to_parse: Optional[str] = None

        for pattern in code_block_patterns:
            match = re.search(pattern, text_output, re.DOTALL)
            if match:
                potential_json = match.group(1).strip()
                try:
                    # Quick validation if it's JSON before full Pydantic parsing
                    json.loads(potential_json) 
                    json_str_to_parse = potential_json
                    break
                except json.JSONDecodeError:
                    logger.debug(f"{self.plugin_id}: Found code block, but content is not valid JSON. Content: '{potential_json[:100]}...'")
                    continue 
        
        if not json_str_to_parse:
            json_str_to_parse = self._extract_json_block(text_output)

        if not json_str_to_parse:
            logger.warning(f"{self.plugin_id}: No valid JSON block found in text_output: '{text_output[:100]}...'")
            raise ValueError("No parsable JSON block found in the input text for Pydantic parsing.")

        try:
            # Pydantic v2 can parse directly from a JSON string using model_validate_json
            # For Pydantic v1, it would be: data = json.loads(json_str_to_parse); return pydantic_model_cls(**data)
            if hasattr(pydantic_model_cls, 'model_validate_json'): # Pydantic v2+
                parsed_model = pydantic_model_cls.model_validate_json(json_str_to_parse)
            else: # Pydantic v1 fallback (less common now)
                data_dict = json.loads(json_str_to_parse)
                parsed_model = pydantic_model_cls(**data_dict)
            
            return parsed_model # type: ignore
        except ValidationError as e_pydantic:
            logger.warning(f"{self.plugin_id}: Pydantic validation failed for model '{pydantic_model_cls.__name__}'. Errors: {e_pydantic.errors()}. Input JSON: '{json_str_to_parse[:200]}...'")
            raise ValueError(f"Pydantic validation failed: {e_pydantic.errors()}") from e_pydantic
        except json.JSONDecodeError as e_json: # Should be caught by earlier checks, but as a safeguard
            logger.warning(f"{self.plugin_id}: Extracted block is not valid JSON (should have been caught earlier): {e_json.msg}. Extracted: '{json_str_to_parse[:100]}...'")
            raise ValueError(f"Extracted JSON block is invalid: {e_json.msg}") from e_json
        except Exception as e:
            logger.error(f"{self.plugin_id}: Unexpected error during Pydantic parsing: {e}", exc_info=True)
            raise ValueError(f"Unexpected Pydantic parsing error: {str(e)}") from e

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Teardown complete.")
