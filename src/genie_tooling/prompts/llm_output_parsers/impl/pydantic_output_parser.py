# src/genie_tooling/prompts/llm_output_parsers/impl/pydantic_output_parser.py
import json
import logging
from typing import Any, Dict, Optional, Type

from genie_tooling.prompts.llm_output_parsers.abc import LLMOutputParserPlugin
from genie_tooling.prompts.llm_output_parsers.types import ParsedOutput
from genie_tooling.utils.json_parser_utils import extract_json_block

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

    def parse(self, text_output: str, schema: Optional[Any] = None) -> ParsedOutput:
        if not PYDANTIC_AVAILABLE or not BaseModel or not ValidationError:
            raise RuntimeError(f"{self.plugin_id}: Pydantic library not available at runtime.")

        if schema is None or not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise ValueError(f"{self.plugin_id}: A Pydantic model class must be provided as the 'schema' argument.")

        pydantic_model_cls: Type[BaseModel] = schema

        if not text_output or not text_output.strip():
            raise ValueError("Input text_output is empty or whitespace.")

        json_str_to_parse = extract_json_block(text_output)

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
