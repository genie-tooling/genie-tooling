### src/genie_tooling/output_transformers/impl/json_transformer.py
import json
import logging
from typing import Any, Dict

# Updated import paths for OutputTransformer and OutputTransformationException
from genie_tooling.output_transformers.abc import (
    OutputTransformationException,
    OutputTransformer,
)

logger = logging.getLogger(__name__)

class JSONOutputTransformer(OutputTransformer):
    plugin_id: str = "json_output_transformer_v1"
    description: str = "Ensures the output is a JSON-serializable dictionary or list, or a JSON string."

    def __init__(self, output_format: str = "dict"): # "dict" or "string"
        if output_format not in ["dict", "string"]:
            raise ValueError("JSONOutputTransformer output_format must be 'dict' or 'string'")
        self.output_format = output_format
        logger.debug(f"JSONOutputTransformer initialized to output format: {self.output_format}")

    def transform(self, output: Any, schema: Dict[str, Any]) -> Any:
        try:
            if self.output_format == "dict":
                # Case 1: Input is already a dict or list.
                # We must still ensure it's JSON serializable.
                if isinstance(output, (dict, list)):
                    try:
                        # Attempt to serialize and deserialize to confirm validity and structure.
                        json_string_check = json.dumps(output)
                        parsed_after_check = json.loads(json_string_check)
                        # Ensure it's still a dict/list after dump/load (should be, but good check)
                        if not isinstance(parsed_after_check, (dict, list)):
                            # This case is highly unlikely if json.dumps succeeded on a dict/list.
                            msg = f"Internal check: Original dict/list became {type(parsed_after_check)} after dump/load."
                            logger.error(f"JSONOutputTransformer: {msg}")
                            raise OutputTransformationException(msg, original_output=output)
                        logger.debug("JSONOutputTransformer: Input was already dict/list and is serializable.")
                        return parsed_after_check # Return the parsed version
                    except (TypeError, OverflowError) as e_serialize_check:
                        # This is where non-serializable elements within the dict/list are caught.
                        msg = f"Input {type(output).__name__} is not JSON serializable: {e_serialize_check!s}"
                        logger.error(f"JSONOutputTransformer: {msg}", exc_info=True)
                        raise OutputTransformationException(msg, original_output=output) from e_serialize_check

                # Case 2: Input is a string. Try to parse it.
                elif isinstance(output, str):
                    try:
                        parsed_from_string_input = json.loads(output)
                        if isinstance(parsed_from_string_input, (dict, list)):
                            logger.debug("JSONOutputTransformer: Input string was valid JSON dict/list, returning parsed.")
                            return parsed_from_string_input
                        else:
                            # Input string was valid JSON, but not a dict or list.
                            msg = f"Input string parsed to JSON type '{type(parsed_from_string_input).__name__}', but expected dict/list for 'dict' output_format."
                            logger.warning(f"JSONOutputTransformer: {msg}")
                            raise OutputTransformationException(msg, original_output=output)
                    except json.JSONDecodeError as e_decode_str:
                        # Input string was not valid JSON at all.
                        msg = f"Input string is not valid JSON: {e_decode_str!s}"
                        logger.warning(f"JSONOutputTransformer: {msg}")
                        raise OutputTransformationException(msg, original_output=output) from e_decode_str

                # Case 3: Input is neither dict/list nor string. Try to dump and load.
                else:
                    json_string_from_dump = json.dumps(output) # This might raise TypeError
                    parsed_output = json.loads(json_string_from_dump)
                    if not isinstance(parsed_output, (dict, list)):
                        msg = f"Output (type {type(output).__name__}) serialized to JSON but did not yield a dict/list (became {type(parsed_output).__name__})."
                        logger.warning(f"JSONOutputTransformer: {msg}")
                        raise OutputTransformationException(msg, original_output=output)
                    logger.debug(f"JSONOutputTransformer: Output (type {type(output).__name__}) transformed to Python {type(parsed_output).__name__}.")
                    return parsed_output

            else: # output_format == "string"
                # Always dump the input to get its JSON string representation.
                json_string = json.dumps(output)
                logger.debug("JSONOutputTransformer: Output transformed to JSON string.")
                return json_string

        except (TypeError, OverflowError) as e_serialize: # Catches json.dumps errors from Case 3 or initial string dump
            msg = f"Output is not JSON serializable: {e_serialize!s}"
            logger.error(f"JSONOutputTransformer: {msg}", exc_info=True)
            raise OutputTransformationException(msg, original_output=output) from e_serialize
        except json.JSONDecodeError as e_decode: # Catches json.loads errors from Case 3
            msg = f"Internal error: Output serialized but failed to decode back from JSON: {e_decode!s}"
            logger.error(f"JSONOutputTransformer: {msg}", exc_info=True)
            raise OutputTransformationException(msg, original_output=output) from e_decode
        except OutputTransformationException: # Re-raise if already handled
            raise
        except Exception as e: # Catch-all for other unexpected errors
            msg = f"Unexpected error during JSON transformation: {e!s}"
            logger.error(f"JSONOutputTransformer: {msg}", exc_info=True)
            raise OutputTransformationException(msg, original_output=output) from e

    # async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    # async def teardown(self) -> None: pass
