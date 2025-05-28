"""JSONOutputTransformer: Ensures the output is JSON-serializable."""
import json
import logging
from typing import Any, Dict, Optional

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
        """
        Attempts to serialize the output to a JSON string and then optionally
        parse it back to a Python dict/list if output_format is 'dict'.
        This primarily ensures JSON serializability.
        The 'schema' argument is present for interface compliance but not heavily used
        by this simple transformer beyond basic type checks it might imply.
        """
        try:
            json_string = json.dumps(output)

            if self.output_format == "string":
                logger.debug("JSONOutputTransformer: Output transformed to JSON string.")
                return json_string
            else: # output_format == "dict" (or list)
                parsed_output = json.loads(json_string)
                if not isinstance(parsed_output, (dict, list)):
                    msg = f"Output serialized to JSON but did not yield a dict/list: {type(parsed_output)}"
                    logger.warning(f"JSONOutputTransformer: {msg}")
                    raise OutputTransformationException(msg, original_output=output)
                logger.debug(f"JSONOutputTransformer: Output transformed to Python {type(parsed_output)}.")
                return parsed_output

        except (TypeError, OverflowError) as e_serialize: # Errors from json.dumps
            msg = f"Output is not JSON serializable: {str(e_serialize)}"
            logger.error(f"JSONOutputTransformer: {msg}", exc_info=True)
            raise OutputTransformationException(msg, original_output=output) from e_serialize
        except json.JSONDecodeError as e_decode: # Error from json.loads (should be rare if dumps succeeded)
            msg = f"Internal error: Output serialized but failed to decode back from JSON: {str(e_decode)}"
            logger.error(f"JSONOutputTransformer: {msg}", exc_info=True)
            raise OutputTransformationException(msg, original_output=output) from e_decode
        except Exception as e:
            msg = f"Unexpected error during JSON transformation: {str(e)}"
            logger.error(f"JSONOutputTransformer: {msg}", exc_info=True)
            raise OutputTransformationException(msg, original_output=output) from e

    # async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    # async def teardown(self) -> None: pass
