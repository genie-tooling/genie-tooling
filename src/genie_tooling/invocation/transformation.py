"""Output Transformation components."""
import json  # For JSONOutputTransformer example
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

logger = logging.getLogger(__name__)

class OutputTransformationException(ValueError):
    """Custom exception for output transformation errors."""
    def __init__(self, message: str, original_output: Any = None, schema: Optional[Dict[str,Any]] = None):
        super().__init__(message)
        self.original_output = original_output
        self.schema = schema

@runtime_checkable
class OutputTransformer(Plugin, Protocol):
    """Protocol for output transformers."""
    # plugin_id: str (from Plugin protocol)

    def transform(self, output: Any, schema: Dict[str, Any]) -> Any:
        """
        Transforms raw tool output according to an output_schema or desired format.
        Raises OutputTransformationException on failure.
        This method is synchronous as transformation is typically CPU-bound.
        """
        ...

class PassThroughOutputTransformer(OutputTransformer):
    """An output transformer that returns the output as-is, without modification."""
    plugin_id: str = "passthrough_output_transformer_v1"
    description: str = "Returns the tool output directly without any transformation."

    def transform(self, output: Any, schema: Dict[str, Any]) -> Any:
        logger.debug(f"PassThroughOutputTransformer: Output type {type(output)} passed through.")
        return output

    # async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    # async def teardown(self) -> None: pass


class JSONOutputTransformer(OutputTransformer): # Example of a more specific transformer
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
            # First, try to serialize to check if it's JSON serializable
            json_string = json.dumps(output)

            if self.output_format == "string":
                logger.debug("JSONOutputTransformer: Output transformed to JSON string.")
                return json_string
            else: # output_format == "dict" (or list)
                # Parse back to ensure it's a valid JSON structure (dict or list at top level)
                parsed_output = json.loads(json_string)
                if not isinstance(parsed_output, (dict, list)):
                    # This case should be rare if json.dumps succeeded and output was complex.
                    # It might happen if output was a simple type like int/str/bool stringified.
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
