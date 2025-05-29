"""PassThroughOutputTransformer: An output transformer that returns the output as-is."""
import logging
from typing import Any, Dict

# Updated import path for OutputTransformer
from genie_tooling.output_transformers.abc import OutputTransformer

logger = logging.getLogger(__name__)

class PassThroughOutputTransformer(OutputTransformer):
    """An output transformer that returns the output as-is, without modification."""
    plugin_id: str = "passthrough_output_transformer_v1"
    description: str = "Returns the tool output directly without any transformation."

    def transform(self, output: Any, schema: Dict[str, Any]) -> Any:
        logger.debug(f"PassThroughOutputTransformer: Output type {type(output)} passed through.")
        return output

    # async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    # async def teardown(self) -> None: pass
