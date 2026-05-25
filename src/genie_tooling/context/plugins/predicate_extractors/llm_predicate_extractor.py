# FILE: genie_tooling.context/src/genie_tooling.context/plugins/predicate_extractors/llm_predicate_extractor.py
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, Field

from genie_tooling.context.protocols import PredicateExtractorPlugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


class PredicateModel(BaseModel):
    """Schema for the LLM's predicate extraction output."""
    predicate: str = Field(
        description="A single, concise predicate string in 'predicate_verb' format (e.g., 'predicate_calculate', 'predicate_summarize', 'predicate_what')."
    )


class LLMPredicateExtractorPlugin(PredicateExtractorPlugin):
    """Extracts a logical predicate from a query using a constrained LLM call."""

    plugin_id: str = "llm_predicate_extractor_v1"
    description: str = "Uses an LLM to reliably extract a logical predicate from a user query."

    _model_id: Optional[str] = None
    _default_predicate: str = "predicate_generic_inquiry"

    async def setup(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self._model_id = cfg.get("model")
        logger.info(
            f"[{self.plugin_id}] Initialized. LLM model override: {self._model_id or 'None'}."
        )

    async def extract(self, query: str, genie: "Genie") -> str:
        # FIX: Added a more forceful and explicit instruction.
        prompt = f"""
        Analyze the user's query and determine its primary logical intent.
        Express this intent as a simple 'predicate_verb' string.

        Examples:
        - "what is the capital of france?" -> "predicate_what"
        - "calculate 25 * 4" -> "predicate_calculate"
        - "who is the ceo of openai?" -> "predicate_who"
        - "summarize this article for me" -> "predicate_summarize"
        - "tell me a joke" -> "predicate_generic_inquiry"

        User Query: "{query}"

        Your response MUST be a single JSON object that conforms to the required schema, containing the key "predicate".
        DO NOT add any conversational text, comments, or explanations before or after the JSON object.
        """

        try:
            llm_response = await genie.llm.chat(
                [{"role": "user", "content": prompt}],
                provider_id=self._model_id,
                output_schema=PredicateModel,
            )
            parsed_output = await genie.llm.parse_output(
                llm_response,
                parser_id="pydantic_output_parser_v1",
                schema=PredicateModel,
            )

            if isinstance(parsed_output, PredicateModel) and parsed_output.predicate:
                logger.info(
                    f"[{self.plugin_id}] Extracted predicate '{parsed_output.predicate}' for query."
                )
                return parsed_output.predicate

            logger.warning(
                f"[{self.plugin_id}] LLM failed to return a valid predicate. Defaulting."
            )
            return self._default_predicate

        except Exception as e:
            logger.error(
                f"[{self.plugin_id}] Predicate extraction failed: {e}. Defaulting.",
                exc_info=True,
            )
            return self._default_predicate
