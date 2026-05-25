# FILE: genie_tooling.context/src/genie_tooling.context/plugins/inference/llm_inference.py
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, Field

from genie_tooling.context.protocols import ContextInferencePlugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


class InferredContextModel(BaseModel):
    """Pydantic model for the structured output of the context inference LLM call."""

    audience_expertise: str = Field(
        description="Layperson, Novice, Expert, or Unknown."
    )
    audience_state: str = Field(
        description="Stressed, Curious, Neutral, Agitated, or Unknown."
    )
    discourse_topic: str = Field(
        description="A concise, 2-3 word summary of the main topic."
    )
    intent: str = Field(
        description="The user's likely intent, e.g., 'reassurance_seeking', 'fact_finding', 'transactional_action'."
    )


class LlmContextInferencePlugin(ContextInferencePlugin):
    """Uses an LLM to infer context properties from raw context data."""

    plugin_id: str = "llm_context_inference_v1"
    description: str = "Infers structured context properties using an LLM."

    _model_id: Optional[str] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self._model_id = cfg.get("model")  # Allow specifying a model override
        logger.info(
            f"[{self.plugin_id}] Initialized. Model override: {self._model_id or 'None'}"
        )

    async def infer_context_properties(
        self, raw_context: Dict[str, Any], genie: "Genie"
    ) -> Dict[str, Any]:
        history = raw_context.get("history", [])
        profile = raw_context.get("profile", {})

        history_summary = "\n".join(
            f"{msg.get('role', 'unknown')}: {str(msg.get('content', ''))[:150]}"
            for msg in history[-5:]
        )

        # FIX: Added a more forceful and explicit instruction to prevent conversational text.
        prompt = f"""
        Analyze the following conversation history and user profile to infer contextual properties.
        Your response MUST be a single, valid JSON object that conforms to the required schema.
        DO NOT add any conversational text, comments, or explanations before or after the JSON object.

        User Profile:
        {profile}

        Recent Conversation History:
        ---
        {history_summary}
        ---

        Based on the data above, infer the audience expertise, emotional state (as 'audience_state'), primary topic (as 'discourse_topic'), and intent.
        """

        try:
            llm_response = await genie.llm.chat(
                [{"role": "user", "content": prompt}],
                provider_id=self._model_id,
                output_schema=InferredContextModel,
            )
            parsed_output = await genie.llm.parse_output(
                llm_response,
                parser_id="pydantic_output_parser_v1",
                schema=InferredContextModel,
            )
            if isinstance(parsed_output, InferredContextModel):
                # Structure the output to match the paper's Ctx_Inf = (T, P) model
                return {
                    "DiscourseTopic": {"primary": parsed_output.discourse_topic},
                    "AudienceProfile": {
                        "expertise": parsed_output.audience_expertise,
                        "state": parsed_output.audience_state,
                        "intent": parsed_output.intent,
                    },
                }
            logger.error(
                f"[{self.plugin_id}] LLM output parsing failed to return the expected model type."
            )
            return {}
        except Exception as e:
            logger.error(
                f"[{self.plugin_id}] Failed to infer context via LLM: {e}",
                exc_info=True,
            )
            return {}
