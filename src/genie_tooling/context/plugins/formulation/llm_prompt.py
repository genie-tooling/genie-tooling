import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from genie_tooling.context.constraints import formulation_constraints_to_instructions
from genie_tooling.context.protocols import FormulationStrategyPlugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


# The bundled cqs prompt templates (`direct_fact_formulation.prompt`,
# `summarize_agent_output.prompt`, and any user-authored equivalents) use
# Jinja syntax — `{{ raw_data.result.value }}` style with dotted attribute
# access. Default the template engine here to the Jinja2 plugin so bundled
# templates render correctly out of the box. Users who prefer str.format
# style can override via `formulation_strategy_config.template_engine_id`.
DEFAULT_TEMPLATE_ENGINE_ID = "jinja2_chat_template_v1"


class LlmPromptFormulationPlugin(FormulationStrategyPlugin):
    """
    A generic formulation strategy that uses a configurable prompt template
    from the Genie PromptManager to formulate a final response.
    """

    plugin_id: str = "llm_prompt_formulation_v1"
    description: str = "Formulates a final response using a flexible, configurable LLM prompt template."

    _template_engine_id: Optional[str] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        engine_id = cfg.get("template_engine_id")
        self._template_engine_id = engine_id if engine_id else DEFAULT_TEMPLATE_ENGINE_ID
        logger.info(
            f"[{self.plugin_id}] Initialized. Template engine: '{self._template_engine_id}'."
        )

    async def formulate(
        self, query: str, raw_data: Any, constraints: Dict[str, Any], genie: "Genie"
    ) -> str:

        # The rule's action should now specify the template to use.
        template_id = constraints.get("prompt_template_id", "default_formulation_prompt")

        # The prompt data is now an aggregation of all available context.
        prompt_data = {
            "original_query": query,
            "raw_data": raw_data,
            "formulation_constraints": constraints,
        }

        logger.info(
            f"[{self.plugin_id}] Formulating response using template '{template_id}' "
            f"via engine '{self._template_engine_id}'."
        )

        try:
            # Use the core genie.prompts interface to render the final prompt
            response_str = await genie.prompts.render_prompt(
                name=template_id,
                data=prompt_data,
                template_engine_id=self._template_engine_id,
            )
            if not response_str:
                raise ValueError(f"Prompt rendering for '{template_id}' returned empty content.")

            # Translate C_F constraints into explicit instructions. Without
            # this, every tone/verbosity/empathy/redaction directive set by
            # a rule would be load-bearing nothing — the bundled templates
            # don't read formulation_constraints themselves. This prepend
            # is the policy interface from rule → LLM.
            instructions = formulation_constraints_to_instructions(constraints)
            if instructions:
                final_prompt = f"{instructions}\n\n{response_str}"
            else:
                final_prompt = response_str

            # Pass the fully rendered prompt to the LLM
            response = await genie.llm.chat([{"role": "user", "content": final_prompt}])
            return response["message"]["content"] or "I'm sorry, I couldn't formulate a response."

        except Exception as e:
            logger.error(f"[{self.plugin_id}] LLM call for formulation failed: {e}", exc_info=True)
            return "I apologize, but I encountered an error while trying to formulate my final answer."
