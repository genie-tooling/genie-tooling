from unittest.mock import AsyncMock, MagicMock

import pytest

# Correctly import from the new files
from genie_tooling.context.plugins.derivation.generic_agent_derivation import (
    GenericAgentDerivationPlugin,
)
from genie_tooling.context.plugins.formulation.llm_prompt import (
    LlmPromptFormulationPlugin,
)
from genie_tooling.context.plugins.predicate_extractors.heuristic_extractor import (
    HeuristicPredicateExtractorPlugin,
)


@pytest.mark.asyncio()
async def test_heuristic_predicate_extractor():
    extractor = HeuristicPredicateExtractorPlugin()
    await extractor.setup()
    assert await extractor.extract("what is the weather?", None) == "predicate_what"
    assert await extractor.extract("calculate 2+2", None) == "predicate_calculate"
    assert (
        await extractor.extract("a generic statement", None)
        == "predicate_generic_inquiry"
    )


@pytest.mark.asyncio()
async def test_generic_agentic_derivation_plugin():
    """
    Tests the GenericAgentDerivationPlugin.
    """
    # ARRANGE
    mock_genie = MagicMock()
    mock_genie.run_command = AsyncMock(
        return_value={"status": "success", "output": "agent completed task"}
    )

    plugin = GenericAgentDerivationPlugin()
    await plugin.setup()

    constraints = {"command_processor_id": "deep_research_agent_v1"}

    # ACT
    result = await plugin.derive(
        query="research the latest AI trends",
        constraints=constraints,
        genie=mock_genie,
    )

    # ASSERT
    mock_genie.run_command.assert_called_once_with(
        command="research the latest AI trends",
        processor_id="deep_research_agent_v1",
        context_for_tools={"cqs_constraints": constraints}
    )

    assert result["status"] == "success"
    assert result["result"]["output"] == "agent completed task"


@pytest.mark.asyncio()
async def test_llm_prompt_formulation_plugin():
    mock_genie = MagicMock()

    mock_genie.prompts.render_prompt = AsyncMock(return_value="Mocked Rendered Prompt")

    mock_genie.llm.chat = AsyncMock(
        return_value={"message": {"content": "Here is your friendly response."}}
    )

    plugin = LlmPromptFormulationPlugin()
    await plugin.setup()

    raw_data = {"result": "42"}
    constraints = {"tone": "friendly", "prompt_template_id": "test_template"}

    result = await plugin.formulate(
        "what is the meaning of life?", raw_data, constraints, mock_genie
    )

    mock_genie.prompts.render_prompt.assert_called_once()
    mock_genie.llm.chat.assert_called_once()

    # The plugin now prepends a constraint-instruction block (A1) before the
    # rendered template so C_F constraints actually reach the LLM. The
    # rendered template must still appear in the final content.
    final_llm_call_args = mock_genie.llm.chat.call_args[0][0]
    content = final_llm_call_args[0]["content"]
    assert "Mocked Rendered Prompt" in content
    assert "Response guidelines:" in content
    assert "friendly" in content  # constraint flowed through

    assert result == "Here is your friendly response."
