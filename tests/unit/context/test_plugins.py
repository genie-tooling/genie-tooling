import pytest
from unittest.mock import AsyncMock, MagicMock

# --- Imports for Plugins Under Test ---
from genie_tooling.context.plugins.derivation.generic_tool_derivation import GenericToolDerivationPlugin
from genie_tooling.context.plugins.inference.llm_inference import LlmContextInferencePlugin, InferredContextModel
from genie_tooling.context.plugins.predicate_extractors.llm_predicate_extractor import LLMPredicateExtractorPlugin, PredicateModel


# ==============================================================================
# == Tests for GenericToolDerivationPlugin (No Changes Needed Here)
# ==============================================================================

@pytest.mark.asyncio
async def test_tool_derivation_success():
    """Tests the success path where a tool is correctly specified and executed."""
    mock_genie = MagicMock()
    mock_genie.execute_tool = AsyncMock(return_value={"data": "tool success"})
    plugin = GenericToolDerivationPlugin()
    await plugin.setup()
    constraints = {"tool_id": "fact_lookup_tool", "params": {"entity": "earth"}}
    
    result = await plugin.derive("what is earth's mass", constraints, mock_genie)
    
    mock_genie.execute_tool.assert_called_once_with("fact_lookup_tool", entity="earth")
    assert result == {"status": "success", "result": {"data": "tool success"}}

# ... other tests for this plugin remain the same ...
@pytest.mark.asyncio
async def test_tool_derivation_fails_on_missing_tool_id():
    mock_genie = MagicMock()
    mock_genie.execute_tool = AsyncMock()
    plugin = GenericToolDerivationPlugin()
    await plugin.setup()
    constraints = {"params": {"entity": "earth"}}
    result = await plugin.derive("what is earth's mass", constraints, mock_genie)
    mock_genie.execute_tool.assert_not_called()
    assert result["status"] == "error"
    assert "No 'tool_id' specified" in result["error"]

@pytest.mark.asyncio
async def test_tool_derivation_fails_on_bad_params_type():
    mock_genie = MagicMock()
    mock_genie.execute_tool = AsyncMock()
    plugin = GenericToolDerivationPlugin()
    await plugin.setup()
    constraints = {"tool_id": "some_tool", "params": "not_a_dict"}
    result = await plugin.derive("a query", constraints, mock_genie)
    mock_genie.execute_tool.assert_not_called()
    assert result["status"] == "error"
    assert "'params' for tool 'some_tool' must be a dictionary" in result["error"]

@pytest.mark.asyncio
async def test_tool_derivation_handles_execution_exception():
    mock_genie = MagicMock()
    mock_genie.execute_tool = AsyncMock(side_effect=ValueError("Tool execution failed"))
    plugin = GenericToolDerivationPlugin()
    await plugin.setup()
    constraints = {"tool_id": "failing_tool", "params": {}}
    result = await plugin.derive("a query", constraints, mock_genie)
    mock_genie.execute_tool.assert_called_once()
    assert result["status"] == "error"
    assert "Failed to derive result via tool execution" in result["error"]

# ==============================================================================
# == Tests for LLMPredicateExtractorPlugin (Corrected)
# ==============================================================================

@pytest.mark.asyncio
async def test_llm_predicate_extractor_success():
    """Tests successful predicate extraction via a mocked LLM call."""
    # ARRANGE
    mock_genie = MagicMock()
    
    # --- THIS IS THE FIX ---
    # Mock the return value of the chat call itself.
    mock_genie.llm.chat = AsyncMock(return_value={"message": {"content": "doesn't matter"}})
    
    # The parse_output mock remains correct.
    mock_genie.llm.parse_output = AsyncMock(
        return_value=PredicateModel(predicate="predicate_what")
    )
    
    plugin = LLMPredicateExtractorPlugin()
    await plugin.setup()
    
    # ACT
    predicate = await plugin.extract("what is the capital of france", mock_genie)
    
    # ASSERT
    mock_genie.llm.chat.assert_called_once()
    mock_genie.llm.parse_output.assert_called_once() # This will now pass
    assert predicate == "predicate_what"

@pytest.mark.asyncio
async def test_llm_predicate_extractor_llm_failure():
    # This test was already correct as it correctly mocks the side_effect.
    mock_genie = MagicMock()
    mock_genie.llm.chat = AsyncMock(side_effect=Exception("API unavailable"))
    plugin = LLMPredicateExtractorPlugin()
    await plugin.setup()
    
    predicate = await plugin.extract("some query", mock_genie)
    
    assert predicate == "predicate_generic_inquiry"

@pytest.mark.asyncio
async def test_llm_predicate_extractor_parsing_failure():
    # This test was also correct, as it mocks the return value of the second call.
    mock_genie = MagicMock()
    mock_genie.llm.chat = AsyncMock(return_value={"message": {"content": "valid response"}})
    mock_genie.llm.parse_output = AsyncMock(return_value=None)
    plugin = LLMPredicateExtractorPlugin()
    await plugin.setup()
    
    predicate = await plugin.extract("some query", mock_genie)
    
    mock_genie.llm.chat.assert_called_once()
    assert predicate == "predicate_generic_inquiry"


# ==============================================================================
# == Tests for LlmContextInferencePlugin (Corrected)
# ==============================================================================

@pytest.mark.asyncio
async def test_llm_inference_success():
    """Tests the success path for context inference."""
    # ARRANGE
    mock_genie = MagicMock()
    
    # --- THIS IS THE FIX ---
    # Mock the return value of the chat call itself.
    mock_genie.llm.chat = AsyncMock(return_value={"message": {"content": "doesn't matter"}})
    
    # The parse_output mock remains correct.
    mock_parsed_output = InferredContextModel(
        audience_expertise="expert",
        audience_state="curious",
        discourse_topic="astrophysics",
        intent="fact_finding"
    )
    mock_genie.llm.parse_output = AsyncMock(return_value=mock_parsed_output)
    
    plugin = LlmContextInferencePlugin()
    await plugin.setup()
    
    raw_context = {
        "history": [{"role": "user", "content": "Tell me about black holes."}],
        "profile": {"name": "Dr. Reed"}
    }
    
    # ACT
    inferred_context = await plugin.infer_context_properties(raw_context, mock_genie)
    
    # ASSERT
    mock_genie.llm.chat.assert_called_once() # This will now pass
    assert inferred_context == {
        "DiscourseTopic": {"primary": "astrophysics"},
        "AudienceProfile": {
            "expertise": "expert",
            "state": "curious",
            "intent": "fact_finding",
        },
    }

@pytest.mark.asyncio
async def test_llm_inference_llm_fails():
    # This test was already correct.
    mock_genie = MagicMock()
    mock_genie.llm.chat = AsyncMock(side_effect=Exception("LLM is down"))
    plugin = LlmContextInferencePlugin()
    await plugin.setup()
    
    inferred_context = await plugin.infer_context_properties({}, mock_genie)
    
    assert inferred_context == {}

@pytest.mark.asyncio
async def test_llm_inference_parsing_fails():
    # This test was already correct.
    mock_genie = MagicMock()
    mock_genie.llm.chat = AsyncMock(return_value={"message": {"content": "valid response"}})
    mock_genie.llm.parse_output = AsyncMock(return_value=None)
    plugin = LlmContextInferencePlugin()
    await plugin.setup()
    
    inferred_context = await plugin.infer_context_properties({}, mock_genie)
    
    mock_genie.llm.chat.assert_called_once()
    assert inferred_context == {}