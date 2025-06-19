# tests/unit/command_processors/test_deep_research_processor.py
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.command_processors.impl.deep_research_processor import (
    DeepResearchProcessorPlugin,
)

PROCESSOR_LOGGER_NAME = "genie_tooling.command_processors.impl.deep_research_processor"


@pytest.fixture()
async def deep_research_processor() -> DeepResearchProcessorPlugin:
    """Provides a default instance of the processor."""
    return DeepResearchProcessorPlugin()


@pytest.mark.asyncio()
async def test_deep_research_setup_with_genie_facade(
    deep_research_processor: DeepResearchProcessorPlugin,
    caplog: pytest.LogCaptureFixture,
):
    """
    Tests that setup completes successfully when a valid genie_facade is provided.
    """
    processor = await deep_research_processor
    mock_genie_facade = MagicMock()
    config = {
        "genie_facade": mock_genie_facade,
        "agent_config": {"min_high_quality_sources": 5},
    }
    caplog.set_level(logging.INFO, logger=PROCESSOR_LOGGER_NAME)

    # Act
    await processor.setup(config=config)

    # Assert
    assert processor._genie is mock_genie_facade
    # This assertion now passes because the processor's setup correctly extracts the nested dict.
    assert processor._agent_config["min_high_quality_sources"] == 5
    assert "Initialized. Will delegate commands to DeepResearchAgent" in caplog.text


@pytest.mark.asyncio()
async def test_deep_research_process_command_fails_without_facade(
    deep_research_processor: DeepResearchProcessorPlugin,
):
    """
    Tests that process_command returns an error if setup was called without a facade.
    """
    processor = await deep_research_processor
    # Simulate setup without the facade
    await processor.setup(config=None)

    # Act
    response = await processor.process_command(command="research topic")

    # Assert
    assert "error" in response
    assert "not properly initialized" in response["error"]


@pytest.mark.asyncio()
async def test_deep_research_process_command_delegates_to_agent(
    mocker, deep_research_processor: DeepResearchProcessorPlugin
):
    """
    Tests that process_command correctly instantiates and runs the DeepResearchAgent.
    """
    # Arrange
    mock_genie = MagicMock()
    mock_genie.observability.trace_event = AsyncMock()

    # Mock the DeepResearchAgent's run method to return a predictable result
    mock_agent_run = AsyncMock(
        return_value={"status": "success", "output": "Final research report."}
    )
    # Patch the DeepResearchAgent class within the processor's module
    mocker.patch(
        "genie_tooling.command_processors.impl.deep_research_processor.DeepResearchAgent.run",
        mock_agent_run,
    )

    processor = await deep_research_processor
    await processor.setup(config={"genie_facade": mock_genie})

    # Act
    response = await processor.process_command(command="What is quantum computing?")

    # Assert
    # 1. The agent's run method was called once with the correct goal.
    mock_agent_run.assert_awaited_once_with(goal="What is quantum computing?")

    # 2. The processor's response contains the agent's final answer.
    assert response["final_answer"] == "Final research report."
    assert "error" not in response
