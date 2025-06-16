###tests/unit/command_processors/impl/test_deep_research_processor.py###
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.agents.types import AgentOutput
from genie_tooling.command_processors.impl.deep_research_processor import (
    DeepResearchProcessorPlugin,
)


@pytest.fixture
def mock_genie_for_processor():
    """Provides a mock Genie facade instance."""
    genie = MagicMock(name="MockGenieFacade")
    genie.observability = AsyncMock()
    genie.observability.trace_event = AsyncMock()
    return genie


@pytest.fixture
def mock_deep_research_agent_class():
    """Mocks the DeepResearchAgent class to control its instances."""
    with patch(
        "genie_tooling.command_processors.impl.deep_research_processor.DeepResearchAgent"
    ) as mock:
        yield mock


@pytest.mark.asyncio
class TestDeepResearchProcessorPlugin:
    async def test_setup_success(self, mock_genie_for_processor):
        """Test successful setup with a valid configuration."""
        processor = DeepResearchProcessorPlugin()
        config = {"genie_facade": mock_genie_for_processor, "agent_config": {"key": "val"}}
        await processor.setup(config)
        assert processor._genie is mock_genie_for_processor
        assert processor._agent_config == {"key": "val"}

    async def test_setup_missing_genie_facade_raises_error(self):
        """Test that setup raises a ValueError if the genie_facade is missing."""
        processor = DeepResearchProcessorPlugin()
        with pytest.raises(
            ValueError, match="requires a 'genie_facade' instance in its config"
        ):
            await processor.setup({})

    async def test_process_command_success(
        self,
        mock_genie_for_processor,
        mock_deep_research_agent_class: MagicMock,
    ):
        """Test a successful command processing run."""
        processor = DeepResearchProcessorPlugin()
        await processor.setup({"genie_facade": mock_genie_for_processor})

        mock_agent_instance = mock_deep_research_agent_class.return_value
        agent_output = AgentOutput(status="success", output="Final report content.", history=[])
        mock_agent_instance.run = AsyncMock(return_value=agent_output)

        command = "Research quantum computing applications."
        result = await processor.process_command(command)

        mock_deep_research_agent_class.assert_called_once_with(
            genie=mock_genie_for_processor, agent_config={}
        )
        mock_agent_instance.run.assert_awaited_once_with(goal=command)
        assert result["final_answer"] == "Final report content."
        assert "raw_response" in result

    async def test_process_command_agent_run_fails(
        self,
        mock_genie_for_processor,
        mock_deep_research_agent_class: MagicMock,
    ):
        """Test handling of exceptions from the agent's run method."""
        processor = DeepResearchProcessorPlugin()
        await processor.setup({"genie_facade": mock_genie_for_processor})

        mock_agent_instance = mock_deep_research_agent_class.return_value
        mock_agent_instance.run.side_effect = Exception("Agent internal error")

        command = "This command will fail."
        result = await processor.process_command(command)

        assert "error" in result
        assert "Failed to execute deep research: Agent internal error" in result["error"]