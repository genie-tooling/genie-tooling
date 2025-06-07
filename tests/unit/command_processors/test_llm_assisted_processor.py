import logging

import pytest
from genie_tooling.command_processors.impl.llm_assisted_processor import (
    LLMAssistedToolSelectionProcessorPlugin,
)

PROCESSOR_LOGGER_NAME = "genie_tooling.command_processors.impl.llm_assisted_processor"

@pytest.fixture
def llm_assisted_processor() -> LLMAssistedToolSelectionProcessorPlugin:
    return LLMAssistedToolSelectionProcessorPlugin()

@pytest.mark.asyncio
async def test_setup_no_genie_facade(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger=PROCESSOR_LOGGER_NAME)
    await llm_assisted_processor.setup(config={})
    assert llm_assisted_processor._genie is None
    assert "Genie facade not found in config" in caplog.text
