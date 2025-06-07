import logging
from unittest.mock import patch

import pytest
from genie_tooling.llm_providers.impl.llama_cpp_internal_provider import (
    LlamaCppInternalLLMProviderPlugin,
)

PROVIDER_LOGGER_NAME = "genie_tooling.llm_providers.impl.llama_cpp_internal_provider"

@pytest.mark.asyncio
async def test_setup_model_path_missing(caplog):
    caplog.set_level(logging.INFO, logger=PROVIDER_LOGGER_NAME)
    provider = LlamaCppInternalLLMProviderPlugin()
    await provider.setup(config={})
    assert provider._model_client is None
    assert "'model_path' not provided in configuration. Plugin will be disabled" in caplog.text
