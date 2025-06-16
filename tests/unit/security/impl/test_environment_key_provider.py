### tests/unit/security/impl/test_environment_key_provider.py
import logging
from unittest.mock import patch

import pytest
from genie_tooling.security.impl.environment_key_provider import (
    EnvironmentKeyProvider,
)

PROVIDER_LOGGER_NAME = "genie_tooling.security.impl.environment_key_provider"


@pytest.fixture()
async def env_key_provider() -> EnvironmentKeyProvider:
    """Provides a fresh instance of EnvironmentKeyProvider for each test."""
    provider = EnvironmentKeyProvider()
    await provider.setup()
    return provider


@pytest.mark.asyncio()
async def test_setup_logs_initialization(caplog: pytest.LogCaptureFixture):
    """Verify that the setup method logs an informational message."""
    caplog.set_level(logging.INFO, logger=PROVIDER_LOGGER_NAME)
    provider = EnvironmentKeyProvider()
    await provider.setup()
    assert f"{provider.plugin_id}: Initialized. Will read keys from environment variables." in caplog.text


@pytest.mark.asyncio()
async def test_get_key_success(env_key_provider: EnvironmentKeyProvider):
    """Test retrieving an existing environment variable."""
    provider = await env_key_provider
    key_name = "TEST_API_KEY_EXISTS"
    key_value = "secret-value-123"

    with patch.dict("os.environ", {key_name: key_value}):
        retrieved_key = await provider.get_key(key_name)

    assert retrieved_key == key_value


@pytest.mark.asyncio()
async def test_get_key_not_found(env_key_provider: EnvironmentKeyProvider):
    """Test retrieving a non-existent environment variable."""
    provider = await env_key_provider
    key_name = "TEST_API_KEY_DOES_NOT_EXIST"

    # Ensure the key is not in the environment for this test
    with patch.dict("os.environ", {}, clear=True):
        retrieved_key = await provider.get_key(key_name)

    assert retrieved_key is None


@pytest.mark.asyncio()
async def test_get_key_logs_debug_messages(
    env_key_provider: EnvironmentKeyProvider, caplog: pytest.LogCaptureFixture
):
    """Test that debug messages are logged correctly for key presence and absence."""
    provider = await env_key_provider
    caplog.set_level(logging.DEBUG, logger=PROVIDER_LOGGER_NAME)

    # Test case: Key exists
    with patch.dict("os.environ", {"EXISTING_KEY": "value"}):
        await provider.get_key("EXISTING_KEY")
    assert f"{provider.plugin_id}: Retrieved key 'EXISTING_KEY' from environment (exists)." in caplog.text
    caplog.clear()

    # Test case: Key does not exist
    with patch.dict("os.environ", {}, clear=True):
        await provider.get_key("MISSING_KEY")
    assert f"{provider.plugin_id}: Key 'MISSING_KEY' not found in environment variables." in caplog.text


@pytest.mark.asyncio()
async def test_teardown_is_noop(env_key_provider: EnvironmentKeyProvider, caplog: pytest.LogCaptureFixture):
    """Verify that teardown completes and logs a debug message."""
    provider = await env_key_provider
    caplog.set_level(logging.DEBUG, logger=PROVIDER_LOGGER_NAME)
    await provider.teardown()
    assert f"{provider.plugin_id}: Teardown complete." in caplog.text
