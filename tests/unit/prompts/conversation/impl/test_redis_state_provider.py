import json
import logging
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from genie_tooling.prompts.conversation.impl.redis_state_provider import (
    REDIS_AVAILABLE,
    RedisError,
    RedisStateProviderPlugin,
)
from genie_tooling.prompts.conversation.types import ConversationState

PROVIDER_LOGGER_NAME = "genie_tooling.prompts.conversation.impl.redis_state_provider"

# Mock Redis client if the library isn't installed
if not REDIS_AVAILABLE:
    aioredis = MagicMock()
    aioredis.from_url = MagicMock()
else:
    from redis import asyncio as aioredis


@pytest.fixture
def mock_redis_client() -> AsyncMock:
    """Provides a mock for the aioredis.Redis client instance."""
    client = AsyncMock()
    client.ping = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=0)
    client.close = AsyncMock()
    return client


@pytest.fixture
async def redis_state_provider(mock_redis_client: AsyncMock) -> RedisStateProviderPlugin:
    """Provides an initialized RedisStateProviderPlugin with a mocked client."""
    provider = RedisStateProviderPlugin()
    with patch(
        "genie_tooling.prompts.conversation.impl.redis_state_provider.aioredis.from_url",
        return_value=mock_redis_client,
    ):
        await provider.setup(config={"redis_url": "redis://mock-server:6379"})
    # Reset mocks after setup to ensure clean state for each test
    mock_redis_client.get.reset_mock(return_value=None)
    mock_redis_client.set.reset_mock(return_value=True)
    mock_redis_client.delete.reset_mock(return_value=0)
    return provider


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
class TestRedisStateProvider:
    async def test_setup_success(self, mock_redis_client: AsyncMock):
        """Test successful setup and connection to Redis."""
        provider = RedisStateProviderPlugin()
        with patch(
            "genie_tooling.prompts.conversation.impl.redis_state_provider.aioredis.from_url",
            return_value=mock_redis_client,
        ):
            await provider.setup(
                config={
                    "redis_url": "redis://test-host:1234",
                    "key_prefix": "test_prefix:",
                    "default_ttl_seconds": 3600,
                }
            )
        assert provider._redis_client is mock_redis_client
        assert provider._key_prefix == "test_prefix:"
        assert provider._default_ttl_seconds == 3600
        mock_redis_client.ping.assert_awaited_once()

    async def test_setup_connection_error(self, caplog: pytest.LogCaptureFixture):
        """Test setup failure when Redis connection fails."""
        caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
        mock_failing_client = AsyncMock()
        mock_failing_client.ping.side_effect = RedisError("Connection refused")
        with patch(
            "genie_tooling.prompts.conversation.impl.redis_state_provider.aioredis.from_url",
            return_value=mock_failing_client,
        ):
            provider = RedisStateProviderPlugin()
            await provider.setup(config={"redis_url": "redis://bad-host"})
        assert provider._redis_client is None
        assert "Failed to connect to Redis: Connection refused" in caplog.text

    async def test_load_state_success(self, redis_state_provider: RedisStateProviderPlugin):
        """Test loading an existing state successfully."""
        provider = await redis_state_provider
        session_id = "session1"
        state_data: ConversationState = {
            "session_id": session_id,
            "history": [{"role": "user", "content": "test"}],
            "metadata": {"user": "test_user"},
        }
        provider._redis_client.get.return_value = json.dumps(state_data).encode("utf-8") # type: ignore

        loaded_state = await provider.load_state(session_id)

        assert loaded_state is not None
        assert loaded_state["session_id"] == session_id
        assert loaded_state["history"][0]["content"] == "test"
        provider._redis_client.get.assert_awaited_once_with(f"genie_cs:{session_id}") # type: ignore

    async def test_load_state_not_found(
        self, redis_state_provider: RedisStateProviderPlugin
    ):
        """Test loading a non-existent state."""
        provider = await redis_state_provider
        provider._redis_client.get.return_value = None # type: ignore
        loaded_state = await provider.load_state("non_existent_session")
        assert loaded_state is None

    async def test_load_state_json_decode_error(
        self, redis_state_provider: RedisStateProviderPlugin, caplog: pytest.LogCaptureFixture
    ):
        """Test handling of malformed JSON data in Redis."""
        caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
        provider = await redis_state_provider
        provider._redis_client.get.return_value = b"{'invalid': json}" # type: ignore
        loaded_state = await provider.load_state("corrupt_session")
        assert loaded_state is None
        assert "Failed to JSON decode state from Redis" in caplog.text

    async def test_save_state_success(self, redis_state_provider: RedisStateProviderPlugin):
        """Test saving a state successfully."""
        provider = await redis_state_provider
        state_data: ConversationState = {
            "session_id": "session_to_save",
            "history": [],
            "metadata": {},
        }
        await provider.save_state(state_data)
        provider._redis_client.set.assert_awaited_once_with( # type: ignore
            "genie_cs:session_to_save",
            json.dumps(state_data).encode("utf-8"),
            ex=None,
        )

    async def test_save_state_with_ttl(self, redis_state_provider: RedisStateProviderPlugin):
        """Test saving a state with a TTL."""
        provider = await redis_state_provider
        provider._default_ttl_seconds = 60
        state_data: ConversationState = {"session_id": "session_ttl", "history": []}
        await provider.save_state(state_data)
        provider._redis_client.set.assert_awaited_once_with( # type: ignore
            "genie_cs:session_ttl",
            json.dumps(state_data).encode("utf-8"),
            ex=60,
        )

    async def test_delete_state_success(
        self, redis_state_provider: RedisStateProviderPlugin
    ):
        """Test deleting an existing state."""
        provider = await redis_state_provider
        provider._redis_client.delete.return_value = 1 # type: ignore
        result = await provider.delete_state("session_to_delete")
        assert result is True
        provider._redis_client.delete.assert_awaited_once_with("genie_cs:session_to_delete") # type: ignore

    async def test_delete_state_not_found(
        self, redis_state_provider: RedisStateProviderPlugin
    ):
        """Test deleting a non-existent state."""
        provider = await redis_state_provider
        provider._redis_client.delete.return_value = 0 # type: ignore
        result = await provider.delete_state("non_existent_session")
        assert result is False

    async def test_teardown(self, redis_state_provider: RedisStateProviderPlugin):
        """Test that teardown closes the client connection."""
        provider = await redis_state_provider
        client_mock = provider._redis_client
        await provider.teardown()
        client_mock.close.assert_awaited_once() # type: ignore
        assert provider._redis_client is None