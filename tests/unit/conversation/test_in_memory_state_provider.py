### tests/unit/prompts/conversation/impl/test_in_memory_state_provider.py
import asyncio
import logging
from typing import Any

import pytest
from genie_tooling.conversation.impl.in_memory_state_provider import (
    InMemoryStateProviderPlugin,
)
from genie_tooling.conversation.types import ConversationState

PROVIDER_LOGGER_NAME = "genie_tooling.prompts.conversation.impl.in_memory_state_provider"

@pytest.fixture()
async def mem_state_provider() -> InMemoryStateProviderPlugin:
    provider = InMemoryStateProviderPlugin()
    await provider.setup()
    return provider

@pytest.mark.asyncio()
async def test_save_and_load_state(mem_state_provider: InMemoryStateProviderPlugin):
    provider = await mem_state_provider
    session_id = "session_123"
    state_to_save: ConversationState = {
        "session_id": session_id,
        "history": [{"role": "user", "content": "Hello"}],
        "metadata": {"user_id": "user_abc"}
    }
    await provider.save_state(state_to_save)
    loaded_state = await provider.load_state(session_id)
    assert loaded_state is not None
    assert loaded_state["session_id"] == session_id
    assert loaded_state["history"] == state_to_save["history"]
    assert loaded_state["metadata"] == state_to_save["metadata"]
    # Ensure it's a copy
    assert loaded_state is not state_to_save
    assert loaded_state["history"] is not state_to_save["history"]

@pytest.mark.asyncio()
async def test_load_non_existent_state(mem_state_provider: InMemoryStateProviderPlugin):
    provider = await mem_state_provider
    loaded_state = await provider.load_state("non_existent_session")
    assert loaded_state is None

@pytest.mark.asyncio()
async def test_save_invalid_state(mem_state_provider: InMemoryStateProviderPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await mem_state_provider
    invalid_state: Any = {"history": []} # Missing session_id
    await provider.save_state(invalid_state)
    assert "Attempted to save invalid state (missing session_id)." in caplog.text

@pytest.mark.asyncio()
async def test_delete_state(mem_state_provider: InMemoryStateProviderPlugin):
    provider = await mem_state_provider
    session_id = "session_to_delete"
    state: ConversationState = {"session_id": session_id, "history": []}
    await provider.save_state(state)
    assert await provider.load_state(session_id) is not None

    delete_result_true = await provider.delete_state(session_id)
    assert delete_result_true is True
    assert await provider.load_state(session_id) is None

    delete_result_false = await provider.delete_state(session_id) # Delete again
    assert delete_result_false is False

@pytest.mark.asyncio()
async def test_teardown_clears_store(mem_state_provider: InMemoryStateProviderPlugin):
    provider = await mem_state_provider
    await provider.save_state({"session_id": "s1", "history": []})
    assert len(provider._store) == 1
    await provider.teardown()
    assert len(provider._store) == 0

@pytest.mark.asyncio()
async def test_concurrent_access(mem_state_provider: InMemoryStateProviderPlugin):
    provider = await mem_state_provider
    session_id = "concurrent_session"
    initial_state: ConversationState = {"session_id": session_id, "history": [], "metadata": {"count": 0}}
    await provider.save_state(initial_state)

    num_tasks = 5
    async def increment_metadata_count(p: InMemoryStateProviderPlugin, sid: str):
        # Simulate read-modify-write cycle
        s = await p.load_state(sid)
        if s and s.get("metadata"):
            current_count = s["metadata"].get("count", 0)
            s["metadata"]["count"] = current_count + 1
            await asyncio.sleep(0.001) # Introduce a small delay to increase chance of race if no lock
            await p.save_state(s)

    tasks = [increment_metadata_count(provider, session_id) for _ in range(num_tasks)]
    await asyncio.gather(*tasks)

    final_state = await provider.load_state(session_id)
    assert final_state is not None
    # Due to the read-modify-write race condition in the test logic itself (not the provider's locks),
    # the final count will likely be 1, as later saves overwrite earlier ones.
    # The provider's individual load/save operations are atomic due to its internal lock.
    assert final_state.get("metadata", {}).get("count") == 1
