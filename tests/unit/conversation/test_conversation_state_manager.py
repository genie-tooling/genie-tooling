from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.conversation.impl.abc import (
    ConversationStateProviderPlugin,
)
from genie_tooling.conversation.impl.manager import (
    ConversationStateManager,
)
from genie_tooling.conversation.types import ConversationState
from genie_tooling.core.plugin_manager import PluginManager


@pytest.fixture()
def mock_plugin_manager_for_convo() -> MagicMock:
    pm = MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture()
def mock_convo_state_provider() -> MagicMock:
    provider = AsyncMock(spec=ConversationStateProviderPlugin)
    provider.load_state = AsyncMock(return_value=None)
    provider.save_state = AsyncMock()
    provider.delete_state = AsyncMock(return_value=False)
    return provider

@pytest.fixture()
def convo_state_manager(
    mock_plugin_manager_for_convo: MagicMock,
    mock_convo_state_provider: MagicMock
) -> ConversationStateManager:
    mock_plugin_manager_for_convo.get_plugin_instance.return_value = mock_convo_state_provider
    return ConversationStateManager(
        plugin_manager=mock_plugin_manager_for_convo,
        default_provider_id="default_convo_provider"
    )

@pytest.mark.asyncio()
async def test_load_state_success(convo_state_manager: ConversationStateManager, mock_convo_state_provider: MagicMock):
    session_id = "s1"
    expected_state: ConversationState = {"session_id": session_id, "history": []}
    mock_convo_state_provider.load_state.return_value = expected_state

    state = await convo_state_manager.load_state(session_id)
    assert state == expected_state
    mock_convo_state_provider.load_state.assert_awaited_once_with(session_id)

@pytest.mark.asyncio()
async def test_load_state_provider_not_found(convo_state_manager: ConversationStateManager, mock_plugin_manager_for_convo: MagicMock):
    mock_plugin_manager_for_convo.get_plugin_instance.return_value = None # Simulate provider not found
    state = await convo_state_manager.load_state("s1", provider_id="bad_provider")
    assert state is None

@pytest.mark.asyncio()
async def test_save_state_success(convo_state_manager: ConversationStateManager, mock_convo_state_provider: MagicMock):
    state_to_save: ConversationState = {"session_id": "s2", "history": [{"role":"user", "content":"Test"}]}
    await convo_state_manager.save_state(state_to_save)
    mock_convo_state_provider.save_state.assert_awaited_once_with(state_to_save)

@pytest.mark.asyncio()
async def test_add_message_new_session(convo_state_manager: ConversationStateManager, mock_convo_state_provider: MagicMock):
    session_id = "s_new"
    message = {"role": "user", "content": "First message"}
    mock_convo_state_provider.load_state.return_value = None # Simulate new session

    await convo_state_manager.add_message(session_id, message)

    mock_convo_state_provider.load_state.assert_awaited_once_with(session_id)
    # Check that save_state was called with the new state
    args, _kwargs = mock_convo_state_provider.save_state.call_args
    saved_state: ConversationState = args[0]
    assert saved_state["session_id"] == session_id
    assert len(saved_state["history"]) == 1
    assert saved_state["history"][0] == message
    assert "last_updated" in saved_state["metadata"] # type: ignore

@pytest.mark.asyncio()
async def test_add_message_existing_session(convo_state_manager: ConversationStateManager, mock_convo_state_provider: MagicMock):
    session_id = "s_existing"
    existing_history = [{"role": "user", "content": "Old message"}]
    existing_state: ConversationState = {"session_id": session_id, "history": existing_history, "metadata": {"user_id": "u1"}}
    mock_convo_state_provider.load_state.return_value = existing_state

    new_message = {"role": "assistant", "content": "New reply"}
    await convo_state_manager.add_message(session_id, new_message)

    args, _kwargs = mock_convo_state_provider.save_state.call_args
    saved_state: ConversationState = args[0]
    assert len(saved_state["history"]) == 2
    assert saved_state["history"][1] == new_message
    assert saved_state["metadata"]["user_id"] == "u1" # type: ignore
    assert "last_updated" in saved_state["metadata"] # type: ignore

@pytest.mark.asyncio()
async def test_delete_state_success(convo_state_manager: ConversationStateManager, mock_convo_state_provider: MagicMock):
    mock_convo_state_provider.delete_state.return_value = True
    result = await convo_state_manager.delete_state("s_del")
    assert result is True
    mock_convo_state_provider.delete_state.assert_awaited_once_with("s_del")

@pytest.mark.asyncio()
async def test_teardown(convo_state_manager: ConversationStateManager):
    # Teardown is currently a no-op for ConversationStateManager itself
    await convo_state_manager.teardown()
    assert True # Just ensure it runs without error
