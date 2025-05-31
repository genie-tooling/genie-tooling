### tests/unit/command_processors/test_command_processor_manager.py
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.manager import CommandProcessorManager
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.security.key_provider import KeyProvider


# --- Mocks ---
class MockCommandProcessor(CommandProcessorPlugin):
    plugin_id: str = "mock_cmd_proc_v1"
    description: str = "Mock Command Processor for Manager Tests"
    setup_config_received: Optional[Dict[str, Any]] = None
    key_provider_received: Optional[KeyProvider] = None
    genie_facade_received: Optional[Any] = None
    teardown_called: bool = False
    setup_should_fail: bool = False
    teardown_should_fail: bool = False

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        self.setup_config_received = config
        if config:
            self.key_provider_received = config.get("key_provider")
            self.genie_facade_received = config.get("genie_facade")
        if self.setup_should_fail:
            raise RuntimeError("Simulated setup failure in MockCommandProcessor")
        self.teardown_called = False

    async def process_command(
        self, command: str, conversation_history: Optional[List[ChatMessage]] = None
    ) -> CommandProcessorResponse:
        return {"chosen_tool_id": "mock_tool", "extracted_params": {"param": command}}

    async def teardown(self) -> None:
        self.teardown_called = True
        if self.teardown_should_fail:
            raise RuntimeError("Simulated teardown failure in MockCommandProcessor")

class NotACommandProcessor(Plugin): # Does not implement CommandProcessorPlugin
    plugin_id: str = "not_a_cmd_proc_v1"
    description: str = "Not a command processor"
    async def setup(self, config: Optional[Dict[str, Any]]) -> None: pass
    async def teardown(self) -> None: pass


@pytest.fixture
def mock_plugin_manager_for_cmd_proc_mgr(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.list_discovered_plugin_classes = mocker.MagicMock(return_value={})
    pm.get_plugin_instance = AsyncMock() # Will be configured per test
    return pm

@pytest.fixture
async def mock_key_provider_for_cmd_proc_mgr(mock_key_provider: KeyProvider) -> KeyProvider: # Uses conftest mock_key_provider
    return await mock_key_provider # Correctly awaits the async fixture

@pytest.fixture
def mock_middleware_config_for_cmd_proc_mgr() -> MiddlewareConfig:
    return MiddlewareConfig(
        command_processor_configurations={
            MockCommandProcessor.plugin_id: {"default_setting": "global_value"}
        }
    )

@pytest.fixture
def mock_genie_facade_for_cmd_proc_mgr(mocker) -> MagicMock:
    # A minimal mock for Genie, just enough for the manager
    genie = MagicMock(name="MockGenieFacadeForCmdProcMgr")
    # Add attributes that might be accessed by the processor's setup or methods
    genie._tool_manager = AsyncMock(name="MockToolManagerOnGenie")
    genie._config = MiddlewareConfig() # Give it a default config
    return genie

@pytest.fixture
def cmd_proc_manager(
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_key_provider_for_cmd_proc_mgr: KeyProvider,
    mock_middleware_config_for_cmd_proc_mgr: MiddlewareConfig
) -> CommandProcessorManager:
    return CommandProcessorManager(
        plugin_manager=mock_plugin_manager_for_cmd_proc_mgr,
        key_provider=mock_key_provider_for_cmd_proc_mgr,
        config=mock_middleware_config_for_cmd_proc_mgr
    )

# --- Tests ---

@pytest.mark.asyncio
async def test_get_command_processor_success_new_instance(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_key_provider_for_cmd_proc_mgr: KeyProvider,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock
):
    processor_id = MockCommandProcessor.plugin_id
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: MockCommandProcessor
    }

    instance = await cmd_proc_manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)

    assert instance is not None
    assert isinstance(instance, MockCommandProcessor)
    assert instance.plugin_id == processor_id
    assert instance.setup_config_received is not None
    assert instance.setup_config_received.get("default_setting") == "global_value"
    assert instance.key_provider_received is mock_key_provider_for_cmd_proc_mgr
    assert instance.genie_facade_received is mock_genie_facade_for_cmd_proc_mgr

@pytest.mark.asyncio
async def test_get_command_processor_cached_instance(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock
):
    processor_id = MockCommandProcessor.plugin_id
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: MockCommandProcessor
    }

    instance1 = await cmd_proc_manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance1 is not None
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.reset_mock() # Ensure it's not called again

    instance2 = await cmd_proc_manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance2 is instance1
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.assert_not_called()

@pytest.mark.asyncio
async def test_get_command_processor_config_override(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock
):
    processor_id = MockCommandProcessor.plugin_id
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: MockCommandProcessor
    }
    override_config = {"local_setting": "override_value", "default_setting": "local_override"}
    instance = await cmd_proc_manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr, config_override=override_config)

    assert instance is not None
    assert isinstance(instance, MockCommandProcessor)
    assert instance.setup_config_received is not None
    assert instance.setup_config_received.get("default_setting") == "local_override" # Overridden
    assert instance.setup_config_received.get("local_setting") == "override_value" # Added

@pytest.mark.asyncio
async def test_get_command_processor_not_found_in_plugin_manager(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    processor_id = "non_existent_proc_id"
    # Ensure list_discovered_plugin_classes returns an empty dict or one without the ID
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {}

    instance = await cmd_proc_manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance is None
    assert f"CommandProcessorPlugin class for ID '{processor_id}' not found in PluginManager." in caplog.text

@pytest.mark.asyncio
async def test_get_command_processor_instantiated_not_correct_type(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    processor_id = NotACommandProcessor.plugin_id
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: NotACommandProcessor
    }
    instance = await cmd_proc_manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance is None
    assert f"Instantiated plugin '{processor_id}' is not a valid CommandProcessorPlugin." in caplog.text

@pytest.mark.asyncio
async def test_get_command_processor_setup_fails(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    processor_id = MockCommandProcessor.plugin_id
    # Configure the mock processor to fail setup
    class FailingSetupMockProcessor(MockCommandProcessor):
        async def setup(self, config: Optional[Dict[str, Any]]) -> None:
            await super().setup(config)
            raise RuntimeError("Simulated setup failure in test")

    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: FailingSetupMockProcessor
    }
    instance = await cmd_proc_manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance is None
    assert f"Error instantiating or setting up CommandProcessorPlugin '{processor_id}': Simulated setup failure in test" in caplog.text

@pytest.mark.asyncio
async def test_command_processor_manager_teardown(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock
):
    processor_id = MockCommandProcessor.plugin_id
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: MockCommandProcessor
    }
    instance = await cmd_proc_manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance is not None
    assert processor_id in cmd_proc_manager._instantiated_processors

    await cmd_proc_manager.teardown()
    assert not cmd_proc_manager._instantiated_processors
    assert instance.teardown_called is True

@pytest.mark.asyncio
async def test_command_processor_manager_teardown_processor_fails(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    processor_id = MockCommandProcessor.plugin_id

    class FailingTeardownMockProcessor(MockCommandProcessor):
        async def teardown(self) -> None:
            await super().teardown() # Sets teardown_called = True
            raise RuntimeError("Simulated teardown failure in test")

    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: FailingTeardownMockProcessor
    }
    instance = await cmd_proc_manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance is not None

    await cmd_proc_manager.teardown()
    assert not cmd_proc_manager._instantiated_processors # Should still be cleared
    assert instance.teardown_called is True # Teardown was attempted
    assert f"Error tearing down CommandProcessorPlugin '{processor_id}': Simulated teardown failure in test" in caplog.text

@pytest.mark.asyncio
async def test_get_command_processor_global_config_not_middleware_config(
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_key_provider_for_cmd_proc_mgr: KeyProvider,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    # Create manager with a non-MiddlewareConfig global_config
    manager_bad_config = CommandProcessorManager(
        plugin_manager=mock_plugin_manager_for_cmd_proc_mgr,
        key_provider=mock_key_provider_for_cmd_proc_mgr,
        config=MagicMock() # Not a MiddlewareConfig instance
    )
    processor_id = "any_proc_id"
    instance = await manager_bad_config.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance is None
    assert "CommandProcessorManager: self._global_config is not a MiddlewareConfig instance." in caplog.text
