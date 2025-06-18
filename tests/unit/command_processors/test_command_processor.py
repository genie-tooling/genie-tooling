### tests/unit/command_processors/test_command_processor.py
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
    setup_called_with_config: Optional[Dict[str, Any]] = None
    key_provider_received: Optional[KeyProvider] = None
    genie_facade_received: Optional[Any] = None
    teardown_called: bool = False
    setup_should_fail: bool = False
    teardown_should_fail: bool = False

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        self.setup_called_with_config = config
        if config:
            self.key_provider_received = config.get("key_provider")
            self.genie_facade_received = config.get("genie_facade")
        if self.setup_should_fail:
            raise RuntimeError("Simulated setup failure in MockCommandProcessor")
        self.teardown_called = False

    async def process_command(
        self, command: str, conversation_history: Optional[List[ChatMessage]] = None, correlation_id: Optional[str] = None
    ) -> CommandProcessorResponse:
        return {"chosen_tool_id": "mock_tool", "extracted_params": {"param": command}}

    async def teardown(self) -> None:
        self.teardown_called = True
        if self.teardown_should_fail:
            raise RuntimeError("Simulated teardown failure in MockCommandProcessor")

class NotACommandProcessor(Plugin): # Does not implement CommandProcessorPlugin
    plugin_id: str = "not_a_cmd_proc_v1"
    description: str = "Not a command processor"
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass


@pytest.fixture()
def mock_plugin_manager_for_cmd_proc_mgr(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.list_discovered_plugin_classes = mocker.MagicMock(return_value={})
    pm.get_plugin_instance = AsyncMock() # Will be configured per test
    return pm

@pytest.fixture()
async def mock_key_provider_for_cmd_proc_mgr(mock_key_provider: KeyProvider) -> KeyProvider: # Uses conftest mock_key_provider
    return await mock_key_provider # Correctly awaits the async fixture

@pytest.fixture()
def mock_middleware_config_for_cmd_proc_mgr() -> MiddlewareConfig:
    return MiddlewareConfig(
        command_processor_configurations={
            MockCommandProcessor.plugin_id: {"default_setting": "global_value"}
        }
    )

@pytest.fixture()
def mock_genie_facade_for_cmd_proc_mgr(mocker) -> MagicMock:
    genie = MagicMock(name="MockGenieFacadeForCmdProcMgr")
    genie._tool_manager = AsyncMock(name="MockToolManagerOnGenie")
    genie._config = MiddlewareConfig()
    return genie

@pytest.fixture()
async def cmd_proc_manager(
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_key_provider_for_cmd_proc_mgr: KeyProvider,
    mock_middleware_config_for_cmd_proc_mgr: MiddlewareConfig
) -> CommandProcessorManager:
    actual_key_provider = await mock_key_provider_for_cmd_proc_mgr
    return CommandProcessorManager(
        plugin_manager=mock_plugin_manager_for_cmd_proc_mgr,
        key_provider=actual_key_provider,
        config=mock_middleware_config_for_cmd_proc_mgr
    )

# --- Tests ---

@pytest.mark.asyncio()
async def test_get_command_processor_success_new_instance(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock,
):
    manager = await cmd_proc_manager
    processor_id = MockCommandProcessor.plugin_id
    mock_proc_instance = MockCommandProcessor()
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: MockCommandProcessor
    }
    # Configure the mock PM to instantiate and setup the plugin
    async def get_instance_side_effect(pid, config):
        if pid == processor_id:
            await mock_proc_instance.setup(config)
            return mock_proc_instance
        return None
    mock_plugin_manager_for_cmd_proc_mgr.get_plugin_instance.side_effect = get_instance_side_effect

    instance = await manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)

    assert instance is not None
    assert isinstance(instance, MockCommandProcessor)
    assert instance.plugin_id == processor_id
    assert instance.setup_called_with_config is not None
    assert instance.setup_called_with_config.get("default_setting") == "global_value"
    assert instance.key_provider_received is manager._key_provider
    assert instance.genie_facade_received is mock_genie_facade_for_cmd_proc_mgr

@pytest.mark.asyncio()
async def test_get_command_processor_cached_instance(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock
):
    manager = await cmd_proc_manager
    processor_id = MockCommandProcessor.plugin_id
    mock_proc_instance = MockCommandProcessor()
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: MockCommandProcessor
    }
    async def get_instance_side_effect(pid, config, **kwargs):
        await mock_proc_instance.setup(config)
        return mock_proc_instance
    mock_plugin_manager_for_cmd_proc_mgr.get_plugin_instance.side_effect = get_instance_side_effect

    instance1 = await manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance1 is not None

    instance2 = await manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance2 is instance1
    mock_plugin_manager_for_cmd_proc_mgr.get_plugin_instance.assert_called_once()

@pytest.mark.asyncio()
async def test_get_command_processor_config_override(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock
):
    manager = await cmd_proc_manager
    processor_id = MockCommandProcessor.plugin_id
    mock_proc_instance = MockCommandProcessor()
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: MockCommandProcessor
    }
    async def get_instance_side_effect(pid, config, **kwargs):
        await mock_proc_instance.setup(config)
        return mock_proc_instance
    mock_plugin_manager_for_cmd_proc_mgr.get_plugin_instance.side_effect = get_instance_side_effect

    override_config = {"local_setting": "override_value", "default_setting": "local_override"}
    instance = await manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr, config_override=override_config)

    assert instance is not None
    assert isinstance(instance, MockCommandProcessor)
    assert instance.setup_called_with_config is not None
    assert instance.setup_called_with_config.get("default_setting") == "local_override"
    assert instance.setup_called_with_config.get("local_setting") == "override_value"

@pytest.mark.asyncio()
async def test_get_command_processor_not_found_in_plugin_manager(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    manager = await cmd_proc_manager
    processor_id = "non_existent_proc_id"
    # The class is not discovered, so list_discovered_plugin_classes returns an empty dict.
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {}

    instance = await manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)

    # ASSERT: The manager should return None immediately.
    assert instance is None
    # ASSERT: `get_plugin_instance` should NOT have been called.
    mock_plugin_manager_for_cmd_proc_mgr.get_plugin_instance.assert_not_called()
    # ASSERT: The correct log message should be present.
    assert f"CommandProcessorPlugin class for ID '{processor_id}' not found in PluginManager." in caplog.text

@pytest.mark.asyncio()
async def test_get_command_processor_instantiated_not_correct_type(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    manager = await cmd_proc_manager
    processor_id = NotACommandProcessor.plugin_id
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: NotACommandProcessor
    }
    # Configure PM mock to return the wrong type instance
    mock_plugin_manager_for_cmd_proc_mgr.get_plugin_instance.return_value = NotACommandProcessor()
    instance = await manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance is None
    assert f"Instantiated plugin '{processor_id}' is not a valid CommandProcessorPlugin." in caplog.text

@pytest.mark.asyncio()
async def test_get_command_processor_setup_fails(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    manager = await cmd_proc_manager
    processor_id = "fail_setup_proc"
    class FailingSetupMockProcessor(MockCommandProcessor):
        async def setup(self, config: Optional[Dict[str, Any]]) -> None:
            await super().setup(config)
            raise RuntimeError("Simulated setup failure in test")

    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: FailingSetupMockProcessor
    }
    # The PM's get_plugin_instance will now raise the error, which is what we want to test.
    mock_plugin_manager_for_cmd_proc_mgr.get_plugin_instance.side_effect = RuntimeError("Simulated setup failure in test")

    instance = await manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance is None
    assert f"Error instantiating or setting up CommandProcessorPlugin '{processor_id}': Simulated setup failure in test" in caplog.text

@pytest.mark.asyncio()
async def test_command_processor_manager_teardown(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock
):
    manager = await cmd_proc_manager
    processor_id = MockCommandProcessor.plugin_id
    mock_proc_instance = MockCommandProcessor()
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: MockCommandProcessor
    }
    async def get_instance_side_effect_teardown(pid, config):
        if pid == processor_id:
            await mock_proc_instance.setup(config)
            return mock_proc_instance
        return None
    mock_plugin_manager_for_cmd_proc_mgr.get_plugin_instance.side_effect = get_instance_side_effect_teardown

    instance = await manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance is not None
    assert processor_id in manager._instantiated_processors

    await manager.teardown()
    assert not manager._instantiated_processors
    assert instance.teardown_called is True

@pytest.mark.asyncio()
async def test_command_processor_manager_teardown_processor_fails(
    cmd_proc_manager: CommandProcessorManager,
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger="genie_tooling.command_processors.manager")
    manager = await cmd_proc_manager
    processor_id = "failing_teardown_proc"

    class FailingTeardownMockProcessor(MockCommandProcessor):
        async def teardown(self) -> None:
            self.teardown_called = True
            raise RuntimeError("Simulated teardown failure in test")

    mock_failing_instance = FailingTeardownMockProcessor()
    mock_plugin_manager_for_cmd_proc_mgr.list_discovered_plugin_classes.return_value = {
        processor_id: FailingTeardownMockProcessor
    }
    async def get_instance_side_effect_teardown_fail(pid, config, **kwargs):
        if pid == processor_id:
            await mock_failing_instance.setup(config)
            return mock_failing_instance
        return None
    mock_plugin_manager_for_cmd_proc_mgr.get_plugin_instance.side_effect = get_instance_side_effect_teardown_fail

    instance = await manager.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance is not None

    await manager.teardown()
    assert not manager._instantiated_processors
    assert instance.teardown_called is True
    assert f"Error tearing down CommandProcessorPlugin '{processor_id}': Simulated teardown failure in test" in caplog.text

@pytest.mark.asyncio()
async def test_get_command_processor_global_config_not_middleware_config(
    mock_plugin_manager_for_cmd_proc_mgr: PluginManager,
    mock_key_provider_for_cmd_proc_mgr: KeyProvider,
    mock_genie_facade_for_cmd_proc_mgr: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    actual_key_provider = await mock_key_provider_for_cmd_proc_mgr
    manager_bad_config = CommandProcessorManager(
        plugin_manager=mock_plugin_manager_for_cmd_proc_mgr,
        key_provider=actual_key_provider,
        config=MagicMock() # Not a MiddlewareConfig instance
    )
    processor_id = "any_proc_id"
    instance = await manager_bad_config.get_command_processor(processor_id, mock_genie_facade_for_cmd_proc_mgr)
    assert instance is None
    assert "CommandProcessorManager: self._global_config is not a MiddlewareConfig instance." in caplog.text
