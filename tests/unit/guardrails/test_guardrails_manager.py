### tests/unit/guardrails/test_guardrail_manager.py
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin
from genie_tooling.guardrails.abc import (
    InputGuardrailPlugin,
    OutputGuardrailPlugin,
    ToolUsageGuardrailPlugin,
)
from genie_tooling.guardrails.manager import GuardrailManager
from genie_tooling.guardrails.types import GuardrailAction, GuardrailViolation
from genie_tooling.tools.abc import Tool as ToolPlugin

MANAGER_LOGGER_NAME = "genie_tooling.guardrails.manager"


# --- Mocks ---
class MockInputGuardrail(InputGuardrailPlugin):
    description: str = "Mock Input Guardrail"
    default_action: GuardrailAction = "allow"
    check_input_should_return: GuardrailViolation = GuardrailViolation(action="allow", reason="Mock input allow")
    check_input_should_raise: bool = False
    teardown_called: bool = False
    _plugin_id_storage: str

    def __init__(self, plugin_id_val: str = "mock_input_gr_v1"):
        self._plugin_id_storage = plugin_id_val
        # Make check_input and teardown AsyncMocks on the instance for easier side_effect/assertion
        self.check_input = AsyncMock(return_value=self.check_input_should_return)
        self.teardown = AsyncMock()

    @property
    def plugin_id(self) -> str:
        return self._plugin_id_storage

    async def setup(self, config=None):
        self.teardown_called = False
        # Reset mock states if setup is called multiple times in tests
        self.check_input.reset_mock(return_value=self.check_input_should_return)
        self.teardown.reset_mock()
        if self.check_input_should_raise: # Re-apply side_effect if needed
            self.check_input.side_effect = RuntimeError("Input check failed")


class MockOutputGuardrail(OutputGuardrailPlugin):
    description: str = "Mock Output Guardrail"
    default_action: GuardrailAction = "allow"
    check_output_should_return: GuardrailViolation = GuardrailViolation(action="allow", reason="Mock output allow")
    check_output_should_raise: bool = False
    teardown_called: bool = False
    _plugin_id_storage: str

    def __init__(self, plugin_id_val: str = "mock_output_gr_v1"):
        self._plugin_id_storage = plugin_id_val
        self.check_output = AsyncMock(return_value=self.check_output_should_return)
        self.teardown = AsyncMock()

    @property
    def plugin_id(self) -> str:
        return self._plugin_id_storage

    async def setup(self, config=None):
        self.teardown_called = False
        self.check_output.reset_mock(return_value=self.check_output_should_return)
        self.teardown.reset_mock()
        if self.check_output_should_raise:
            self.check_output.side_effect = RuntimeError("Output check failed")


class MockToolUsageGuardrail(ToolUsageGuardrailPlugin):
    description: str = "Mock Tool Usage Guardrail"
    default_action: GuardrailAction = "allow"
    check_tool_usage_should_return: GuardrailViolation = GuardrailViolation(action="allow", reason="Mock tool usage allow")
    check_tool_usage_should_raise: bool = False
    teardown_called: bool = False
    _plugin_id_storage: str

    def __init__(self, plugin_id_val: str = "mock_tool_usage_gr_v1"):
        self._plugin_id_storage = plugin_id_val
        self.check_tool_usage = AsyncMock(return_value=self.check_tool_usage_should_return)
        self.teardown = AsyncMock()

    @property
    def plugin_id(self) -> str:
        return self._plugin_id_storage

    async def setup(self, config=None):
        self.teardown_called = False
        self.check_tool_usage.reset_mock(return_value=self.check_tool_usage_should_return)
        self.teardown.reset_mock()
        if self.check_tool_usage_should_raise:
            self.check_tool_usage.side_effect = RuntimeError("Tool usage check failed")


class NotAGuardrail(Plugin):
    plugin_id: str = "not_a_guardrail_v1"
    description: str = "Not a guardrail"
    async def setup(self, config=None): pass
    async def teardown(self): pass

@pytest.fixture
def mock_tool_for_guardrail_test() -> MagicMock:
    tool = MagicMock(spec=ToolPlugin)
    tool.identifier = "test_tool_for_guardrail"
    return tool

@pytest.fixture
def mock_plugin_manager_for_gr_mgr() -> MagicMock:
    pm = MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture
def guardrail_manager(
    mock_plugin_manager_for_gr_mgr: MagicMock
) -> GuardrailManager:
    return GuardrailManager(
        plugin_manager=mock_plugin_manager_for_gr_mgr,
        default_input_guardrail_ids=[],
        default_output_guardrail_ids=[],
        default_tool_usage_guardrail_ids=[],
        guardrail_configurations={}
    )

# --- Tests ---

@pytest.mark.asyncio
async def test_initialize_guardrails_success(
    guardrail_manager: GuardrailManager,
    mock_plugin_manager_for_gr_mgr: MagicMock,
):
    mock_input_gr = MockInputGuardrail(plugin_id_val="input_gr1")
    mock_output_gr = MockOutputGuardrail(plugin_id_val="output_gr1")
    mock_tool_usage_gr = MockToolUsageGuardrail(plugin_id_val="tool_usage_gr1")

    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == "input_gr1": return mock_input_gr
        if plugin_id == "output_gr1": return mock_output_gr
        if plugin_id == "tool_usage_gr1": return mock_tool_usage_gr
        return None
    mock_plugin_manager_for_gr_mgr.get_plugin_instance.side_effect = get_instance_side_effect

    guardrail_manager._input_guardrail_ids = ["input_gr1"]
    guardrail_manager._output_guardrail_ids = ["output_gr1"]
    guardrail_manager._tool_usage_guardrail_ids = ["tool_usage_gr1"]
    guardrail_manager._guardrail_configurations = {
        "input_gr1": {"cfg_in": True},
        "output_gr1": {"cfg_out": True},
        "tool_usage_gr1": {"cfg_tool": True},
    }

    await guardrail_manager._initialize_guardrails()

    assert len(guardrail_manager._active_input_guardrails) == 1
    assert guardrail_manager._active_input_guardrails[0] is mock_input_gr
    assert len(guardrail_manager._active_output_guardrails) == 1
    assert guardrail_manager._active_output_guardrails[0] is mock_output_gr
    assert len(guardrail_manager._active_tool_usage_guardrails) == 1
    assert guardrail_manager._active_tool_usage_guardrails[0] is mock_tool_usage_gr

    mock_plugin_manager_for_gr_mgr.get_plugin_instance.assert_any_call("input_gr1", config={"cfg_in": True})
    mock_plugin_manager_for_gr_mgr.get_plugin_instance.assert_any_call("output_gr1", config={"cfg_out": True})
    mock_plugin_manager_for_gr_mgr.get_plugin_instance.assert_any_call("tool_usage_gr1", config={"cfg_tool": True})
    assert guardrail_manager._initialized is True

    mock_plugin_manager_for_gr_mgr.get_plugin_instance.reset_mock()
    await guardrail_manager._initialize_guardrails()
    mock_plugin_manager_for_gr_mgr.get_plugin_instance.assert_not_called()


@pytest.mark.asyncio
async def test_initialize_guardrails_plugin_not_found(
    mock_plugin_manager_for_gr_mgr: MagicMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=MANAGER_LOGGER_NAME)
    mock_plugin_manager_for_gr_mgr.get_plugin_instance.return_value = None
    manager = GuardrailManager(
        plugin_manager=mock_plugin_manager_for_gr_mgr,
        default_input_guardrail_ids=["non_existent_gr"]
    )
    await manager._initialize_guardrails()
    assert len(manager._active_input_guardrails) == 0
    assert "InputGuardrailPlugin 'non_existent_gr' not found or failed to load." in caplog.text


@pytest.mark.asyncio
async def test_initialize_guardrails_plugin_wrong_type(
    mock_plugin_manager_for_gr_mgr: MagicMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=MANAGER_LOGGER_NAME)
    wrong_type_plugin = NotAGuardrail()
    mock_plugin_manager_for_gr_mgr.get_plugin_instance.return_value = wrong_type_plugin
    manager = GuardrailManager(
        plugin_manager=mock_plugin_manager_for_gr_mgr,
        default_output_guardrail_ids=[wrong_type_plugin.plugin_id]
    )
    await manager._initialize_guardrails()
    assert len(manager._active_output_guardrails) == 0
    assert f"Plugin '{wrong_type_plugin.plugin_id}' loaded but is not a valid OutputGuardrailPlugin." in caplog.text


@pytest.mark.asyncio
async def test_initialize_guardrails_load_error(
    mock_plugin_manager_for_gr_mgr: MagicMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    mock_plugin_manager_for_gr_mgr.get_plugin_instance.side_effect = RuntimeError("Load failed")
    manager = GuardrailManager(
        plugin_manager=mock_plugin_manager_for_gr_mgr,
        default_tool_usage_guardrail_ids=["error_gr"]
    )
    await manager._initialize_guardrails()
    assert len(manager._active_tool_usage_guardrails) == 0
    assert "Error loading ToolUsageGuardrailPlugin 'error_gr': Load failed" in caplog.text


@pytest.mark.asyncio
async def test_check_input_guardrails_all_allow(guardrail_manager: GuardrailManager):
    mock_gr1 = MockInputGuardrail(plugin_id_val="in_gr1")
    mock_gr2 = MockInputGuardrail(plugin_id_val="in_gr2")
    guardrail_manager._active_input_guardrails = [mock_gr1, mock_gr2]
    guardrail_manager._initialized = True

    result = await guardrail_manager.check_input_guardrails("test data")
    assert result["action"] == "allow"
    mock_gr1.check_input.assert_awaited_once()
    mock_gr2.check_input.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_input_guardrails_one_blocks(guardrail_manager: GuardrailManager):
    mock_gr1 = MockInputGuardrail(plugin_id_val="in_gr1_allow")
    mock_gr2_block = MockInputGuardrail(plugin_id_val="in_gr2_block")
    mock_gr2_block.check_input.return_value = GuardrailViolation(action="block", reason="Blocked by gr2") # Set return value on the AsyncMock
    mock_gr3 = MockInputGuardrail(plugin_id_val="in_gr3_allow_after_block")

    guardrail_manager._active_input_guardrails = [mock_gr1, mock_gr2_block, mock_gr3]
    guardrail_manager._initialized = True

    result = await guardrail_manager.check_input_guardrails("test data")
    assert result["action"] == "block"
    assert result["reason"] == "Blocked by gr2"
    mock_gr1.check_input.assert_awaited_once()
    mock_gr2_block.check_input.assert_awaited_once()
    mock_gr3.check_input.assert_not_called() # check_input is an AsyncMock, use .called


@pytest.mark.asyncio
async def test_check_output_guardrails_warn(guardrail_manager: GuardrailManager):
    mock_gr_warn = MockOutputGuardrail(plugin_id_val="out_gr_warn")
    mock_gr_warn.check_output.return_value = GuardrailViolation(action="warn", reason="Output warning")
    guardrail_manager._active_output_guardrails = [mock_gr_warn]
    guardrail_manager._initialized = True

    result = await guardrail_manager.check_output_guardrails("output data")
    assert result["action"] == "warn"
    assert result["reason"] == "Output warning"


@pytest.mark.asyncio
async def test_check_tool_usage_guardrails_pass(
    guardrail_manager: GuardrailManager, mock_tool_for_guardrail_test: MagicMock
):
    mock_gr = MockToolUsageGuardrail(plugin_id_val="tool_gr_pass")
    guardrail_manager._active_tool_usage_guardrails = [mock_gr]
    guardrail_manager._initialized = True

    result = await guardrail_manager.check_tool_usage_guardrails(mock_tool_for_guardrail_test, {"param": "val"})
    assert result["action"] == "allow"
    mock_gr.check_tool_usage.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_no_active_guardrails_of_type(guardrail_manager: GuardrailManager):
    guardrail_manager._initialized = True
    result = await guardrail_manager.check_input_guardrails("some data")
    assert result["action"] == "allow"
    assert result["reason"] == "All input guardrails passed."


@pytest.mark.asyncio
async def test_teardown_calls_guardrail_teardown(
    guardrail_manager: GuardrailManager,
    mock_plugin_manager_for_gr_mgr: MagicMock
):
    mock_input_gr = MockInputGuardrail(plugin_id_val="in_gr_td_unique")
    mock_output_gr = MockOutputGuardrail(plugin_id_val="out_gr_td_unique")
    mock_tool_gr = MockToolUsageGuardrail(plugin_id_val="tool_gr_td_unique")

    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == "in_gr_td_unique": return mock_input_gr
        if plugin_id == "out_gr_td_unique": return mock_output_gr
        if plugin_id == "tool_gr_td_unique": return mock_tool_gr
        return None
    mock_plugin_manager_for_gr_mgr.get_plugin_instance.side_effect = get_instance_side_effect

    guardrail_manager._input_guardrail_ids = ["in_gr_td_unique"]
    guardrail_manager._output_guardrail_ids = ["out_gr_td_unique"]
    guardrail_manager._tool_usage_guardrail_ids = ["tool_gr_td_unique"]
    await guardrail_manager._initialize_guardrails()

    assert mock_input_gr in guardrail_manager._active_input_guardrails
    assert mock_output_gr in guardrail_manager._active_output_guardrails
    assert mock_tool_gr in guardrail_manager._active_tool_usage_guardrails

    await guardrail_manager.teardown()

    mock_input_gr.teardown.assert_awaited_once()
    mock_output_gr.teardown.assert_awaited_once()
    mock_tool_gr.teardown.assert_awaited_once()

    assert len(guardrail_manager._active_input_guardrails) == 0
    assert len(guardrail_manager._active_output_guardrails) == 0
    assert len(guardrail_manager._active_tool_usage_guardrails) == 0
    assert guardrail_manager._initialized is False


@pytest.mark.asyncio
async def test_teardown_guardrail_teardown_error(
    guardrail_manager: GuardrailManager,
    mock_plugin_manager_for_gr_mgr: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    mock_failing_gr = MockInputGuardrail(plugin_id_val="failing_gr_td")
    # Set side_effect on the AsyncMock instance of teardown
    mock_failing_gr.teardown.side_effect = RuntimeError("Teardown error")


    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == "failing_gr_td": return mock_failing_gr
        return None
    mock_plugin_manager_for_gr_mgr.get_plugin_instance.side_effect = get_instance_side_effect
    guardrail_manager._input_guardrail_ids = ["failing_gr_td"]
    await guardrail_manager._initialize_guardrails()

    await guardrail_manager.teardown()
    assert f"Error tearing down guardrail '{mock_failing_gr.plugin_id}': Teardown error" in caplog.text
    assert len(guardrail_manager._active_input_guardrails) == 0
