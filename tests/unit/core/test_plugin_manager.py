"""Unit tests for the PluginManager."""
import abc  # For testing abstract class detection
import importlib.metadata
import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from typing import Protocol as TypingProtocol
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin


class DummyPluginAlpha(Plugin):
    plugin_id: str = "dummy_alpha_v1"
    description: str = "Alpha"
    some_value: int = 1
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

class DummyPluginBeta(Plugin):
    plugin_id: str = "dummy_beta_v1"
    description: str = "A beta plugin"
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

class NotAPlugin:
    pass

class SetupFailsPlugin(Plugin):
    plugin_id: str = "setup_fails_v1"
    description: str = "Fails setup"
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        raise RuntimeError("Setup deliberately failed")
    async def teardown(self) -> None: pass

class TeardownFailsPlugin(Plugin):
    plugin_id: str = "teardown_fails_v1"
    description: str = "Fails teardown"
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None:
        raise RuntimeError("Teardown deliberately failed")

class AbstractPluginBase(Plugin, abc.ABC): # Abstract Plugin
    plugin_id: str = "abstract_base_v1" # Won't be discovered
    description: str = "Abstract base"
    @abc.abstractmethod
    async def an_abstract_method(self) -> None:
        pass
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass


@pytest.fixture
async def fresh_plugin_manager() -> PluginManager:
    pm = PluginManager(plugin_dev_dirs=[])
    return pm

@pytest.mark.asyncio
async def test_plugin_manager_initialization(fresh_plugin_manager: PluginManager):
    pm = await fresh_plugin_manager
    assert pm is not None
    assert isinstance(pm, PluginManager)
    assert pm.list_discovered_plugin_classes() is not None
    assert pm.plugin_dev_dirs == []

@pytest.mark.asyncio
async def test_plugin_manager_discover_from_empty_dev_dirs(fresh_plugin_manager: PluginManager):
    pm = await fresh_plugin_manager
    await pm.discover_plugins()
    assert len(pm.list_discovered_plugin_classes()) == 0

    pm_none_dirs = PluginManager(plugin_dev_dirs=None)
    await pm_none_dirs.discover_plugins()
    assert len(pm_none_dirs.list_discovered_plugin_classes()) == 0

@pytest.mark.asyncio
async def test_plugin_manager_discover_from_non_existent_dev_dir(fresh_plugin_manager: PluginManager, caplog: pytest.LogCaptureFixture):
    await fresh_plugin_manager # This pm has pm.plugin_dev_dirs = []

    # Create a NEW PluginManager instance for this specific test case
    # so its __init__ processes the non_existent_dir_str correctly for its internal list.
    non_existent_dir_str = "./this_dir_does_not_exist_hopefully"
    pm_test_specific = PluginManager(plugin_dev_dirs=[non_existent_dir_str])

    caplog.set_level(logging.WARNING)
    # Call discover_plugins on the test-specific instance
    await pm_test_specific.discover_plugins()

    assert len(pm_test_specific.list_discovered_plugin_classes()) == 0

    # The PluginManager's __init__ converts dev_dirs to Path objects and resolves them.
    # The log message inside discover_plugins iterates over self.plugin_dev_dirs
    # which are already resolved Path objects.
    resolved_path_logged = Path(non_existent_dir_str).resolve()
    expected_log_message = f"Plugin dev dir '{resolved_path_logged}' not found. Skipping."
    assert expected_log_message in caplog.text


@pytest.fixture
def plugin_manager_for_sync_test() -> PluginManager:
    return PluginManager()

def test_is_valid_plugin_class_check(plugin_manager_for_sync_test: PluginManager):
    pm_sync = plugin_manager_for_sync_test
    assert pm_sync._is_valid_plugin_class(DummyPluginAlpha) is True
    assert pm_sync._is_valid_plugin_class(DummyPluginBeta) is True
    assert pm_sync._is_valid_plugin_class(NotAPlugin) is False
    assert pm_sync._is_valid_plugin_class(Plugin) is False
    assert pm_sync._is_valid_plugin_class(object()) is False
    assert pm_sync._is_valid_plugin_class(int) is False
    assert pm_sync._is_valid_plugin_class(AbstractPluginBase) is False

@patch("pathlib.Path.rglob")
@pytest.mark.asyncio
async def test_plugin_manager_discover_from_dev_dir(mock_rglob: MagicMock, tmp_path: Path, fresh_plugin_manager: PluginManager):
    pm = await fresh_plugin_manager
    dev_plugin_dir = tmp_path / "test_plugins_deterministic_order"
    dev_plugin_dir.mkdir()

    plugin_alpha_content = """
from genie_tooling.core.types import Plugin
from typing import Dict, Any, Optional
class DiscoveredPluginAlpha(Plugin):
    plugin_id: str = "discovered_alpha_v1"
    description: str = "Discovered Alpha"
    async def setup(self, config: Optional[Dict[str, Any]] = None):pass
    async def teardown(self):pass
"""
    alpha_file = dev_plugin_dir / "01_alpha_plugin.py"
    alpha_file.write_text(plugin_alpha_content)

    plugin_beta_content_duplicate_id = """
from genie_tooling.core.types import Plugin
from typing import Dict, Any, Optional
class DiscoveredPluginBetaDuplicate(Plugin):
    plugin_id: str = "discovered_alpha_v1"
    description: str = "Discovered Beta with duplicate ID"
    async def setup(self, config: Optional[Dict[str, Any]] = None):pass
    async def teardown(self):pass

class UniquePluginInBetaFile(Plugin):
    plugin_id: str = "unique_beta_v1"
    description: str = "Unique Beta"
    async def setup(self, config: Optional[Dict[str, Any]] = None):pass
    async def teardown(self):pass
"""
    beta_file = dev_plugin_dir / "02_beta_plugin_with_duplicate.py"
    beta_file.write_text(plugin_beta_content_duplicate_id)

    ignored_file = dev_plugin_dir / "_ignored_plugin.py"
    ignored_file.write_text("class Ignored: pass")
    txt_file = dev_plugin_dir / "not_a_py_file.txt"
    txt_file.write_text("class NotAPlugin: pass")
    init_file = dev_plugin_dir / "__init__.py"
    init_file.write_text("# Empty init")

    # Simulate rglob returning files in a specific, sorted order
    # The mock should be applied to the Path object representing the dev_dir
    # However, it's simpler to patch Path.rglob globally for this test if pm.plugin_dev_dirs contains Path objects.
    # Since pm.plugin_dev_dirs is initialized with Path objects, this should work.
    # Or, if dev_dir itself is a MagicMock(spec=Path), mock its rglob.

    # Correct approach: Patch rglob on the Path class or a specific instance if possible.
    # Here, self.plugin_dev_dirs in PluginManager will iterate over Path objects.
    # So, if `dev_dir` is a `Path` object, its `rglob` method is called.
    # The current `fresh_plugin_manager` has `pm.plugin_dev_dirs = [dev_plugin_dir]`.
    # So when `dev_dir.rglob` is called, our `mock_rglob` (from the test signature) isn't used
    # unless `dev_dir` itself *is* the mock.

    # Let's make the dev_dir a mock path and control its rglob.
    mock_dev_dir_path_instance = MagicMock(spec=Path)
    mock_dev_dir_path_instance.is_dir.return_value = True
    mock_dev_dir_path_instance.rglob.return_value = sorted([alpha_file, beta_file, ignored_file, txt_file, init_file])
    mock_dev_dir_path_instance.name = dev_plugin_dir.name # for module_import_name
    # For relative_to:
    def mock_relative_to(base_path):
        if base_path == mock_dev_dir_path_instance:
             # This is tricky. We need to return a new Path-like object for each file
             # that can then have .with_suffix("").parts
             # For simplicity, let's assume file paths are already relative enough for the test.
             # This part of PluginManager: `py_file.relative_to(dev_dir).with_suffix("").parts`
             # is hard to mock perfectly without reconstructing Path's behavior.
             # Instead, we can directly control what `importlib.util.spec_from_file_location` gets.
             pass
        return Path("mocked_relative_path") # Placeholder
    mock_dev_dir_path_instance.relative_to.side_effect = lambda p: Path(p.name).parent # Simplistic relative path


    pm.plugin_dev_dirs = [mock_dev_dir_path_instance]

    # The test was already correctly patching Path.rglob globally. The issue lies in `Path.rglob` not being called
    # on the `dev_dir` object we control in the test *if* `pm.plugin_dev_dirs` was set before the patch.
    # By setting `pm.plugin_dev_dirs` *after* patching or ensuring the `Path` objects in it are mocks,
    # we can control it. The current fixture `fresh_plugin_manager` has `plugin_dev_dirs=[]`.
    # So, setting `pm.plugin_dev_dirs = [dev_plugin_dir]` inside the test is fine.
    # The original patch on `pathlib.Path.rglob` from the test signature *should* work.

    # Revert to simpler patching if sort doesn't fix it.
    # The `mock_rglob` from the test signature will patch `Path.rglob`.
    # Ensure it's called on `dev_plugin_dir` instance.

    # The most reliable way to control `rglob` is to ensure the `Path` instance
    # being iterated in `PluginManager` is the one whose `rglob` is mocked.
    # `pm.plugin_dev_dirs` is set from constructor.
    # Let's reconstruct the PM for this test.

    pm_for_this_test = PluginManager(plugin_dev_dirs=[dev_plugin_dir])
    # Now mock_rglob will patch `Path(dev_plugin_dir_str).rglob` if it's called on that.
    # If `dev_plugin_dir` is already a Path object, its `rglob` method is called.
    # The global patch `patch("pathlib.Path.rglob")` means ANY Path instance's rglob is mocked.

    mock_rglob.return_value = sorted([alpha_file, beta_file, ignored_file, txt_file, init_file])


    with patch.object(logging.getLogger("genie_tooling.core.plugin_manager"), "warning") as mock_log_warning:
        await pm_for_this_test.discover_plugins()

    discovered_classes = pm_for_this_test.list_discovered_plugin_classes()
    assert len(discovered_classes) == 2
    assert "discovered_alpha_v1" in discovered_classes
    assert "unique_beta_v1" in discovered_classes

    assert discovered_classes["discovered_alpha_v1"].description == "Discovered Alpha"
    assert pm_for_this_test.get_plugin_source("discovered_alpha_v1") == str(alpha_file)
    assert discovered_classes["unique_beta_v1"].description == "Unique Beta"

    mock_log_warning.assert_any_call(
        "Plugin ID 'discovered_alpha_v1' (dev file) already discovered. Skipping."
    )


@pytest.mark.asyncio
async def test_get_plugin_instance_success_and_caching(fresh_plugin_manager: PluginManager):
    pm = await fresh_plugin_manager
    pm._discovered_plugin_classes[DummyPluginAlpha.plugin_id] = DummyPluginAlpha

    instance1 = await pm.get_plugin_instance(DummyPluginAlpha.plugin_id)
    assert instance1 is not None
    assert isinstance(instance1, DummyPluginAlpha)
    assert instance1.some_value == 1 # type: ignore

    instance2 = await pm.get_plugin_instance(DummyPluginAlpha.plugin_id)
    assert instance2 is instance1

@pytest.mark.asyncio
async def test_get_plugin_instance_not_found(fresh_plugin_manager: PluginManager, caplog: pytest.LogCaptureFixture):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.WARNING)
    instance = await pm.get_plugin_instance("non_existent_plugin_id")
    assert instance is None
    assert "Plugin class ID 'non_existent_plugin_id' not found." in caplog.text

@pytest.mark.asyncio
async def test_get_plugin_instance_setup_called_with_config(fresh_plugin_manager: PluginManager, mocker: MagicMock):
    pm = await fresh_plugin_manager
    mock_setup_method = mocker.AsyncMock()

    class TestSetupPlugin(Plugin):
        plugin_id: str = "test_setup_plugin_v1"
        description: str = "Test Setup"
        setup = mock_setup_method # type: ignore
        async def teardown(self) -> None: pass

    pm._discovered_plugin_classes[TestSetupPlugin.plugin_id] = TestSetupPlugin
    test_config = {"key": "value"}
    instance = await pm.get_plugin_instance(TestSetupPlugin.plugin_id, config=test_config)
    assert instance is not None
    mock_setup_method.assert_awaited_once_with(config=test_config)

@pytest.mark.asyncio
async def test_get_plugin_instance_instantiation_kwargs(fresh_plugin_manager: PluginManager, mocker: MagicMock):
    pm = await fresh_plugin_manager
    mock_init_arg = MagicMock()

    class InitKwargsPlugin(Plugin):
        plugin_id: str = "init_kwargs_plugin_v1"
        description: str = "Plugin with init kwargs"
        arg_received: Any = None
        def __init__(self, custom_arg: Any):
            self.arg_received = custom_arg
        async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
        async def teardown(self) -> None: pass

    pm._discovered_plugin_classes[InitKwargsPlugin.plugin_id] = InitKwargsPlugin
    instance = await pm.get_plugin_instance(InitKwargsPlugin.plugin_id, custom_arg=mock_init_arg) # type: ignore
    assert instance is not None
    assert isinstance(instance, InitKwargsPlugin)
    assert instance.arg_received is mock_init_arg


@pytest.mark.asyncio
async def test_get_all_plugin_instances_by_type(fresh_plugin_manager: PluginManager):
    pm = await fresh_plugin_manager

    class SpecificProtocol(Plugin, TypingProtocol):
        specific_method: Callable[[], str]

    class ImplementsSpecific(Plugin):
        plugin_id: str = "implements_specific_v1"
        description: str = "Implements Specific"
        def specific_method(self) -> str: return "implemented"
        async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
        async def teardown(self) -> None: pass

    pm._discovered_plugin_classes = {
        DummyPluginAlpha.plugin_id: DummyPluginAlpha,
        ImplementsSpecific.plugin_id: ImplementsSpecific,
        SetupFailsPlugin.plugin_id: SetupFailsPlugin
    }

    all_plugins = await pm.get_all_plugin_instances_by_type(Plugin)
    assert len(all_plugins) == 2
    assert any(isinstance(p, DummyPluginAlpha) for p in all_plugins)
    assert any(isinstance(p, ImplementsSpecific) for p in all_plugins)

    specific_plugins = await pm.get_all_plugin_instances_by_type(SpecificProtocol) # type: ignore
    assert len(specific_plugins) == 1
    assert isinstance(specific_plugins[0], ImplementsSpecific)

    configs_for_get_all = {
        "default": {"global_key": "global_val"},
        DummyPluginAlpha.plugin_id: {"alpha_key": "alpha_val"}
    }
    with patch.object(DummyPluginAlpha, "setup", new_callable=AsyncMock) as mock_alpha_setup, \
         patch.object(ImplementsSpecific, "setup", new_callable=AsyncMock) as mock_specific_setup:
        pm._plugin_instances.clear()
        await pm.get_all_plugin_instances_by_type(Plugin, config=configs_for_get_all)

        mock_alpha_setup.assert_called_once_with(config={"global_key": "global_val", "alpha_key": "alpha_val"})
        mock_specific_setup.assert_called_once_with(config={"global_key": "global_val"})


@pytest.mark.asyncio
async def test_teardown_all_plugins_handles_errors(fresh_plugin_manager: PluginManager, caplog: pytest.LogCaptureFixture):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.ERROR)

    pm._discovered_plugin_classes = {
        DummyPluginAlpha.plugin_id: DummyPluginAlpha,
        TeardownFailsPlugin.plugin_id: TeardownFailsPlugin
    }
    await pm.get_plugin_instance(DummyPluginAlpha.plugin_id)
    await pm.get_plugin_instance(TeardownFailsPlugin.plugin_id)
    assert len(pm._plugin_instances) == 2

    await pm.teardown_all_plugins()
    assert len(pm._plugin_instances) == 0
    expected_log_fragment = f"Error tearing down plugin '{TeardownFailsPlugin.plugin_id}': Teardown deliberately failed"
    assert any(expected_log_fragment in record.message for record in caplog.records if "Traceback" not in record.message)


@pytest.mark.asyncio
@patch("importlib.metadata.entry_points")
async def test_discover_plugins_from_entry_points_load_error(mock_entry_points_func: MagicMock, fresh_plugin_manager: PluginManager, caplog: pytest.LogCaptureFixture):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.ERROR)

    mock_ep_fails_load = MagicMock(spec=importlib.metadata.EntryPoint)
    mock_ep_fails_load.name = "entry_point_fails_load"
    mock_ep_fails_load.group = "genie_tooling.plugins"
    mock_ep_fails_load.load.side_effect = ImportError("Cannot load this entry point")

    mock_eps_selectable = MagicMock()
    mock_eps_selectable.select.return_value = [mock_ep_fails_load]
    mock_entry_points_func.return_value = mock_eps_selectable

    await pm.discover_plugins()
    expected_log_fragment = "Error loading plugin from entry point entry_point_fails_load: Cannot load this entry point"
    assert any(expected_log_fragment in record.message for record in caplog.records if "Traceback" not in record.message)
    assert len(pm.list_discovered_plugin_classes()) == 0


@pytest.mark.asyncio
@patch("importlib.metadata.entry_points")
async def test_discover_plugins_from_entry_points_invalid_object_type(mock_entry_points_func: MagicMock, fresh_plugin_manager: PluginManager, caplog: pytest.LogCaptureFixture):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.WARNING)

    mock_ep_invalid_obj = MagicMock(spec=importlib.metadata.EntryPoint)
    mock_ep_invalid_obj.name = "entry_point_invalid_obj"
    mock_ep_invalid_obj.group = "genie_tooling.plugins"
    mock_ep_invalid_obj.load.return_value = "not_a_class_or_module"

    mock_eps_selectable = MagicMock()
    mock_eps_selectable.select.return_value = [mock_ep_invalid_obj]
    mock_entry_points_func.return_value = mock_eps_selectable

    await pm.discover_plugins()
    assert "Entry point 'entry_point_invalid_obj' loaded invalid object type '<class 'str'>'." in caplog.text
    assert len(pm.list_discovered_plugin_classes()) == 0


@pytest.mark.asyncio
@patch("importlib.metadata.entry_points")
async def test_discover_plugins_entry_points_general_error(mock_entry_points_func: MagicMock, fresh_plugin_manager: PluginManager, caplog: pytest.LogCaptureFixture):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.ERROR)
    mock_entry_points_func.side_effect = Exception("General error listing entry points")
    await pm.discover_plugins()
    expected_log_fragment = "Error iterating entry points for group 'genie_tooling.plugins': General error listing entry points"
    assert any(expected_log_fragment in record.message for record in caplog.records if "Traceback" not in record.message)


@pytest.mark.asyncio
async def test_get_plugin_instance_setup_failure(fresh_plugin_manager: PluginManager, caplog: pytest.LogCaptureFixture):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.ERROR)
    pm._discovered_plugin_classes[SetupFailsPlugin.plugin_id] = SetupFailsPlugin

    instance = await pm.get_plugin_instance(SetupFailsPlugin.plugin_id)
    assert instance is None
    assert SetupFailsPlugin.plugin_id not in pm._plugin_instances
    expected_log_fragment = f"Error instantiating/setting up plugin '{SetupFailsPlugin.plugin_id}': Setup deliberately failed"
    assert any(expected_log_fragment in record.message for record in caplog.records if "Traceback" not in record.message)

@pytest.mark.asyncio
async def test_discover_plugins_dev_dir_malformed_file(tmp_path: Path, fresh_plugin_manager: PluginManager, caplog: pytest.LogCaptureFixture):
    pm = await fresh_plugin_manager
    dev_dir = tmp_path / "malformed_plugins"
    dev_dir.mkdir()

    malformed_content = "class MalformedPlugin(Plugin):\n  plugin_id = 'malformed_v1'\n  description = 'test'\n  async def setup(self, config=None) pass\n"
    (dev_dir / "malformed_plugin.py").write_text(malformed_content)

    import_error_content = "import non_existent_module_for_test\nclass ImportProblemPlugin(Plugin):\n  plugin_id='import_error_v1'\n  description='fails import'"
    (dev_dir / "import_error_plugin.py").write_text(import_error_content)


    pm.plugin_dev_dirs = [dev_dir]

    caplog.set_level(logging.ERROR)
    await pm.discover_plugins()

    assert f"Error loading plugin module from dev file {dev_dir / 'malformed_plugin.py'}" in caplog.text
    assert f"Error loading plugin module from dev file {dev_dir / 'import_error_plugin.py'}" in caplog.text
    assert "No module named 'non_existent_module_for_test'" in caplog.text

    assert len(pm.list_discovered_plugin_classes()) == 0
