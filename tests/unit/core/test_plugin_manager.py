### tests/unit/core/test_plugin_manager.py
import abc
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch  # Added PropertyMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin


class DummyPluginAlpha(Plugin):
    plugin_id: str = "dummy_alpha_v1"
    description: str = "Alpha"
    some_value: int = 1
    setup_called_with_config: Optional[Dict[str, Any]] = None
    teardown_called: bool = False

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.setup_called_with_config = config
        self.teardown_called = False

    async def teardown(self) -> None:
        self.teardown_called = True

class DummyPluginBeta(Plugin):
    plugin_id: str = "dummy_beta_v1"
    description: str = "A beta plugin"
    setup_called_with_config: Optional[Dict[str, Any]] = None
    teardown_called: bool = False

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.setup_called_with_config = config
        self.teardown_called = False

    async def teardown(self) -> None:
        self.teardown_called = True

class NotAPlugin:
    pass

class InitFailsPlugin(Plugin):
    plugin_id: str = "init_fails_v1"
    description: str = "Fails __init__"
    def __init__(self, required_arg: str): # pylint: disable=super-init-not-called
        self.required_arg = required_arg
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass


class SetupFailsPlugin(Plugin):
    plugin_id: str = "setup_fails_v1"
    description: str = "Fails setup"
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        raise RuntimeError("Setup deliberately failed")
    async def teardown(self) -> None:
        pass

class TeardownFailsPlugin(Plugin):
    plugin_id: str = "teardown_fails_v1"
    description: str = "Fails teardown"
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass
    async def teardown(self) -> None:
        raise RuntimeError("Teardown deliberately failed")

class AbstractPluginBase(Plugin, abc.ABC):
    plugin_id: str = "abstract_base_v1"
    description: str = "Abstract base"
    @abc.abstractmethod
    async def an_abstract_method(self) -> None:
        pass
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass
    async def teardown(self) -> None:
        pass

@pytest.fixture
async def fresh_plugin_manager() -> PluginManager:
    return PluginManager(plugin_dev_dirs=[])

@pytest.mark.asyncio
async def test_plugin_manager_initialization(fresh_plugin_manager: PluginManager):
    actual_pm = await fresh_plugin_manager
    assert actual_pm is not None
    assert isinstance(actual_pm, PluginManager)
    assert actual_pm.list_discovered_plugin_classes() is not None
    assert actual_pm.plugin_dev_dirs == []

@patch("importlib.metadata.entry_points", return_value=MagicMock(select=MagicMock(return_value=[])))
@pytest.mark.asyncio
async def test_plugin_manager_discover_from_empty_dev_dirs(mock_eps, fresh_plugin_manager: PluginManager):
    actual_pm = await fresh_plugin_manager
    await actual_pm.discover_plugins()
    assert len(actual_pm.list_discovered_plugin_classes()) == 0
    pm_none_dirs = PluginManager(plugin_dev_dirs=None)
    await pm_none_dirs.discover_plugins()
    assert len(pm_none_dirs.list_discovered_plugin_classes()) == 0

@patch("importlib.metadata.entry_points", return_value=MagicMock(select=MagicMock(return_value=[])))
@pytest.mark.asyncio
async def test_plugin_manager_discover_from_non_existent_dev_dir(mock_eps, caplog: pytest.LogCaptureFixture):
    non_existent_dir_str = "./this_dir_does_not_exist_hopefully"
    pm_test_specific = PluginManager(plugin_dev_dirs=[non_existent_dir_str])
    caplog.set_level(logging.WARNING)
    await pm_test_specific.discover_plugins()
    assert len(pm_test_specific.list_discovered_plugin_classes()) == 0
    resolved_path_logged = Path(non_existent_dir_str).resolve()
    expected_log_message = f"Plugin dev dir '{resolved_path_logged}' not found. Skipping."
    assert expected_log_message in caplog.text

@pytest.fixture
def plugin_manager_for_sync_test() -> PluginManager:
    return PluginManager()

def test_is_valid_plugin_class_check(plugin_manager_for_sync_test: PluginManager):
    pm_sync = plugin_manager_for_sync_test
    assert pm_sync._is_valid_plugin_class(DummyPluginAlpha) is True
    assert pm_sync._is_valid_plugin_class(NotAPlugin) is False
    assert pm_sync._is_valid_plugin_class(Plugin) is False
    assert pm_sync._is_valid_plugin_class(AbstractPluginBase) is False
    assert pm_sync._is_valid_plugin_class(None) is False
    assert pm_sync._is_valid_plugin_class(123) is False

@patch("importlib.metadata.entry_points")
@patch("inspect.ismodule")
@pytest.mark.asyncio
async def test_discover_plugins_from_entry_points(mock_ismodule: MagicMock, mock_entry_points_func: MagicMock, fresh_plugin_manager: PluginManager):
    pm = await fresh_plugin_manager

    mock_ep1 = MagicMock(name="ep1")
    mock_ep1.name = "plugin_alpha_ep"
    mock_ep1.load.return_value = DummyPluginAlpha

    mock_ep2_module = MagicMock(name="mock_ep2_module_instance")
    # Simulate module content: the attribute name is 'BetaPluginFromModule', the value is the class DummyPluginBeta
    type(mock_ep2_module).BetaPluginFromModule = DummyPluginBeta # Use type() to set attribute on MagicMock
    type(mock_ep2_module).NotAPluginInModule = NotAPlugin

    mock_ep2 = MagicMock(name="ep2")
    mock_ep2.name = "plugin_beta_ep_module"
    mock_ep2.load.return_value = mock_ep2_module

    def ismodule_side_effect(obj):
        if obj is mock_ep2_module:
            return True
        return isinstance(obj, type(abc))
    mock_ismodule.side_effect = ismodule_side_effect


    mock_eps_container = MagicMock()
    mock_eps_container.select.return_value = [mock_ep1, mock_ep2]
    mock_entry_points_func.return_value = mock_eps_container

    await pm.discover_plugins()
    discovered = pm.list_discovered_plugin_classes()

    assert len(discovered) == 2
    assert "dummy_alpha_v1" in discovered
    assert discovered["dummy_alpha_v1"] == DummyPluginAlpha
    assert pm.get_plugin_source("dummy_alpha_v1") == "entry_point:plugin_alpha_ep"

    assert "dummy_beta_v1" in discovered
    assert discovered["dummy_beta_v1"] == DummyPluginBeta
    # Corrected assertion: plugin_class.__name__ will be "DummyPluginBeta"
    assert pm.get_plugin_source("dummy_beta_v1") == "entry_point_module:plugin_beta_ep_module:DummyPluginBeta"


    mock_eps_container.select.assert_called_once_with(group="genie_tooling.plugins")

@patch("importlib.metadata.entry_points")
@pytest.mark.asyncio
async def test_discover_plugins_entry_point_load_error(mock_entry_points_func, fresh_plugin_manager: PluginManager, caplog):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.ERROR)

    mock_ep_fail = MagicMock(name="ep_fail")
    mock_ep_fail.name = "failing_ep"
    mock_ep_fail.load.side_effect = ImportError("Simulated import error")

    mock_eps_container = MagicMock()
    mock_eps_container.select.return_value = [mock_ep_fail]
    mock_entry_points_func.return_value = mock_eps_container

    await pm.discover_plugins()
    assert len(pm.list_discovered_plugin_classes()) == 0
    assert "Error loading plugin from entry point failing_ep: Simulated import error" in caplog.text

@patch("importlib.metadata.entry_points")
@patch("inspect.ismodule")
@pytest.mark.asyncio
async def test_discover_plugins_entry_point_invalid_object(mock_ismodule_invalid: MagicMock, mock_entry_points_func, fresh_plugin_manager: PluginManager, caplog):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.WARNING)

    mock_ep_invalid = MagicMock(name="ep_invalid")
    mock_ep_invalid.name = "invalid_object_ep"
    invalid_obj = "this_is_a_string_not_a_plugin"
    mock_ep_invalid.load.return_value = invalid_obj
    mock_ismodule_invalid.side_effect = lambda obj: False if obj is invalid_obj else isinstance(obj, type(abc))


    mock_eps_container = MagicMock()
    mock_eps_container.select.return_value = [mock_ep_invalid]
    mock_entry_points_func.return_value = mock_eps_container

    await pm.discover_plugins()
    assert len(pm.list_discovered_plugin_classes()) == 0
    assert "Entry point 'invalid_object_ep' loaded invalid object type '<class 'str'>'" in caplog.text


@patch("importlib.metadata.entry_points", return_value=MagicMock(select=MagicMock(return_value=[])))
@patch("pathlib.Path.rglob")
@pytest.mark.asyncio
async def test_plugin_manager_discover_from_dev_dir(mock_rglob: MagicMock, mock_eps, tmp_path: Path):
    dev_plugin_dir = tmp_path / "test_plugins_deterministic_order"
    dev_plugin_dir.mkdir()
    plugin_alpha_content = "from genie_tooling.core.types import Plugin\nfrom typing import Dict, Any, Optional\nclass DiscoveredPluginAlpha(Plugin):\n    plugin_id: str = 'discovered_alpha_v1'\n    description: str = 'Discovered Alpha'\n    async def setup(self, config: Optional[Dict[str, Any]] = None):pass\n    async def teardown(self):pass\n"
    alpha_file = dev_plugin_dir / "01_alpha_plugin.py"
    alpha_file.write_text(plugin_alpha_content)
    plugin_beta_content_duplicate_id = "from genie_tooling.core.types import Plugin\nfrom typing import Dict, Any, Optional\nclass DiscoveredPluginBetaDuplicate(Plugin):\n    plugin_id: str = 'discovered_alpha_v1'\n    description: str = 'Discovered Beta with duplicate ID'\n    async def setup(self, config: Optional[Dict[str, Any]] = None):pass\n    async def teardown(self):pass\nclass UniquePluginInBetaFile(Plugin):\n    plugin_id: str = 'unique_beta_v1'\n    description: str = 'Unique Beta'\n    async def setup(self, config: Optional[Dict[str, Any]] = None):pass\n    async def teardown(self):pass\n"
    beta_file = dev_plugin_dir / "02_beta_plugin_with_duplicate.py"
    beta_file.write_text(plugin_beta_content_duplicate_id)

    mock_rglob.return_value = sorted([alpha_file, beta_file])

    pm_for_this_test = PluginManager(plugin_dev_dirs=[str(dev_plugin_dir)])
    with patch.object(logging.getLogger("genie_tooling.core.plugin_manager"), "warning") as mock_log_warning:
        await pm_for_this_test.discover_plugins()

    discovered_classes = pm_for_this_test.list_discovered_plugin_classes()
    assert len(discovered_classes) == 2, f"Discovered: {discovered_classes.keys()}"
    assert "discovered_alpha_v1" in discovered_classes
    assert "unique_beta_v1" in discovered_classes
    assert discovered_classes["discovered_alpha_v1"].description == "Discovered Alpha"
    assert pm_for_this_test.get_plugin_source("discovered_alpha_v1") == str(alpha_file)
    mock_log_warning.assert_any_call("Plugin ID 'discovered_alpha_v1' (dev file) already discovered. Skipping.")


@patch("importlib.metadata.entry_points", return_value=MagicMock(select=MagicMock(return_value=[])))
@pytest.mark.asyncio
async def test_discover_plugins_dev_dir_malformed_file(mock_eps, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    pm = PluginManager(plugin_dev_dirs=[])
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


@pytest.mark.asyncio
async def test_get_plugin_instance_success_and_cache(fresh_plugin_manager: PluginManager):
    pm = await fresh_plugin_manager
    pm._discovered_plugin_classes["dummy_alpha_v1"] = DummyPluginAlpha
    config_arg = {"key": "value"}

    instance1 = await pm.get_plugin_instance("dummy_alpha_v1", config=config_arg)
    assert isinstance(instance1, DummyPluginAlpha)
    assert instance1.setup_called_with_config == config_arg

    instance2 = await pm.get_plugin_instance("dummy_alpha_v1", config={"other": "config"})
    assert instance2 is instance1
    # Verify setup was NOT called again on the cached instance with the new config
    assert instance1.setup_called_with_config == config_arg # Original config should persist


@pytest.mark.asyncio
async def test_get_plugin_instance_not_found(fresh_plugin_manager: PluginManager, caplog):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.WARNING)
    instance = await pm.get_plugin_instance("non_existent_plugin_id")
    assert instance is None
    assert "Plugin class ID 'non_existent_plugin_id' not found." in caplog.text

@pytest.mark.asyncio
async def test_get_plugin_instance_init_fails(fresh_plugin_manager: PluginManager, caplog):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.ERROR)
    pm._discovered_plugin_classes["init_fails_v1"] = InitFailsPlugin
    instance_no_kwargs = await pm.get_plugin_instance("init_fails_v1")
    assert instance_no_kwargs is None
    assert "Error instantiating/setting up plugin 'init_fails_v1'" in caplog.text
    assert "missing 1 required positional argument: 'required_arg'" in caplog.text or \
           "takes no arguments" in caplog.text


@pytest.mark.asyncio
async def test_get_plugin_instance_setup_fails(fresh_plugin_manager: PluginManager, caplog):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.ERROR)
    pm._discovered_plugin_classes["setup_fails_v1"] = SetupFailsPlugin
    instance = await pm.get_plugin_instance("setup_fails_v1")
    assert instance is None
    assert "Error instantiating/setting up plugin 'setup_fails_v1'" in caplog.text
    assert "Setup deliberately failed" in caplog.text

@pytest.mark.asyncio
async def test_get_all_plugin_instances_by_type(fresh_plugin_manager: PluginManager):
    pm = await fresh_plugin_manager
    pm._discovered_plugin_classes = {
        "dummy_alpha_v1": DummyPluginAlpha,
        "dummy_beta_v1": DummyPluginBeta,
        "not_plugin_id": NotAPlugin, # type: ignore
        "setup_fails_v1": SetupFailsPlugin
    }
    config_map = {
        "dummy_alpha_v1": {"alpha_specific": True},
        "default": {"global_default": True}
    }
    alpha_instances = await pm.get_all_plugin_instances_by_type(DummyPluginAlpha, config=config_map)
    assert len(alpha_instances) == 1
    assert isinstance(alpha_instances[0], DummyPluginAlpha)
    assert alpha_instances[0].setup_called_with_config == {"global_default": True, "alpha_specific": True}

    alpha_instances2 = await pm.get_all_plugin_instances_by_type(DummyPluginAlpha, config=config_map)
    assert len(alpha_instances2) == 1
    assert alpha_instances2[0] is alpha_instances[0]

    all_valid_plugins = await pm.get_all_plugin_instances_by_type(Plugin)
    assert len(all_valid_plugins) == 2
    assert any(isinstance(p, DummyPluginAlpha) for p in all_valid_plugins)
    assert any(isinstance(p, DummyPluginBeta) for p in all_valid_plugins)
    beta_instance = next(p for p in all_valid_plugins if isinstance(p, DummyPluginBeta))
    assert beta_instance.setup_called_with_config == {"global_default": True}

    pm_empty = PluginManager()
    pm_empty._discovered_plugin_classes = { "dummy_alpha_v1": DummyPluginAlpha }
    discovered_alpha = await pm_empty.get_all_plugin_instances_by_type(DummyPluginAlpha)
    assert len(discovered_alpha) == 1


@pytest.mark.asyncio
async def test_teardown_all_plugins(fresh_plugin_manager: PluginManager, caplog):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.ERROR)
    pm._discovered_plugin_classes = {
        "dummy_alpha_v1": DummyPluginAlpha,
        "teardown_fails_v1": TeardownFailsPlugin
    }
    alpha_instance = await pm.get_plugin_instance("dummy_alpha_v1")
    teardown_fail_instance = await pm.get_plugin_instance("teardown_fails_v1")
    assert alpha_instance is not None
    assert teardown_fail_instance is not None

    await pm.teardown_all_plugins()

    assert alpha_instance.teardown_called
    assert "Error tearing down plugin 'teardown_fails_v1'" in caplog.text
    assert "Teardown deliberately failed" in caplog.text
    assert len(pm._plugin_instances) == 0

    await pm.teardown_all_plugins()

@pytest.mark.asyncio
async def test_get_plugin_source_not_found(fresh_plugin_manager: PluginManager):
    pm = await fresh_plugin_manager
    assert pm.get_plugin_source("non_existent") is None


@patch("importlib.metadata.entry_points")
@pytest.mark.asyncio
async def test_discover_plugins_entry_point_duplicate_id_handling(mock_entry_points_func, fresh_plugin_manager: PluginManager, caplog):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.WARNING)

    mock_ep1 = MagicMock(name="ep1_dup")
    mock_ep1.name = "ep1_dup_name"
    mock_ep1.load.return_value = DummyPluginAlpha

    mock_ep2 = MagicMock(name="ep2_dup")
    mock_ep2.name = "ep2_dup_name"
    class DummyPluginAlphaDuplicate(Plugin):
        plugin_id: str = "dummy_alpha_v1"
        description: str = "Alpha Duplicate via EP2"
        async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
        async def teardown(self) -> None: pass
    mock_ep2.load.return_value = DummyPluginAlphaDuplicate

    mock_eps_container = MagicMock()
    mock_eps_container.select.return_value = [mock_ep1, mock_ep2]
    mock_entry_points_func.return_value = mock_eps_container

    await pm.discover_plugins()
    discovered = pm.list_discovered_plugin_classes()

    assert len(discovered) == 1
    assert "dummy_alpha_v1" in discovered
    assert discovered["dummy_alpha_v1"] == DummyPluginAlpha
    assert discovered["dummy_alpha_v1"].description == "Alpha"
    assert "Plugin ID 'dummy_alpha_v1' (direct entry point) already discovered. Skipping." in caplog.text

@patch("importlib.metadata.entry_points", return_value=MagicMock(select=MagicMock(return_value=[])))
@patch("pathlib.Path.rglob")
@patch("importlib.util.spec_from_file_location")
@patch("importlib.util.module_from_spec")
@pytest.mark.asyncio
async def test_discover_plugins_dev_dir_skips_dunder_files(
    mock_module_from_spec: MagicMock,
    mock_spec_from_file: MagicMock,
    mock_rglob: MagicMock,
    mock_eps_func: MagicMock,
    tmp_path: Path,
    fresh_plugin_manager: PluginManager,
    caplog
):
    pm = await fresh_plugin_manager
    pm.plugin_dev_dirs = [tmp_path]
    caplog.set_level(logging.DEBUG)

    mock_eps_func.return_value.select.return_value = []

    init_py_path = tmp_path / "__init__.py"
    init_py_path.touch()
    dot_file_path = tmp_path / ".hidden_plugin.py"
    dot_file_path.touch()
    underscore_file_path = tmp_path / "_internal_plugin.py"
    underscore_file_path.touch()

    mock_rglob.return_value = [init_py_path, dot_file_path, underscore_file_path]

    await pm.discover_plugins()
    mock_spec_from_file.assert_not_called()
    mock_module_from_spec.assert_not_called()
    assert len(pm.list_discovered_plugin_classes()) == 0

@patch("importlib.metadata.entry_points")
@pytest.mark.asyncio
async def test_discover_plugins_entry_point_loader_failure(
    mock_entry_points_func,
    fresh_plugin_manager: PluginManager,
    caplog
):
    pm = await fresh_plugin_manager
    caplog.set_level(logging.ERROR)
    mock_entry_points_func.side_effect = Exception("Failed to access entry points system.")
    await pm.discover_plugins()
    assert "Error iterating entry points for group 'genie_tooling.plugins'" in caplog.text
    assert "Failed to access entry points system." in caplog.text
    assert len(pm.list_discovered_plugin_classes()) == 0

@patch("importlib.metadata.entry_points", return_value=MagicMock(select=MagicMock(return_value=[])))
@patch("pathlib.Path.rglob")
@patch("importlib.util.spec_from_file_location")
@pytest.mark.asyncio
async def test_discover_plugins_dev_dir_spec_load_returns_none(
    mock_spec_from_file: MagicMock,
    mock_rglob: MagicMock,
    mock_eps_func: MagicMock,
    tmp_path: Path,
    fresh_plugin_manager: PluginManager,
    caplog
):
    pm = await fresh_plugin_manager
    pm.plugin_dev_dirs = [tmp_path]
    caplog.set_level(logging.WARNING) # Changed from ERROR to WARNING for this log
    mock_eps_func.return_value.select.return_value = []

    plugin_file = tmp_path / "some_plugin.py"
    plugin_file.touch()
    mock_rglob.return_value = [plugin_file]
    mock_spec_from_file.return_value = None

    await pm.discover_plugins()

    # Check that it logged the specific warning
    assert f"Could not create module spec for dev file {plugin_file}. Skipping." in caplog.text
    assert len(pm.list_discovered_plugin_classes()) == 0
