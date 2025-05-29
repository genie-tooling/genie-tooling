import abc
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin


class DummyPluginAlpha(Plugin):
    plugin_id: str = "dummy_alpha_v1"; description: str = "Alpha"; some_value: int = 1
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass
class DummyPluginBeta(Plugin):
    plugin_id: str = "dummy_beta_v1"; description: str = "A beta plugin"
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass
class NotAPlugin: pass
class SetupFailsPlugin(Plugin):
    plugin_id: str = "setup_fails_v1"; description: str = "Fails setup"
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: raise RuntimeError("Setup deliberately failed")
    async def teardown(self) -> None: pass
class TeardownFailsPlugin(Plugin):
    plugin_id: str = "teardown_fails_v1"; description: str = "Fails teardown"
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: raise RuntimeError("Teardown deliberately failed")
class AbstractPluginBase(Plugin, abc.ABC):
    plugin_id: str = "abstract_base_v1"; description: str = "Abstract base"
    @abc.abstractmethod
    async def an_abstract_method(self) -> None: pass
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

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

@patch("importlib.metadata.entry_points", return_value=MagicMock(select=MagicMock(return_value=[])))
@patch("pathlib.Path.rglob")
@pytest.mark.asyncio
async def test_plugin_manager_discover_from_dev_dir(mock_rglob: MagicMock, mock_eps, tmp_path: Path):
    dev_plugin_dir = tmp_path / "test_plugins_deterministic_order"; dev_plugin_dir.mkdir()
    plugin_alpha_content = "from genie_tooling.core.types import Plugin\nfrom typing import Dict, Any, Optional\nclass DiscoveredPluginAlpha(Plugin):\n    plugin_id: str = 'discovered_alpha_v1'\n    description: str = 'Discovered Alpha'\n    async def setup(self, config: Optional[Dict[str, Any]] = None):pass\n    async def teardown(self):pass\n"
    alpha_file = dev_plugin_dir / "01_alpha_plugin.py"; alpha_file.write_text(plugin_alpha_content)
    plugin_beta_content_duplicate_id = "from genie_tooling.core.types import Plugin\nfrom typing import Dict, Any, Optional\nclass DiscoveredPluginBetaDuplicate(Plugin):\n    plugin_id: str = 'discovered_alpha_v1'\n    description: str = 'Discovered Beta with duplicate ID'\n    async def setup(self, config: Optional[Dict[str, Any]] = None):pass\n    async def teardown(self):pass\nclass UniquePluginInBetaFile(Plugin):\n    plugin_id: str = 'unique_beta_v1'\n    description: str = 'Unique Beta'\n    async def setup(self, config: Optional[Dict[str, Any]] = None):pass\n    async def teardown(self):pass\n"
    beta_file = dev_plugin_dir / "02_beta_plugin_with_duplicate.py"; beta_file.write_text(plugin_beta_content_duplicate_id)
    mock_rglob.return_value = sorted([alpha_file, beta_file])
    pm_for_this_test = PluginManager(plugin_dev_dirs=[str(dev_plugin_dir)])
    with patch.object(logging.getLogger("genie_tooling.core.plugin_manager"), "warning") as mock_log_warning:
        await pm_for_this_test.discover_plugins()
    discovered_classes = pm_for_this_test.list_discovered_plugin_classes()
    assert len(discovered_classes) == 2, f"Discovered: {discovered_classes.keys()}"
    assert "discovered_alpha_v1" in discovered_classes; assert "unique_beta_v1" in discovered_classes
    assert discovered_classes["discovered_alpha_v1"].description == "Discovered Alpha"
    assert pm_for_this_test.get_plugin_source("discovered_alpha_v1") == str(alpha_file)
    mock_log_warning.assert_any_call("Plugin ID 'discovered_alpha_v1' (dev file) already discovered. Skipping.")

@patch("importlib.metadata.entry_points", return_value=MagicMock(select=MagicMock(return_value=[])))
@pytest.mark.asyncio
async def test_discover_plugins_dev_dir_malformed_file(mock_eps, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    pm = PluginManager(plugin_dev_dirs=[])
    dev_dir = tmp_path / "malformed_plugins"; dev_dir.mkdir()
    malformed_content = "class MalformedPlugin(Plugin):\n  plugin_id = 'malformed_v1'\n  description = 'test'\n  async def setup(self, config=None) pass\n"; (dev_dir / "malformed_plugin.py").write_text(malformed_content)
    import_error_content = "import non_existent_module_for_test\nclass ImportProblemPlugin(Plugin):\n  plugin_id='import_error_v1'\n  description='fails import'"; (dev_dir / "import_error_plugin.py").write_text(import_error_content)
    pm.plugin_dev_dirs = [dev_dir]; caplog.set_level(logging.ERROR); await pm.discover_plugins()
    assert f"Error loading plugin module from dev file {dev_dir / 'malformed_plugin.py'}" in caplog.text
    assert f"Error loading plugin module from dev file {dev_dir / 'import_error_plugin.py'}" in caplog.text
    assert "No module named 'non_existent_module_for_test'" in caplog.text
    assert len(pm.list_discovered_plugin_classes()) == 0
