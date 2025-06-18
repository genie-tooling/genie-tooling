### src/genie_tooling/core/plugin_manager.py
"""
PluginManager for discovering, loading, and managing plugins.
"""
import importlib
import importlib.metadata
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, cast

from .types import Plugin, PluginType

logger = logging.getLogger(__name__)

class PluginManager:
    """
    Manages discovery, loading, and access to plugins.
    Implements Hybrid: Entry Points + Configured Dev Directory discovery.
    """
    def __init__(self, plugin_dev_dirs: Optional[List[str]] = None):
        self.plugin_dev_dirs = [Path(p).resolve() for p in plugin_dev_dirs] if plugin_dev_dirs else []
        self._plugin_instances: Dict[str, Plugin] = {}
        self._discovered_plugin_classes: Dict[str, Type[Plugin]] = {}
        self._plugin_source_map: Dict[str, str] = {}
        logger.debug(f"PluginManager initialized. Dev dirs: {self.plugin_dev_dirs}")

    async def discover_plugins(self) -> None:
        self._discovered_plugin_classes.clear()
        self._plugin_source_map.clear()
        discovered_ids: Set[str] = set()
        entry_point_group = "genie_tooling.plugins"
        logger.debug(f"Discovering plugins from entry point group: '{entry_point_group}'")
        try:
            eps = importlib.metadata.entry_points()
            if hasattr(eps, "select"):
                selected_eps = eps.select(group=entry_point_group)
            elif isinstance(eps, dict):
                selected_eps = eps.get(entry_point_group, [])
            else:
                selected_eps = []

            for entry_point in selected_eps:
                logger.debug(f"Processing entry point: {entry_point.name}")
                try:
                    plugin_class_or_module = entry_point.load()
                    if inspect.ismodule(plugin_class_or_module):
                        for _, member_obj in inspect.getmembers(plugin_class_or_module):
                            if self._is_valid_plugin_class(member_obj):
                                plugin_class = cast(Type[Plugin], member_obj)
                                if plugin_class.plugin_id in discovered_ids:
                                    logger.warning(f"Plugin ID '{plugin_class.plugin_id}' (module member) already discovered. Skipping.")
                                    continue
                                self._discovered_plugin_classes[plugin_class.plugin_id] = plugin_class
                                self._plugin_source_map[plugin_class.plugin_id] = f"entry_point_module:{entry_point.name}:{plugin_class.__name__}"
                                discovered_ids.add(plugin_class.plugin_id)
                                logger.debug(f"Discovered plugin class '{plugin_class.plugin_id}' from entry point '{entry_point.name}' (module scan).")
                    elif self._is_valid_plugin_class(plugin_class_or_module):
                        plugin_class = cast(Type[Plugin], plugin_class_or_module)
                        if plugin_class.plugin_id in discovered_ids:
                            logger.warning(f"Plugin ID '{plugin_class.plugin_id}' (direct entry point) already discovered. Skipping.")
                            continue
                        self._discovered_plugin_classes[plugin_class.plugin_id] = plugin_class
                        self._plugin_source_map[plugin_class.plugin_id] = f"entry_point:{entry_point.name}"
                        discovered_ids.add(plugin_class.plugin_id)
                        logger.debug(f"Discovered plugin class '{plugin_class.plugin_id}' from entry point '{entry_point.name}'.")
                    elif inspect.isfunction(plugin_class_or_module):
                        logger.debug(f"Entry point '{entry_point.name}' points to a function, not a plugin class. It should be registered via `genie.register_tool_functions()`. Skipping discovery.")
                    else:
                        logger.warning(f"Entry point '{entry_point.name}' loaded invalid object type '{type(plugin_class_or_module)}' for plugin discovery.")
                except Exception as e:
                    logger.error(f"Error loading plugin from entry point {entry_point.name}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error iterating entry points for group '{entry_point_group}': {e}", exc_info=True)

        for dev_dir in self.plugin_dev_dirs:
            if not dev_dir.is_dir():
                logger.warning(f"Plugin dev dir '{dev_dir}' not found. Skipping.")
                continue
            logger.debug(f"Scanning for plugins in dev directory: {dev_dir}")
            for py_file in dev_dir.rglob("*.py"):
                if py_file.name.startswith(("_", ".")) or py_file.name == "__init__.py":
                    continue
                relative_path_parts = py_file.relative_to(dev_dir).with_suffix("").parts
                module_import_name = f"dev_plugins.{dev_dir.name}.{'.'.join(relative_path_parts)}"
                try:
                    module_spec = importlib.util.spec_from_file_location(module_import_name, py_file)
                    if module_spec and module_spec.loader:
                        module = importlib.util.module_from_spec(module_spec)
                        module_spec.loader.exec_module(module)
                        for _, member_obj in inspect.getmembers(module):
                            if self._is_valid_plugin_class(member_obj):
                                plugin_class = cast(Type[Plugin], member_obj)
                                if plugin_class.plugin_id in discovered_ids:
                                    logger.warning(f"Plugin ID '{plugin_class.plugin_id}' (dev file) already discovered. Skipping.")
                                    continue
                                self._discovered_plugin_classes[plugin_class.plugin_id] = plugin_class
                                self._plugin_source_map[plugin_class.plugin_id] = str(py_file)
                                discovered_ids.add(plugin_class.plugin_id)
                                logger.debug(f"Discovered plugin class '{plugin_class.plugin_id}' from dev file '{py_file}'.")
                    elif not module_spec:
                        logger.warning(f"Could not create module spec for dev file {py_file}. Skipping.")

                except Exception as e:
                    logger.error(f"Error loading plugin module from dev file {py_file}: {e}", exc_info=True)
        logger.info(f"Plugin discovery complete. Found {len(self._discovered_plugin_classes)} unique plugin classes.")

    def _is_valid_plugin_class(self, obj: Any) -> bool:
        if not inspect.isclass(obj):
            return False
        if not hasattr(obj, "plugin_id"): # Ensure plugin_id attribute exists
            return False
        if inspect.isabstract(obj):
            return False
        if obj is Plugin: # Explicitly exclude the base Plugin protocol
            return False
        return True


    async def get_plugin_instance(self, plugin_id: str, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Optional[PluginType]:
        if plugin_id in self._plugin_instances:
            logger.debug(f"Returning existing instance for plugin '{plugin_id}'.")
            return cast(PluginType, self._plugin_instances[plugin_id])
        plugin_class = self._discovered_plugin_classes.get(plugin_id)
        if not plugin_class:
            logger.warning(f"Plugin class ID '{plugin_id}' not found.")
            return None
        try:
            instance = plugin_class(**kwargs) # type: ignore
            logger.debug(f"PluginManager.get_plugin_instance: Instantiated plugin '{plugin_id}'. Calling setup with config: {config}")
            await instance.setup(config=config or {})
            self._plugin_instances[plugin_id] = instance
            return cast(PluginType, instance)
        except Exception as e:
            logger.error(f"Error instantiating/setting up plugin '{plugin_id}': {e}", exc_info=True)
            return None

    async def get_all_plugin_instances_by_type(self, plugin_protocol_type: Type[PluginType], config: Optional[Dict[str, Any]] = None) -> List[PluginType]:
        instances: List[PluginType] = []
        for plugin_id, _ in self._discovered_plugin_classes.items():
            if plugin_id in self._plugin_instances:
                existing_instance = self._plugin_instances[plugin_id]
                if isinstance(existing_instance, plugin_protocol_type):
                    instances.append(cast(PluginType, existing_instance))
                continue

            plugin_specific_config = (config or {}).get(plugin_id, {})
            instance_setup_config = {**(config or {}).get("default", {}), **plugin_specific_config}

            instance = await self.get_plugin_instance(plugin_id, config=instance_setup_config)
            if instance and isinstance(instance, plugin_protocol_type):
                instances.append(cast(PluginType, instance))
            elif instance:
                logger.debug(f"Plugin '{plugin_id}' instantiated but not type {plugin_protocol_type.__name__}.")
        logger.info(f"Retrieved {len(instances)} plugin instances of type {plugin_protocol_type.__name__}.")
        return instances

    async def teardown_all_plugins(self) -> None:
        logger.info("Tearing down all instantiated plugins...")
        for plugin_id, instance in list(self._plugin_instances.items()):
            try:
                await instance.teardown()
                logger.debug(f"Plugin '{plugin_id}' torn down.")
            except Exception as e:
                logger.error(f"Error tearing down plugin '{plugin_id}': {e}", exc_info=True)
        self._plugin_instances.clear()
        logger.info("All plugin instances cleared after teardown.")

    def list_discovered_plugin_classes(self) -> Dict[str, Type[Plugin]]: return self._discovered_plugin_classes.copy()
    def get_plugin_source(self, plugin_id: str) -> Optional[str]: return self._plugin_source_map.get(plugin_id)