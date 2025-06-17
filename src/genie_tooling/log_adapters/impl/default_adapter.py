import json
import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.log_adapters.abc import LogAdapter
from genie_tooling.redactors.abc import Redactor
from genie_tooling.redactors.impl.noop_redactor import (
    NoOpRedactorPlugin,
)
from genie_tooling.redactors.impl.schema_aware import (
    sanitize_data_with_schema_based_rules,
)

logger = logging.getLogger(__name__)
DEFAULT_LIBRARY_LOGGER_NAME = "genie_tooling"

class DefaultLogAdapter(LogAdapter):
    plugin_id: str = "default_log_adapter_v1"
    description: str = "Configures standard Python logging for the library and processes events with redaction."

    _library_logger: Optional[logging.Logger] = None
    _redactor: Optional[Redactor] = None
    _plugin_manager: Optional[PluginManager] = None
    _enable_schema_redaction: bool = False
    _enable_key_name_redaction: bool = False

    async def setup(self, config: Dict[str, Any]) -> None:
        cfg = config or {}
        log_level_str = cfg.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        library_logger_name = cfg.get("library_logger_name", DEFAULT_LIBRARY_LOGGER_NAME)
        self._library_logger = logging.getLogger(library_logger_name)
        add_console_handler = cfg.get("add_console_handler_if_no_handlers", True)
        if add_console_handler and not self._library_logger.handlers:
            console_h = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(module)s:%(lineno)d)")
            console_h.setFormatter(formatter)
            self._library_logger.addHandler(console_h)
            self._library_logger.propagate = False
            logger.debug(f"Added default console handler to logger '{library_logger_name}'.")
        self._library_logger.setLevel(log_level)
        logger.info(f"{self.plugin_id}: Logging configured for '{library_logger_name}' at level {log_level_str}.")

        self._plugin_manager = cfg.get("plugin_manager")
        if not self._plugin_manager or not isinstance(self._plugin_manager, PluginManager):
            logger.warning(f"{self.plugin_id}: PluginManager not provided in config. Custom Redactor plugin cannot be loaded. Using NoOpRedactor.")
            self._redactor = NoOpRedactorPlugin()
            await self._redactor.setup()
        else:
            redactor_id_to_load = cfg.get("redactor_plugin_id", NoOpRedactorPlugin.plugin_id) # type: ignore
            redactor_setup_config = cfg.get("redactor_config", {}).copy()
            redactor_setup_config.setdefault("plugin_manager", self._plugin_manager)
            redactor_instance_any = await self._plugin_manager.get_plugin_instance(redactor_id_to_load, config=redactor_setup_config)
            if redactor_instance_any and isinstance(redactor_instance_any, Redactor):
                self._redactor = cast(Redactor, redactor_instance_any)
                logger.info(f"{self.plugin_id}: Using Redactor plugin '{redactor_id_to_load}'.")
            else:
                logger.warning(f"{self.plugin_id}: Redactor plugin '{redactor_id_to_load}' not found or invalid. Falling back to NoOpRedactor.")
                self._redactor = NoOpRedactorPlugin()
                await self._redactor.setup()
        self._enable_schema_redaction = bool(cfg.get("enable_schema_redaction", False))
        self._enable_key_name_redaction = bool(cfg.get("enable_key_name_redaction", False))
        logger.info(f"{self.plugin_id}: Schema redaction: {self._enable_schema_redaction}, Key name redaction: {self._enable_key_name_redaction}")


    async def process_event(self, event_type: str, data: Dict[str, Any], schema_for_data: Optional[Dict[str, Any]] = None) -> None:
        if not self._library_logger:
            logger.debug(f"EMERGENCY LOG (logger not init): EVENT: {event_type} | RAW_DATA: {str(data)[:500]}...")
            return
        sanitized_data_for_log = data
        if self._enable_schema_redaction:
            try:
                sanitized_data_for_log = sanitize_data_with_schema_based_rules(
                    sanitized_data_for_log,
                    schema=schema_for_data,
                    redact_matching_key_names=self._enable_key_name_redaction
                )
                logger.debug(f"Event '{event_type}' data after schema-based redaction (first pass).")
            except Exception as e_schema_redact:
                logger.error(f"Error during schema-based redaction for event '{event_type}': {e_schema_redact}", exc_info=True)

        if self._redactor and not isinstance(self._redactor, NoOpRedactorPlugin):
            try:

                sanitized_data_for_log = self._redactor.sanitize(sanitized_data_for_log, schema_hints=schema_for_data)
                logger.debug(f"Event '{event_type}' data after custom Redactor plugin '{self._redactor.plugin_id}'.")
            except Exception as e_custom_redact:
                logger.error(f"Error during custom Redactor plugin '{self._redactor.plugin_id}' for event '{event_type}': {e_custom_redact}", exc_info=True)
        try:
            log_data_str = json.dumps(sanitized_data_for_log, sort_keys=True, default=str)
            log_data_str = log_data_str[:2000] + "..." if len(log_data_str) > 2000 else log_data_str
        except Exception:
            log_data_str = str(sanitized_data_for_log)[:2000] + "..."
        self._library_logger.info(f"EVENT: {event_type} | DATA: {log_data_str}")

    async def teardown(self) -> None:
        logger.info(f"{self.plugin_id}: Tearing down.")
        if self._redactor and hasattr(self._redactor, "teardown"):
            try:
                await self._redactor.teardown()
            except Exception as e_redact_td:
                logger.error(f"Error tearing down redactor '{self._redactor.plugin_id}': {e_redact_td}", exc_info=True)
        self._library_logger = None
        self._redactor = None
        self._plugin_manager = None
