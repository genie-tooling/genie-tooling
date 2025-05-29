"""DefaultLogAdapter: Basic logging setup using standard Python logging and pluggable redaction."""
import json
import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
# Updated import path for LogAdapter
from genie_tooling.log_adapters.abc import LogAdapter
# Updated import path for Redactor and its implementations
from genie_tooling.redactors.abc import Redactor
from genie_tooling.redactors.impl.noop_redactor import (
    NoOpRedactorPlugin,  # Default redactor
)
from genie_tooling.redactors.impl.schema_aware import (
    sanitize_data_with_schema_based_rules,
)

logger = logging.getLogger(__name__) # Logger for this adapter's own messages

DEFAULT_LIBRARY_LOGGER_NAME = "genie_tooling" # Logger name used by the library components

class DefaultLogAdapter(LogAdapter):
    plugin_id: str = "default_log_adapter_v1"
    description: str = "Configures standard Python logging for the library and processes events with redaction."

    _library_logger: Optional[logging.Logger] = None
    _redactor: Optional[Redactor] = None
    _plugin_manager: Optional[PluginManager] = None # Needed to load configured redactor

    async def setup_logging(self, config: Dict[str, Any]) -> None:
        """
        Configures logging for the library.
        Args:
            config: Configuration dictionary. Expected keys:
                "log_level": str (e.g., "INFO", "DEBUG", default: "INFO")
                "library_logger_name": str (default: "genie_tooling")
                "add_console_handler_if_no_handlers": bool (default: True)
                "plugin_manager": PluginManager instance (Required to load redactor)
                "redactor_plugin_id": Optional[str] (ID of Redactor plugin to use, default: NoOpRedactorPlugin.plugin_id)
                "redactor_config": Optional[Dict[str,Any]] (Config for redactor's setup)
                "enable_schema_redaction": bool (default: True) - Whether to use built-in schema redaction.
                "enable_key_name_redaction": bool (default: True) - Whether schema redaction should also check key names.
        """
        cfg = config or {}
        log_level_str = cfg.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        library_logger_name = cfg.get("library_logger_name", DEFAULT_LIBRARY_LOGGER_NAME)
        self._library_logger = logging.getLogger(library_logger_name)

        add_console_handler = cfg.get("add_console_handler_if_no_handlers", True)
        if add_console_handler and not self._library_logger.hasHandlers():
            console_h = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(module)s:%(lineno)d)")
            console_h.setFormatter(formatter)
            self._library_logger.addHandler(console_h)
            self._library_logger.propagate = False # Avoid duplicate messages if root logger also has handler
            logger.debug(f"Added default console handler to logger '{library_logger_name}'.")

        self._library_logger.setLevel(log_level)
        logger.info(f"{self.plugin_id}: Logging configured for '{library_logger_name}' at level {log_level_str}.")

        # Load Redactor Plugin
        self._plugin_manager = cfg.get("plugin_manager")
        if not self._plugin_manager or not isinstance(self._plugin_manager, PluginManager):
            logger.warning(f"{self.plugin_id}: PluginManager not provided in config. Custom Redactor plugin cannot be loaded. Using NoOpRedactor.")
            self._redactor = NoOpRedactorPlugin() # Fallback
            await self._redactor.setup() # Call setup for default
        else:
            redactor_id_to_load = cfg.get("redactor_plugin_id", NoOpRedactorPlugin.plugin_id) # type: ignore
            redactor_setup_config = cfg.get("redactor_config", {})

            redactor_instance_any = await self._plugin_manager.get_plugin_instance(redactor_id_to_load, config=redactor_setup_config)
            if redactor_instance_any and isinstance(redactor_instance_any, Redactor):
                self._redactor = cast(Redactor, redactor_instance_any)
                logger.info(f"{self.plugin_id}: Using Redactor plugin '{redactor_id_to_load}'.")
            else:
                logger.warning(f"{self.plugin_id}: Redactor plugin '{redactor_id_to_load}' not found or invalid. Falling back to NoOpRedactor.")
                self._redactor = NoOpRedactorPlugin()
                await self._redactor.setup() # Call setup for fallback

        self._enable_schema_redaction = bool(cfg.get("enable_schema_redaction", True))
        self._enable_key_name_redaction = bool(cfg.get("enable_key_name_redaction", True))


    async def process_event(self, event_type: str, data: Dict[str, Any], schema_for_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Logs the event after sanitizing the data.
        Sanitization steps:
        1. Built-in schema-aware redaction (if enabled).
        2. Custom RedactorPlugin (if configured and different from NoOp).
        """
        if not self._library_logger:
            # Should not happen if setup_logging was called and successful
            logger.debug(f"EMERGENCY LOG (logger not init): EVENT: {event_type} | RAW_DATA: {str(data)[:500]}...") # Fallback logger.debug
            return

        sanitized_data_for_log = data # Start with original data

        # Step 1: Built-in schema-aware redaction (if enabled)
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
                # Continue with potentially unredacted or partially redacted data, but log the error.

        # Step 2: Custom RedactorPlugin (if configured and not NoOp)
        if self._redactor and not isinstance(self._redactor, NoOpRedactorPlugin): # Only call if it's a custom one
            try:
                sanitized_data_for_log = self._redactor.sanitize(sanitized_data_for_log, schema_hints=schema_for_data)
                logger.debug(f"Event '{event_type}' data after custom Redactor plugin '{self._redactor.plugin_id}'.")
            except Exception as e_custom_redact:
                logger.error(f"Error during custom Redactor plugin '{self._redactor.plugin_id}' for event '{event_type}': {e_custom_redact}", exc_info=True)
                # Data remains as it was after schema redaction (if any).

        # Log the event using the library's configured logger
        # Convert dict to string for logging; a structured logger (e.g., python-json-logger) would be better for machine parsing.
        try:
            # Truncate very long data for basic logging to avoid flooding.
            # A proper structured logging setup would handle this better.
            log_data_str = json.dumps(sanitized_data_for_log, sort_keys=True, default=str) # default=str for non-serializable
            if len(log_data_str) > 2000: # Arbitrary limit for log message
                log_data_str = log_data_str[:2000] + "..."
        except Exception:
            log_data_str = str(sanitized_data_for_log)[:2000] + "..." # Fallback to str()

        self._library_logger.info(f"EVENT: {event_type} | DATA: {log_data_str}")

    async def teardown(self) -> None:
        """Cleans up logging resources, if any were specifically managed by this adapter."""
        logger.info(f"{self.plugin_id}: Tearing down.")
        if self._redactor and hasattr(self._redactor, "teardown"):
            try:
                await self._redactor.teardown()
            except Exception as e_redact_td:
                logger.error(f"Error tearing down redactor '{self._redactor.plugin_id}': {e_redact_td}", exc_info=True)

        # If this adapter added handlers to the library_logger, it might remove them here.
        # However, typical library logging practice is to let the application manage handlers.
        # For the default handler added in setup, it will persist unless explicitly removed.
        # For simplicity, we don't remove handlers here, assuming app might want them.
        self._library_logger = None
        self._redactor = None
        self._plugin_manager = None
