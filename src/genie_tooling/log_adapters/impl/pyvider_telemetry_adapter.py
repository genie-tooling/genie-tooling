# src/genie_tooling/log_adapters/impl/pyvider_telemetry_adapter.py
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.log_adapters.abc import LogAdapter
from genie_tooling.redactors.abc import Redactor
from genie_tooling.redactors.impl.noop_redactor import NoOpRedactorPlugin
from genie_tooling.redactors.impl.schema_aware import (
    sanitize_data_with_schema_based_rules,
)

logger = logging.getLogger(__name__)

try:
    from pyvider.telemetry import (
        CustomDasEmojiSet,
        SemanticFieldDefinition,
        SemanticLayer,
    )
    from pyvider.telemetry import (
        LoggingConfig as PyviderLoggingConfig,
    )
    from pyvider.telemetry import TelemetryConfig as PyviderTelemetryConfig
    from pyvider.telemetry import logger as pyvider_global_logger
    from pyvider.telemetry import (
        setup_telemetry as pyvider_setup_telemetry,
    )
    from pyvider.telemetry import (
        shutdown_pyvider_telemetry as pyvider_shutdown_telemetry,
    )
    PYVIDER_AVAILABLE = True

    # --- Custom Semantic Layer Definition for Genie Tooling ---
    # MOVED INSIDE the try block to ensure imports succeeded first.

    GENIE_COMPONENT_EMOJIS = CustomDasEmojiSet(
        name="genie_component_emojis",
        emojis={
            "genie": "🧞", "agent": "🤖", "toolmanager": "🛠️", "ragmanager": "📚",
            "llmprovidermanager": "🧠", "observability": "🔭", "commandprocessor": "⚙️",
            "cache": "🗄️", "tool": "🔧", "default": "🧩",
        }
    )

    GENIE_ACTION_EMOJIS = CustomDasEmojiSet(
        name="genie_action_emojis",
        emojis={
            "invoke": "▶️", "execute": "💥", "search": "🔍", "index": "✍️",
            "generate": "✨", "chat": "💬", "plan": "🗺️", "setup": "🏗️",
            "teardown": "🔌", "trace": "👣", "redact": "🤫", "process": "🔄",
            "default": "➡️",
        }
    )

    GENIE_OUTCOME_EMOJIS = CustomDasEmojiSet(
        name="genie_outcome_emojis",
        emojis={
            "success": "✅", "error": "🔥", "warn": "⚠️", "start": "🏁",
            "end": "🏁", "hit": "🎯", "miss": "💨", "blocked": "🚫",
            "denied": "🚫", "default": "➡️",
        }
    )

    GENIE_SEMANTIC_LAYER = SemanticLayer(
        name="genie_tooling_layer",
        description="Semantic conventions for Genie Tooling operations.",
        priority=100,
        emoji_sets=[
            GENIE_COMPONENT_EMOJIS,
            GENIE_ACTION_EMOJIS,
            GENIE_OUTCOME_EMOJIS,
        ],
        field_definitions=[
            # The order here defines the emoji prefix order: [Component][Action][Outcome]
            SemanticFieldDefinition(log_key="domain", emoji_set_name="genie_component_emojis"),
            SemanticFieldDefinition(log_key="action", emoji_set_name="genie_action_emojis"),
            SemanticFieldDefinition(log_key="status", emoji_set_name="genie_outcome_emojis"),
        ]
    )
    # --- End of Semantic Layer Definition ---

except ImportError:
    PYVIDER_AVAILABLE = False
    PyviderLoggingConfig = None
    PyviderTelemetryConfig = None
    pyvider_global_logger = None
    pyvider_setup_telemetry = None
    pyvider_shutdown_telemetry = None
    CustomDasEmojiSet, SemanticFieldDefinition, SemanticLayer = (None, None, None) # type: ignore
    GENIE_SEMANTIC_LAYER = None # Ensure this is also None if import fails
    logger.warning(
        "PyviderTelemetryLogAdapter: 'pyvider-telemetry' library not installed. "
        "This adapter will not be functional. Please install it: poetry add --editable path/to/pyvider-telemetry"
    )


class PyviderTelemetryLogAdapter(LogAdapter, CorePluginType):
    plugin_id: str = "pyvider_telemetry_log_adapter_v1"
    description: str = "Log adapter using the pyvider-telemetry library for structured logging."

    _pyvider_logger: Optional[Any] = None
    _redactor: Optional[Redactor] = None
    _plugin_manager: Optional[PluginManager] = None
    _enable_schema_redaction: bool = False
    _enable_key_name_redaction: bool = False
    _is_setup_successful: bool = False

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not PYVIDER_AVAILABLE or not PyviderTelemetryConfig or not PyviderLoggingConfig or not pyvider_setup_telemetry:
            logger.error(f"{self.plugin_id}: Pyvider telemetry library not available. Cannot setup.")
            return

        cfg = config or {}
        self._plugin_manager = cfg.get("plugin_manager")

        if self._plugin_manager and isinstance(self._plugin_manager, PluginManager):
            redactor_id_to_load = cfg.get("redactor_plugin_id", NoOpRedactorPlugin.plugin_id)
            redactor_setup_config = cfg.get("redactor_config", {})
            redactor_instance_any = await self._plugin_manager.get_plugin_instance(redactor_id_to_load, config=redactor_setup_config)
            if redactor_instance_any and isinstance(redactor_instance_any, Redactor):
                self._redactor = cast(Redactor, redactor_instance_any)
            else:
                logger.warning(f"{self.plugin_id}: Redactor plugin '{redactor_id_to_load}' not found or invalid. Falling back to NoOpRedactor.")
                self._redactor = NoOpRedactorPlugin()
                await self._redactor.setup()
        else:
            logger.warning(f"{self.plugin_id}: PluginManager not provided. Using NoOpRedactor.")
            self._redactor = NoOpRedactorPlugin()
            await self._redactor.setup()

        self._enable_schema_redaction = bool(cfg.get("enable_schema_redaction", False))
        self._enable_key_name_redaction = bool(cfg.get("enable_key_name_redaction", False))
        logger.info(f"{self.plugin_id}: Schema redaction: {self._enable_schema_redaction}, Key name redaction: {self._enable_key_name_redaction}")

        # Pyvider Telemetry Configuration
        pyvider_service_name = cfg.get("service_name")
        pyvider_default_level = cfg.get("default_level", "DEBUG")
        pyvider_module_levels = cfg.get("module_levels", {})
        pyvider_console_formatter = cfg.get("console_formatter", "key_value")
        pyvider_logger_emoji = bool(cfg.get("logger_name_emoji_prefix_enabled", True))
        pyvider_das_emoji = bool(cfg.get("das_emoji_prefix_enabled", True))
        pyvider_omit_timestamp = bool(cfg.get("omit_timestamp", False))
        pyvider_globally_disabled = bool(cfg.get("globally_disabled", False))

        try:
            # Only add the layer if it was defined successfully
            custom_layers = [GENIE_SEMANTIC_LAYER] if GENIE_SEMANTIC_LAYER else []

            logging_config = PyviderLoggingConfig(
                default_level=pyvider_default_level,
                module_levels=pyvider_module_levels,
                console_formatter=pyvider_console_formatter,
                logger_name_emoji_prefix_enabled=pyvider_logger_emoji,
                das_emoji_prefix_enabled=pyvider_das_emoji,
                omit_timestamp=pyvider_omit_timestamp,
                custom_semantic_layers=custom_layers,
            )
            telemetry_config = PyviderTelemetryConfig(
                service_name=pyvider_service_name,
                logging=logging_config,
                globally_disabled=pyvider_globally_disabled,
            )
            pyvider_setup_telemetry(config=telemetry_config)
            self._pyvider_logger = pyvider_global_logger
            self._is_setup_successful = True
            logger.info(f"{self.plugin_id}: Pyvider telemetry setup complete. Service: {pyvider_service_name or 'pyvider-default'}.")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to setup Pyvider telemetry: {e}", exc_info=True)
            self._is_setup_successful = False

    async def process_event(self, event_type: str, data: Dict[str, Any], schema_for_data: Optional[Dict[str, Any]] = None) -> None:
        if not self._is_setup_successful or not self._pyvider_logger:
            logger.error(f"{self.plugin_id}: Not properly set up or Pyvider logger unavailable. Cannot process event: {event_type}")
            return

        sanitized_data = data
        if self._enable_schema_redaction:
            try:
                sanitized_data = sanitize_data_with_schema_based_rules(
                    sanitized_data, schema_for_data, self._enable_key_name_redaction
                )
            except Exception as e_schema_redact:
                logger.error(f"Error during schema-based redaction for event '{event_type}': {e_schema_redact}", exc_info=True)

        if self._redactor and not isinstance(self._redactor, NoOpRedactorPlugin):
            try:
                sanitized_data = self._redactor.sanitize(sanitized_data, schema_hints=schema_for_data)
            except Exception as e_custom_redact:
                logger.error(f"Error during custom redactor '{self._redactor.plugin_id}' for event '{event_type}': {e_custom_redact}", exc_info=True)

        pyvider_kwargs = sanitized_data.copy()

        # *** START OF FIX: Handle exc_info for tracebacks ***
        # Check if the event data includes the signal to log an exception traceback.
        # This allows other parts of the system to pass this information through the observability layer.
        should_log_exc_info = bool(pyvider_kwargs.pop("exc_info", False))
        # *** END OF FIX ***

        # --- Semantic Key Mapping ---
        component_val = pyvider_kwargs.get("component")
        if component_val:
            pyvider_kwargs["domain"] = str(component_val).lower().split(":")[0]

        event_parts = event_type.lower().split(".")
        if len(event_parts) > 1:
            pyvider_kwargs["action"] = event_parts[1]
        if len(event_parts) > 2:
            pyvider_kwargs["status"] = event_parts[2]

        if pyvider_kwargs.get("status_override"):
            pyvider_kwargs["action"] = str(pyvider_kwargs.pop("status_override")).lower()

        if pyvider_kwargs.get("error"):
            pyvider_kwargs["status"] = "error"

        # Determine Pyvider log level
        log_level_method = self._pyvider_logger.info

        explicit_log_level = str(pyvider_kwargs.pop("_log_level", "")).lower()
        if explicit_log_level == "error":
            log_level_method = self._pyvider_logger.error
        elif explicit_log_level == "warning":
            log_level_method = self._pyvider_logger.warning
        elif explicit_log_level == "debug":
            log_level_method = self._pyvider_logger.debug
        elif explicit_log_level == "trace":
            log_level_method = self._pyvider_logger.trace
        else:
            outcome = pyvider_kwargs.get("status")
            if "error" in event_type.lower() or outcome == "error":
                log_level_method = self._pyvider_logger.error
            elif "warn" in event_type.lower() or outcome == "warn":
                log_level_method = self._pyvider_logger.warning
            elif "debug" in event_type.lower() or "trace" in event_type.lower():
                log_level_method = self._pyvider_logger.debug
                if "trace" in event_type.lower():
                    log_level_method = self._pyvider_logger.trace

        try:
            # Pass the exc_info flag to the logger method
            log_level_method(event_type, exc_info=should_log_exc_info, **pyvider_kwargs)
        except Exception as e_log:
            logger.error(f"Error logging event '{event_type}' with Pyvider: {e_log}. Data: {str(pyvider_kwargs)[:500]}", exc_info=True)

    async def teardown(self) -> None:
        if self._redactor and hasattr(self._redactor, "teardown"):
            try:
                await self._redactor.teardown()
            except Exception as e_redact_td:
                logger.error(f"Error tearing down redactor '{self._redactor.plugin_id}': {e_redact_td}", exc_info=True)
        self._redactor = None
        if PYVIDER_AVAILABLE and pyvider_shutdown_telemetry:
            try:
                await pyvider_shutdown_telemetry()
            except Exception as e:
                logger.error(f"{self.plugin_id}: Error shutting down Pyvider telemetry: {e}", exc_info=True)
        self._pyvider_logger = None
        self._is_setup_successful = False
        logger.debug(f"{self.plugin_id}: Teardown complete.")
