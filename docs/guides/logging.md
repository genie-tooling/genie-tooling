# Logging

Genie Tooling uses a pluggable `LogAdapterPlugin` system for its internal structured event logging. This allows application developers to integrate Genie's detailed operational events into their existing logging infrastructure or use specialized logging libraries. The `InteractionTracingManager` (and by extension, tracers like `ConsoleTracerPlugin`) relies on the configured `LogAdapterPlugin` to process and output trace events.

## Core Concepts

*   **`LogAdapterPlugin` (`genie_tooling.log_adapters.abc.LogAdapter`)**: The protocol for plugins that handle structured log events. They are responsible for final formatting, redaction, and output.
*   **`DefaultLogAdapter` (alias: `default_log_adapter`)**: The standard built-in adapter. It uses Python's `logging` module and can integrate a `RedactorPlugin` for sanitizing data.
*   **`PyviderTelemetryLogAdapter` (alias: `pyvider_log_adapter`)**: An adapter that integrates with the `pyvider-telemetry` library for advanced structured logging, including emoji-based DAS (Domain-Action-Status) logging.
*   **Library Logger Name**: The root logger name for messages originating directly from Genie Tooling library components (when not processed through a `LogAdapterPlugin`) is `genie_tooling`.

## Configuring the Log Adapter

You select and configure the primary `LogAdapterPlugin` using `FeatureSettings` or explicit configurations in `MiddlewareConfig`.

**1. Via `FeatureSettings`:**

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        # ... other features ...
        logging_adapter="default_log_adapter", # Default
        # OR
        # logging_adapter="pyvider_log_adapter",
        # logging_pyvider_service_name="my-genie-app-with-pyvider" # Optional for Pyvider
    ),
    # Configuration for the chosen adapter goes into log_adapter_configurations
    log_adapter_configurations={
        "default_log_adapter_v1": { # If default_log_adapter is chosen
            "log_level": "DEBUG", # For the 'genie_tooling' logger
            "library_logger_name": "genie_tooling_app_logs", # Custom name for library logs
            "redactor_plugin_id": "schema_aware_redactor_v1", # Example custom redactor
            "enable_schema_redaction": True,
            "enable_key_name_redaction": True
        },
        "pyvider_telemetry_log_adapter_v1": { # If pyvider_log_adapter is chosen
            "service_name": "MyGenieService", # Overrides feature setting if both present
            "default_level": "INFO", # For Pyvider's default logger
            "module_levels": {"genie_tooling.llm_providers": "DEBUG"},
            "console_formatter": "json",
            "redactor_plugin_id": "noop_redactor_v1" # Pyvider might have its own redaction
        }
    }
)
```

**2. Explicit Configuration:**

```python
app_config = MiddlewareConfig(
    default_log_adapter_id="pyvider_telemetry_log_adapter_v1", # Explicitly set
    log_adapter_configurations={
        "pyvider_telemetry_log_adapter_v1": {
            "service_name": "MyExplicitPyviderService",
            "default_level": "TRACE",
            # ... other Pyvider specific settings ...
            # "redactor_plugin_id": "my_custom_redactor_for_pyvider_v1"
        }
    }
    # Ensure plugin_manager is passed to Genie.create() if the adapter needs it
    # (e.g., to load its own redactor). Genie.create now handles injecting
    # the main PluginManager into the LogAdapter's config.
)
```

The `Genie.create()` method will instantiate the configured `LogAdapterPlugin` (passing the main `PluginManager` in its configuration, allowing it to load its own redactor) and then pass this `LogAdapterPlugin` instance to the `InteractionTracingManager`. Tracers like `ConsoleTracerPlugin` will then use this central `LogAdapterPlugin` to process their trace events.

## Redaction

Both `DefaultLogAdapter` and the new `PyviderTelemetryLogAdapter` (as per the plan) will integrate with a `RedactorPlugin` to sanitize sensitive data before logging.
*   The `DefaultLogAdapter` has built-in schema-aware redaction which can be toggled, and it also uses the configured `RedactorPlugin`.
*   The `PyviderTelemetryLogAdapter` will also use a configured `RedactorPlugin` for a first pass of redaction before handing data to Pyvider's logging methods.

## Standard Python Logging

If you need to configure logging for Genie Tooling library modules that log *directly* (not through the `LogAdapterPlugin` event system), you can still use the standard Python `logging` module:

```python
import logging

# Get the Genie Tooling library logger
library_logger = logging.getLogger("genie_tooling")
library_logger.setLevel(logging.DEBUG) 
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(module)s:%(lineno)d)')
console_handler.setFormatter(formatter)
if not any(isinstance(h, logging.StreamHandler) for h in library_logger.handlers):
    library_logger.addHandler(console_handler)
```
This standard setup is separate from the `LogAdapterPlugin` system, which is for structured event processing. The `DefaultLogAdapter` bridges these by configuring the `genie_tooling` logger as part of its setup.
