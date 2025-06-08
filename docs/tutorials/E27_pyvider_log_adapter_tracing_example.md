# Tutorial: Pyvider Log Adapter (E27)

This tutorial corresponds to the example file `examples/E27_pyvider_log_adapter_tracing_example.py`.

It demonstrates how to integrate Genie's eventing system with the `pyvider-telemetry` library for advanced structured logging. It shows how to:
- Configure Genie to use the `PyviderTelemetryLogAdapter`.
- Use the `ConsoleTracerPlugin`, which then delegates its output formatting to the configured Pyvider adapter.
- Observe how trace events are formatted by Pyvider (e.g., key-value or JSON, potentially with emojis).

## Example Code

--8<-- "examples/E27_pyvider_log_adapter_tracing_example.py"
