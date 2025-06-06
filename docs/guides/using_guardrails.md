# Using Guardrails

Guardrails in Genie Tooling provide a mechanism to enforce policies and safety checks on inputs, outputs, and tool usage attempts. They are implemented as plugins and integrated into the core `Genie` facade operations.

## Core Concepts

*   **`GuardrailManager`**: Orchestrates the execution of different types of guardrail plugins.
*   **Guardrail Plugin Types**:
    *   **`InputGuardrailPlugin`**: Checks data provided *to* the system or an LLM (e.g., user prompts, chat messages).
    *   **`OutputGuardrailPlugin`**: Checks data produced *by* the system or an LLM (e.g., LLM responses, tool execution results before final output).
    *   **`ToolUsageGuardrailPlugin`**: Checks if a specific tool usage attempt (tool + parameters) is permissible before execution.
*   **`GuardrailViolation` (TypedDict)**: The result of a guardrail check:
    ```python
    from typing import Literal, Optional, Dict, Any, TypedDict

    class GuardrailViolation(TypedDict):
        action: Literal["allow", "block", "warn"]
        reason: Optional[str]
        guardrail_id: Optional[str]
        details: Optional[Dict[str, Any]]
    ```
    *   `allow`: The check passed.
    *   `block`: The operation should be prevented.
    *   `warn`: The operation can proceed, but a warning should be logged or noted.

## Configuration

Guardrails are configured in `MiddlewareConfig`, primarily through `FeatureSettings` for enabling lists of guardrails, and `guardrail_configurations` for specific plugin settings.

**Via `FeatureSettings`:**

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        # ... other features ...
        input_guardrails=["keyword_blocklist_guardrail"], # Enable by alias
        output_guardrails=["keyword_blocklist_guardrail"],
        # tool_usage_guardrails=["my_custom_tool_usage_policy_v1"] # Example
    ),
    guardrail_configurations={
        "keyword_blocklist_guardrail_v1": { # Canonical ID
            "blocklist": ["unsafe_topic", "banned_phrase"],
            "case_sensitive": False,
            "action_on_match": "block" # or "warn"
        },
        # "my_custom_tool_usage_policy_v1": { ... }
    }
)
```

*   `features.input_guardrails`, `features.output_guardrails`, `features.tool_usage_guardrails`: Lists of plugin IDs or aliases for guardrails to activate for each category.
*   `guardrail_configurations`: A dictionary where keys are canonical guardrail plugin IDs (or aliases) and values are their specific configuration dictionaries.

## Implicit Integration

Guardrails are automatically invoked by relevant `Genie` facade methods:

*   **Input Guardrails**:
    *   Checked by `genie.llm.chat()` and `genie.llm.generate()` before sending data to the LLM.
    *   Checked by `genie.run_command()` on the user's command string before processing.
*   **Output Guardrails**:
    *   Checked by `genie.llm.chat()` and `genie.llm.generate()` on the LLM's response before returning it.
    *   Checked by `genie.execute_tool()` (via the invocation strategy) on the raw tool result before transformation.
*   **Tool Usage Guardrails**:
    *   Checked by `genie.execute_tool()` (via the invocation strategy) before the tool's `execute()` method is called.
    *   Checked by `genie.run_command()` after a tool and its parameters have been determined by the command processor, but before execution and before HITL (if HITL is also active).

**Behavior on "block":**
*   If an input guardrail blocks, the operation (e.g., LLM call) is prevented, and a `PermissionError` is typically raised by the `Genie` facade method.
*   If an output guardrail blocks, the original output is replaced with a message indicating it was blocked (e.g., `"[RESPONSE BLOCKED: Reason]"`).
*   If a tool usage guardrail blocks, the tool execution is prevented, and an error is typically returned by `genie.run_command()` or `genie.execute_tool()`.

## Built-in Guardrails

*   **`KeywordBlocklistGuardrailPlugin` (alias: `keyword_blocklist_guardrail`)**:
    *   Implements `InputGuardrailPlugin` and `OutputGuardrailPlugin`.
    *   Checks text data against a configurable list of keywords.
    *   Configuration:
        *   `blocklist` (List[str]): Keywords to block/warn on.
        *   `case_sensitive` (bool, default: `False`): Whether matching is case-sensitive.
        *   `action_on_match` (Literal["block", "warn"], default: `"block"`): Action to take if a keyword is found.

## Creating Custom Guardrail Plugins

1.  **Choose the Base Protocol**:
    *   `InputGuardrailPlugin`: Implement `async def check_input(self, data: Any, context: Optional[Dict[str, Any]]) -> GuardrailViolation`.
    *   `OutputGuardrailPlugin`: Implement `async def check_output(self, data: Any, context: Optional[Dict[str, Any]]) -> GuardrailViolation`.
    *   `ToolUsageGuardrailPlugin`: Implement `async def check_tool_usage(self, tool: Tool, params: Dict[str, Any], context: Optional[Dict[str, Any]]) -> GuardrailViolation`.
    A single plugin class can implement multiple of these protocols if it's designed to check different types of data.

2.  **Implement the Check Logic**: Your method should analyze the `data` (and `context` or `tool`/`params`) and return a `GuardrailViolation` dictionary.

3.  **Register Your Plugin**: Use entry points in `pyproject.toml` or place it in a directory specified by `plugin_dev_dirs` in `MiddlewareConfig`.

4.  **Configure It**: Add its ID to the appropriate list in `features` (e.g., `features.input_guardrails`) and provide any necessary configuration in `guardrail_configurations`.
