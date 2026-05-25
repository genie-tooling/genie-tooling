# Human-in-the-Loop (HITL) Approvals (`genie.human_in_loop`)

Genie Tooling supports Human-in-the-Loop (HITL) workflows, allowing critical actions or ambiguous decisions to be paused for human review and approval before proceeding. This is primarily managed via the `genie.human_in_loop` interface.

## Core Concepts

*   **`HITLInterface` (`genie.human_in_loop`)**: The facade interface for requesting human approval.
*   **`HumanApprovalRequestPlugin`**: A plugin responsible for presenting an approval request to a human and collecting their response.

    Built-in approvers (the *HITL ladder*, from least to most production-ready):

    | Plugin ID | Alias | When to use |
    |---|---|---|
    | `cli_approval_plugin_v1` | `cli_hitl_approver` | Local development only — prompts on the command line. |
    | `dev_auto_approve_hitl_v1` | `dev_auto_approve_hitl` | Tests and prototyping — auto-approves every request. Emits a loud error log if `MiddlewareConfig.environment == "production"`. (Old alias `auto_approve_hitl_v1` still works for one cycle with a deprecation warning.) |
    | `webhook_approval_v1` | — | Corporate standard — POSTs each request to a configured URL and expects `{status: "approved" \| "denied", ...}`. Plug a Slack interactive-message endpoint, Teams adaptive card, or JIRA/ServiceNow workflow behind it. **Safe-by-default: denies on timeout or HTTP error.** |
    | `policy_auto_approve_hitl_v1` | — | High-volume deployments — reads a YAML policy file; each decision is logged with the matching policy ID. Suitable when most decisions are deterministic ("admins can write, nobody else can") with humans in the loop only for the edge cases. |
*   **`ApprovalRequest` (TypedDict)**: Data structure for an approval request:
    ```python
    from typing import Literal, Optional, Dict, Any, TypedDict

    class ApprovalRequest(TypedDict):
        request_id: str
        prompt: str # Message shown to the human
        data_to_approve: Dict[str, Any] # Data needing approval
        context: Optional[Dict[str, Any]]
        timeout_seconds: Optional[int]
    ```
*   **`ApprovalResponse` (TypedDict)**: Data structure for the human's response:
    ```python
    class ApprovalResponse(TypedDict):
        request_id: str
        status: Literal["pending", "approved", "denied", "timeout", "error"]
        approver_id: Optional[str]
        reason: Optional[str] # Reason for denial or comments
        timestamp: Optional[float]
    ```

## Configuration

Configure the default HITL approval plugin via `FeatureSettings` or explicit `MiddlewareConfig`.

**Via `FeatureSettings`:**

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    environment="production",  # turns on the dev-approver guard rail
    features=FeatureSettings(
        # ... other features ...
        hitl_approver="cli_hitl_approver" # Use CLI for approvals
    ),
    # Example: Configure the CliApprovalPlugin (though it has no specific config by default)
    # hitl_approver_configurations={
    #     "cli_approval_plugin_v1": {} 
    # }
)
```

**Webhook approver** (corporate-standard configuration):

```python
app_config = MiddlewareConfig(
    environment="production",
    features=FeatureSettings(
        hitl_approver="webhook_approval_v1",
    ),
    hitl_approver_configurations={
        "webhook_approval_v1": {
            "webhook_url": "https://hooks.example.com/approve",
            "timeout_seconds": 300,
            # Optional auth header, polling settings, etc.
        }
    },
)
```

**Policy approver** (YAML-driven auto-approve):

```python
app_config = MiddlewareConfig(
    environment="production",
    features=FeatureSettings(
        hitl_approver="policy_auto_approve_hitl_v1",
    ),
    hitl_approver_configurations={
        "policy_auto_approve_hitl_v1": {
            "policy_file_path": "./hitl_policy.yml",
        }
    },
)
```

A minimal `hitl_policy.yml`:

```yaml
policies:
  - policy_id: "ALLOW_CALCULATOR"
    match:
      tool_id_glob: "calculator_tool*"
    action: "approve"
  - policy_id: "DEFAULT_DENY"
    match: {}
    action: "deny"
```

Every match is logged as an `auto_approved_by_policy` (or `auto_denied_by_policy`) event with the matching `policy_id` — same audit trail shape as a human-issued decision.

## Integration with `genie.run_command()`

The most common use case for HITL is to require approval before a tool selected by `genie.run_command()` is executed. If an `HITLManager` is active (i.e., `features.hitl_approver` is not `"none"`), `genie.run_command()` will automatically:
1.  Determine the tool and parameters.
2.  Create an `ApprovalRequest` with the tool ID and parameters.
3.  Call `genie.human_in_loop.request_approval()`.
4.  If approved, execute the tool. If denied or timed out, return an error/message.

**Example:**
If `hitl_approver` is set to `"cli_hitl_approver"` in `FeatureSettings` and `calculator_tool` is enabled in `tool_configurations`:
```python
# genie = await Genie.create(config=app_config_with_hitl)
# command_result = await genie.run_command("What is 15 times 7?") 

# The CLI will prompt:
# --- HUMAN APPROVAL REQUIRED ---
# Request ID: <uuid>
# Prompt: Approve execution of tool 'calculator_tool' with params: {'num1': 15, 'num2': 7, 'operation': 'multiply'} for goal 'What is 15 times 7?'?
# Data/Action: {'tool_id': 'calculator_tool', 'params': {'num1': 15, 'num2': 7, 'operation': 'multiply'}, 'step_reasoning': None}
# Approve? (yes/no/y/n): 
```
If the user types "no" and provides a reason, `command_result` might look like:
```json
{
  "error": "Tool execution denied by HITL: User intervention required.",
  "thought_process": "The LLM selected calculator_tool for 15 times 7.",
  "hitl_decision": {
    "request_id": "...", "status": "denied", "approver_id": "cli_user", 
    "reason": "User intervention required.", "timestamp": ...
  }
}
```

## Manual Approval Requests with `genie.human_in_loop.request_approval()`

You can also manually trigger an approval request from your custom logic:

```python
from genie_tooling.hitl.types import ApprovalRequest
import uuid

request_details = ApprovalRequest(
    request_id=str(uuid.uuid4()),
    prompt="Is it okay to proceed with this sensitive operation on customer data?",
    data_to_approve={"customer_id": "cust_789", "operation": "data_wipe"},
    timeout_seconds=300 # 5 minutes
)

approval_response = await genie.human_in_loop.request_approval(request_details)
# approval_response = await genie.human_in_loop.request_approval(request_details, approver_id="my_specific_hitl_plugin")


if approval_response["status"] == "approved":
    print(f"Operation approved by {approval_response.get('approver_id', 'N/A')}. Reason: {approval_response.get('reason')}")
    # ... proceed with sensitive operation ...
else:
    print(f"Operation {approval_response['status']}. Reason: {approval_response.get('reason')}")
```

## Per-Action HITL Gate on `ReActAgent`

`PlanAndExecuteAgent` has always gated each plan step on HITL approval. As
of 0.2.0, `ReActAgent` supports the same — every `execute_tool` call
inside the loop can be gated on approval:

```python
from genie_tooling.agents.react_agent import ReActAgent

agent = ReActAgent(
    genie=genie,
    agent_config={
        "hitl_per_action": True,       # gate every tool call
        # "use_native_tool_use": True, # honored on both regex and native loops
    },
)
result = await agent.run(goal="Look up the weather in Berlin and convert it to Fahrenheit.")
```

If the approver denies a tool call, the agent observes the denial in the
scratchpad and decides on the next iteration whether to retry, pick a
different tool, or terminate with `status="user_stopped"`.

## Production Environment Guard

Setting `MiddlewareConfig.environment="production"` makes the framework
log a loud error if the resolved HITL approver is the dev-mode one
(`dev_auto_approve_hitl_v1`). This is intended to catch accidental
deployments where the auto-approver was left wired in from CI.

## Creating Custom HITL Approval Plugins

Implement the `HumanApprovalRequestPlugin` protocol, defining the `request_approval(request: ApprovalRequest)` method. This method should handle the presentation of the request to a human (e.g., via a web UI, messaging platform, ticketing system) and return an `ApprovalResponse`. Register your plugin via entry points or `plugin_dev_dirs`.
