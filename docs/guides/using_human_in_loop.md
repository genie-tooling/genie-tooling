# Human-in-the-Loop (HITL) Approvals (`genie.human_in_loop`)

Genie Tooling supports Human-in-the-Loop (HITL) workflows, allowing critical actions or ambiguous decisions to be paused for human review and approval before proceeding. This is primarily managed via the `genie.human_in_loop` interface.

## Core Concepts

*   **`HITLInterface` (`genie.human_in_loop`)**: The facade interface for requesting human approval.
*   **`HumanApprovalRequestPlugin`**: A plugin responsible for presenting an approval request to a human and collecting their response.
    *   Built-in: `CliApprovalPlugin` (alias: `cli_hitl_approver`) - Prompts on the command line.
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

## Creating Custom HITL Approval Plugins

Implement the `HumanApprovalRequestPlugin` protocol, defining the `request_approval(request: ApprovalRequest)` method. This method should handle the presentation of the request to a human (e.g., via a web UI, messaging platform, ticketing system) and return an `ApprovalResponse`. Register your plugin via entry points or `plugin_dev_dirs`.
