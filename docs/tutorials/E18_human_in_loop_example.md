# Tutorial: Human-in-the-Loop (E18)

This tutorial corresponds to the example file `examples/E18_human_in_loop_example.py`.

It demonstrates how to add a human approval step before executing critical actions. It shows how to:
- Configure a `HumanApprovalRequestPlugin` (e.g., `cli_hitl_approver`).
- See how `genie.run_command()` automatically triggers the HITL flow before executing a tool.
- Understand the approval/denial workflow from the user's perspective.

## Example Code

--8<-- "examples/E18_human_in_loop_example.py"
