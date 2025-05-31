# Tutorial: Human-in-the-Loop (HITL) Example

This tutorial corresponds to the example file `examples/E18_human_in_loop_example.py`.

It demonstrates how to:
- Configure a `HumanApprovalRequestPlugin` (e.g., `CliApprovalPlugin`).
- See HITL automatically triggered by `genie.run_command()` before tool execution.
- Understand the structure of `ApprovalRequest` and `ApprovalResponse`.

```python
# Full code from examples/E18_human_in_loop_example.py
```

**Key Takeaways:**
- HITL provides a checkpoint for critical or ambiguous agent actions.
- The `CliApprovalPlugin` allows for simple console-based approvals during development.
- `genie.run_command()` integrates HITL seamlessly if an approver is configured.
