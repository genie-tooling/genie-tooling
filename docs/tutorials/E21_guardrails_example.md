# Tutorial: Guardrails Example

This tutorial corresponds to the example file `examples/E21_guardrails_example.py`.

It demonstrates how to:
- Configure and enable `GuardrailPlugin`s (e.g., `KeywordBlocklistGuardrailPlugin`).
- See how input guardrails can block or warn on data sent to LLMs or command processors.
- See how output guardrails can block or warn on data received from LLMs or tools.
- Understand the `GuardrailViolation` structure and `action_on_match` behavior.

```python
# Full code from examples/E21_guardrails_example.py
```

**Key Takeaways:**
- Guardrails are implicitly integrated into core `Genie` operations.
- The `KeywordBlocklistGuardrailPlugin` offers a simple way to filter content.
- Guardrails can be configured to either "block" or "warn" upon detecting a violation.
