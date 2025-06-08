# Tutorial: Guardrails (E21)

This tutorial corresponds to the example file `examples/E21_guardrails_example.py`.

It demonstrates how to enforce safety policies on LLM inputs and outputs. It shows how to:
- Configure the built-in `KeywordBlocklistGuardrailPlugin`.
- See how input guardrails can block potentially harmful prompts before they reach the LLM.
- See how output guardrails can sanitize or block LLM responses that violate a policy.

## Example Code

--8<-- "examples/E21_guardrails_example.py"
