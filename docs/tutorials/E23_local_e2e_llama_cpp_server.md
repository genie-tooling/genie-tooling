# Tutorial: Llama.cpp Server E2E (E23)

This tutorial corresponds to the example file `examples/E23_local_e2e_llama_cpp_server.py`.

It provides a comprehensive end-to-end test using the Llama.cpp server provider. It covers most of Genie's features in a single, integrated script, including:
- LLM chat and generation with Pydantic parsing (GBNF).
- RAG indexing and search.
- Custom tool registration and execution.
- Command processing with HITL.
- Prompt management and conversation state.
- Guardrails and token usage tracking.
- A simple ReAct agent loop.

## Example Code

--8<-- "examples/E23_local_e2e_llama_cpp_server.py"
