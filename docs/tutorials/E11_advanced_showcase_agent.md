# Tutorial: Advanced Showcase (E11)

This tutorial corresponds to the example file `examples/E11_advanced_showcase_agent.py`.

It demonstrates a more complex configuration, combining multiple features and plugins. It shows how to:
- Configure multiple LLM providers (`ollama`, `openai`, `gemini`) and select them at runtime.
- Configure multiple command processors (`llm_assisted`, `simple_keyword`) and select them at runtime.
- Use a custom `KeyProvider` for fetching API keys.
- Combine RAG, command processing, and direct tool execution in a single application.

## Example Code

--8<-- "examples/E11_advanced_showcase_agent.py"
