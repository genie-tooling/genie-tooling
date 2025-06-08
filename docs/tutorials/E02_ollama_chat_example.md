# Tutorial: Ollama Chat Example (E02)

This tutorial corresponds to the example file `examples/E02_ollama_chat_example.py`.

It demonstrates the simplest way to interact with a Large Language Model using Genie. It shows how to:
- Use `FeatureSettings` to quickly configure Genie to use an Ollama provider (`llm="ollama"`).
- Specify the model to use (e.g., `mistral:latest`).
- Use `genie.llm.chat()` to send a message to the LLM and receive a response.

**Prerequisite**: Ensure you have an Ollama instance running and the specified model pulled (e.g., `ollama pull mistral`).

## Example Code

--8<-- "examples/E02_ollama_chat_example.py"
