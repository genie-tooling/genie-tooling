# Tutorial: OpenAI Chat Example (E03)

This tutorial corresponds to the example file `examples/E03_openai_chat_example.py`.

It demonstrates how to configure Genie to use the OpenAI API. It shows how to:
- Use `FeatureSettings` to select the OpenAI provider (`llm="openai"`).
- Specify the model to use (e.g., `gpt-3.5-turbo`).
- Rely on the default `EnvironmentKeyProvider` to automatically pick up the `OPENAI_API_KEY` from your environment variables.

**Prerequisite**: You must have your `OPENAI_API_KEY` set as an environment variable.

## Example Code

--8<-- "examples/E03_openai_chat_example.py"
