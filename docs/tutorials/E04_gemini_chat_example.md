# Tutorial: Gemini Chat Example (E04)

This tutorial corresponds to the example file `examples/E04_gemini_chat_example.py`.

It demonstrates how to configure Genie to use the Google Gemini API. It shows how to:
- Use `FeatureSettings` to select the Gemini provider (`llm="gemini"`).
- Specify the model to use (e.g., `gemini-1.5-flash-latest`).
- Rely on the default `EnvironmentKeyProvider` to automatically pick up the `GOOGLE_API_KEY` from your environment variables.

**Prerequisite**: You must have your `GOOGLE_API_KEY` set as an environment variable.

## Example Code

--8<-- "examples/E04_gemini_chat_example.py"
