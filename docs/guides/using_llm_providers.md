# Using LLM Providers

Genie Tooling allows you to easily interact with various Large Language Models (LLMs) through a unified interface: `genie.llm`. This interface abstracts the specifics of different LLM provider APIs.

## The `genie.llm` Interface

Once you have a `Genie` instance, you can access LLM functionalities:

*   **`async genie.llm.generate(prompt: str, provider_id: Optional[str] = None, **kwargs) -> LLMCompletionResponse`**:
    For text completion or generation tasks.
    *   `prompt`: The input prompt string.
    *   `provider_id`: Optional. The ID of the LLM provider plugin to use. If `None`, the default LLM provider configured in `MiddlewareConfig` (via `features.llm` or `default_llm_provider_id`) is used.
    *   `**kwargs`: Additional provider-specific parameters (e.g., `temperature`, `max_tokens`, `model` to override the default for that provider).
*   **`async genie.llm.chat(messages: List[ChatMessage], provider_id: Optional[str] = None, **kwargs) -> LLMChatResponse`**:
    For conversational interactions.
    *   `messages`: A list of `ChatMessage` dictionaries (see [Types](#chatmessage-type)).
    *   `provider_id`: Optional. The ID of the LLM provider plugin to use.
    *   `**kwargs`: Additional provider-specific parameters (e.g., `temperature`, `tools`, `tool_choice`).

## Configuring LLM Providers

LLM providers are primarily configured using `FeatureSettings` in your `MiddlewareConfig`.

### Example: Using Ollama

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import ChatMessage
import asyncio

async def main():
    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",  # Select Ollama as the default
            llm_ollama_model_name="mistral:latest" # Specify the model for Ollama
        )
    )
    genie = await Genie.create(config=app_config)

    response = await genie.llm.chat([{"role": "user", "content": "Hi from Genie!"}])
    print(response['message']['content'])

    await genie.close()

if __name__ == "__main__":
    asyncio.run(main())
```
Ensure Ollama is running (`ollama serve`) and the specified model (`mistral:latest`) is pulled (`ollama pull mistral`).

### Example: Using OpenAI

```python
# Requires OPENAI_API_KEY environment variable to be set.
# Genie's default EnvironmentKeyProvider will pick it up.

app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="openai",
        llm_openai_model_name="gpt-3.5-turbo"
    )
)
# genie = await Genie.create(config=app_config)
# ... use genie.llm.chat or genie.llm.generate ...
```

### Example: Using Gemini

```python
# Requires GOOGLE_API_KEY environment variable to be set.

app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="gemini",
        llm_gemini_model_name="gemini-1.5-flash-latest"
    )
)
# genie = await Genie.create(config=app_config)
# ... use genie.llm.chat or genie.llm.generate ...
```

### Overriding Provider Settings

You can override settings for specific LLM providers using the `llm_provider_configurations` dictionary in `MiddlewareConfig`. Keys can be the canonical plugin ID or a recognized alias.

```python
app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="openai", # Default is OpenAI
        llm_openai_model_name="gpt-3.5-turbo" # Default model for OpenAI
    ),
    llm_provider_configurations={
        "openai_llm_provider_v1": { # Canonical ID for OpenAI provider
            "model_name": "gpt-4-turbo-preview", # Override the model for OpenAI
            "request_timeout_seconds": 180 
        },
        "ollama": { # Alias for Ollama provider
            "model_name": "llama3:latest",
            "request_timeout_seconds": 240
        }
    }
)
genie = await Genie.create(config=app_config)

# This will use OpenAI with gpt-4-turbo-preview
await genie.llm.chat([{"role": "user", "content": "Hello OpenAI!"}])

# This will use Ollama with llama3:latest
await genie.llm.chat([{"role": "user", "content": "Hello Ollama!"}], provider_id="ollama")
```

### API Keys and `KeyProvider`

LLM providers that require API keys (like OpenAI and Gemini) will attempt to fetch them using the configured `KeyProvider`. By default, Genie uses `EnvironmentKeyProvider`, which reads keys from environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`). You can provide a custom `KeyProvider` instance to `Genie.create()` for more sophisticated key management. See the [Configuration Guide](configuration.md) for details.

## `ChatMessage` Type

The `messages` parameter for `genie.llm.chat()` expects a list of `ChatMessage` dictionaries:

```python
from genie_tooling.llm_providers.types import ChatMessage, ToolCall

# User message
user_message: ChatMessage = {"role": "user", "content": "What's the weather in London?"}

# Assistant message (simple text response)
assistant_text_response: ChatMessage = {"role": "assistant", "content": "The weather in London is pleasant."}

# Assistant message requesting a tool call
assistant_tool_call_request: ChatMessage = {
    "role": "assistant",
    "content": None, # Content can be None if only tool_calls are present
    "tool_calls": [
        {
            "id": "call_weather_london_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "London", "units": "celsius"}'}
        }
    ]
}

# Tool message (response from executing a tool)
tool_response_message: ChatMessage = {
    "role": "tool",
    "tool_call_id": "call_weather_london_123", # Matches the ID from assistant's request
    "name": "get_weather", # Name of the function that was called
    "content": '{"temperature": 15, "condition": "Cloudy"}' # JSON string of the tool's output
}
```

The `LLMChatResponse` from `genie.llm.chat()` will contain an assistant's message in this format.
