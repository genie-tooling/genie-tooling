# Using LLM Providers

Genie Tooling allows you to easily interact with various Large Language Models (LLMs) through a unified interface: `genie.llm`. This interface abstracts the specifics of different LLM provider APIs.

## The `genie.llm` Interface

Once you have a `Genie` instance, you can access LLM functionalities:

*   **`async genie.llm.generate(prompt: str, provider_id: Optional[str] = None, stream: bool = False, **kwargs) -> Union[LLMCompletionResponse, AsyncIterable[LLMCompletionChunk]]`**:
    For text completion or generation tasks.
    *   `prompt`: The input prompt string.
    *   `provider_id`: Optional. The ID of the LLM provider plugin to use. If `None`, the default LLM provider configured in `MiddlewareConfig` (via `features.llm` or `default_llm_provider_id`) is used.
    *   `stream`: Optional (default `False`). If `True`, returns an async iterable of `LLMCompletionChunk` objects.
    *   `**kwargs`: Additional provider-specific parameters (e.g., `temperature`, `max_tokens`, `model` to override the default for that provider).
*   **`async genie.llm.chat(messages: List[ChatMessage], provider_id: Optional[str] = None, stream: bool = False, **kwargs) -> Union[LLMChatResponse, AsyncIterable[LLMChatChunk]]`**:
    For conversational interactions.
    *   `messages`: A list of `ChatMessage` dictionaries (see [Types](#chatmessage-type)).
    *   `provider_id`: Optional. The ID of the LLM provider plugin to use.
    *   `stream`: Optional (default `False`). If `True`, returns an async iterable of `LLMChatChunk` objects.
    *   `**kwargs`: Additional provider-specific parameters (e.g., `temperature`, `tools`, `tool_choice`).
*   **`async genie.llm.parse_output(response: Union[LLMChatResponse, LLMCompletionResponse], parser_id: Optional[str] = None, schema: Optional[Any] = None) -> ParsedOutput`**:
    Parses the text content from an LLM response (either `LLMChatResponse` or `LLMCompletionResponse`) using a configured `LLMOutputParserPlugin`.
    *   `response`: The LLM response object.
    *   `parser_id`: Optional. The ID of the `LLMOutputParserPlugin` to use. If `None`, the default configured parser is used.
    *   `schema`: Optional. A schema (e.g., Pydantic model class, JSON schema dict) to guide parsing, if supported by the parser.
    *   Returns the parsed data (e.g., a dictionary, a Pydantic model instance).
    *   Raises `ValueError` if parsing fails or content cannot be extracted.

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

## Parsing LLM Output

Often, you'll want an LLM to produce structured output (e.g., JSON). The `genie.llm.parse_output()` method helps with this.

**Example: Parsing JSON output**
```python
# Assuming 'genie' is initialized and an LLM provider is configured.
# And a JSONOutputParserPlugin (json_output_parser_v1) is available.

# Configure default output parser (optional, can also specify per call)
# app_config = MiddlewareConfig(
#     features=FeatureSettings(llm="ollama", default_llm_output_parser="json_output_parser")
# )
# genie = await Genie.create(config=app_config)


prompt_for_json = "Generate a JSON object with keys 'name' and 'city'."
llm_response = await genie.llm.generate(prompt_for_json)
# llm_response['text'] might be: 'Sure, here is the JSON: {"name": "Test User", "city": "Genieville"}'

try:
    # Uses default parser if configured, or specify with parser_id="json_output_parser_v1"
    parsed_data = await genie.llm.parse_output(llm_response) 
    print(f"Parsed data: {parsed_data}")
    # Output: Parsed data: {'name': 'Test User', 'city': 'Genieville'}
except ValueError as e:
    print(f"Failed to parse LLM output: {e}")
```

**Example: Parsing into a Pydantic model**
```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# Configure Pydantic parser (pydantic_output_parser_v1)
# app_config = MiddlewareConfig(
#     features=FeatureSettings(llm="ollama", default_llm_output_parser="pydantic_output_parser")
# )
# genie = await Genie.create(config=app_config)

prompt_for_pydantic = "Create a JSON for a user named Bob, age 42."
llm_response = await genie.llm.generate(prompt_for_pydantic)
# llm_response['text'] might be: '```json\n{"name": "Bob", "age": 42}\n```'

try:
    user_instance = await genie.llm.parse_output(llm_response, schema=UserInfo)
    if isinstance(user_instance, UserInfo):
        print(f"User: {user_instance.name}, Age: {user_instance.age}")
except ValueError as e:
    print(f"Failed to parse into Pydantic model: {e}")
```
See the specific `LLMOutputParserPlugin` documentation for details on their capabilities and configuration (e.g., `JSONOutputParserPlugin`, `PydanticOutputParserPlugin`).

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
