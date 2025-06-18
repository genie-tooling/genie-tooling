# Using LLM Providers

Genie Tooling allows you to easily interact with various Large Language Models (LLMs) through a unified interface: `genie.llm`. This interface abstracts the specifics of different LLM provider APIs.

## The `genie.llm` Interface

Once you have a `Genie` instance, you can access LLM functionalities:

*   **`async genie.llm.generate(prompt: str, provider_id: Optional[str] = None, stream: bool = False, **kwargs) -> Union[LLMCompletionResponse, AsyncIterable[LLMCompletionChunk]]`**:
    For text completion or generation tasks.
    *   `prompt`: The input prompt string.
    *   `provider_id`: Optional. The ID of the LLM provider plugin to use. If `None`, the default LLM provider configured in `MiddlewareConfig` (via `features.llm` or `default_llm_provider_id`) is used.
    *   `stream`: Optional (default `False`). If `True`, returns an async iterable of `LLMCompletionChunk` objects.
    *   `**kwargs`: Additional provider-specific parameters (e.g., `temperature`, `max_tokens`, `model` to override the default for that provider, `output_schema` for structured output).
*   **`async genie.llm.chat(messages: List[ChatMessage], provider_id: Optional[str] = None, stream: bool = False, **kwargs) -> Union[LLMChatResponse, AsyncIterable[LLMChatChunk]]`**:
    For conversational interactions.
    *   `messages`: A list of `ChatMessage` dictionaries (see [Types](#chatmessage-type)).
    *   `provider_id`: Optional. The ID of the LLM provider plugin to use.
    *   `stream`: Optional (default `False`). If `True`, returns an async iterable of `LLMChatChunk` objects.
    *   `**kwargs`: Additional provider-specific parameters (e.g., `temperature`, `tools`, `tool_choice`, `output_schema` for structured output).
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

### Example: Using Llama.cpp (Server Mode)

```python
# Assumes a Llama.cpp server is running at the specified URL.
app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="llama_cpp",
        llm_llama_cpp_model_name="your-model-alias-on-server", # Alias/model server uses
        llm_llama_cpp_base_url="http://localhost:8080" # Default, adjust if needed
        # llm_llama_cpp_api_key_name="MY_LLAMA_SERVER_KEY" # If server requires API key
    )
)
# genie = await Genie.create(config=app_config)
# ... use genie.llm.chat or genie.llm.generate ...
```

### Example: Using Llama.cpp (Internal Mode)

This provider runs a GGUF model file directly in your application's process using the `llama-cpp-python` library. It offers a fully local, serverless setup.

```python
# Requires llama-cpp-python library and a GGUF model file.
# Install with: poetry install --extras llama_cpp_internal
app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="llama_cpp_internal",
        llm_llama_cpp_internal_model_path="/path/to/your/model.gguf", # IMPORTANT: Set this path
        llm_llama_cpp_internal_n_gpu_layers=-1, # Offload all possible layers to GPU
        llm_llama_cpp_internal_n_ctx=4096,      # Example context size
        llm_llama_cpp_internal_chat_format="mistral" # Or "llama-2", "chatml", etc.
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
        },
        "llama_cpp_internal_llm_provider_v1": { # Canonical ID for internal Llama.cpp
            "model_path": "/another/model.gguf",
            "n_gpu_layers": 0, # CPU only for this specific override
            "chat_format": "chatml",
            "model_name_for_logging": "custom_internal_llama"
        }
    }
)
# genie = await Genie.create(config=app_config)

# This will use OpenAI with gpt-4-turbo-preview
# await genie.llm.chat([{"role": "user", "content": "Hello OpenAI!"}])

# This will use Ollama with llama3:latest
# await genie.llm.chat([{"role": "user", "content": "Hello Ollama!"}], provider_id="ollama")

# This will use the internal Llama.cpp provider with /another/model.gguf
# await genie.llm.generate("Test internal Llama.cpp", provider_id="llama_cpp_internal")
```

### API Keys and `KeyProvider`

LLM providers that require API keys (like OpenAI, Gemini, or a secured Llama.cpp server) will attempt to fetch them using the configured `KeyProvider`. By default, Genie uses `EnvironmentKeyProvider`, which reads keys from environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`). You can provide a custom `KeyProvider` instance to `Genie.create()` for more sophisticated key management. The internal Llama.cpp provider does not use API keys managed via `KeyProvider`.

## Structured Output (GBNF with Llama.cpp Providers)

Both the Llama.cpp server provider (`llama_cpp_llm_provider_v1`) and the internal Llama.cpp provider (`llama_cpp_internal_llm_provider_v1`) support GBNF grammar for constrained, structured output. You can pass a Pydantic model class or a JSON schema dictionary to the `output_schema` parameter of `genie.llm.generate()` or `genie.llm.chat()`.

```python
from pydantic import BaseModel

class MyData(BaseModel):
    name: str
    value: int

# Assuming 'genie' is configured with a Llama.cpp provider (server or internal)
# response_chat = await genie.llm.chat(
#     messages=[{"role": "user", "content": "User: Name is Beta, Value is 20. Output JSON."}],
#     output_schema=MyData
# )
# if response_chat['message']['content']:
#     parsed_chat = await genie.llm.parse_output(response_chat, schema=MyData)
```
The provider will attempt to convert the schema into a GBNF grammar string and pass it to the Llama.cpp backend.

## Parsing LLM Output

The `genie.llm.parse_output()` method helps convert LLM text responses into structured data.

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# Assuming 'genie' is configured with default_llm_output_parser="pydantic_output_parser"
# llm_response = await genie.llm.generate("Create JSON for Bob, age 42.")
# user_instance = await genie.llm.parse_output(llm_response, schema=UserInfo)
```

## `ChatMessage` Type

The `messages` parameter for `genie.llm.chat()` expects a list of `ChatMessage` dictionaries:

```python
from genie_tooling.llm_providers.types import ChatMessage

# User message
user_message: ChatMessage = {"role": "user", "content": "What's the weather in London?"}

# Assistant message requesting a tool call
assistant_tool_call_request: ChatMessage = {
    "role": "assistant",
    "tool_calls": [
        {
            "id": "call_weather_london_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "London"}'}
        }
    ]
}

# Tool message (response from executing a tool)
tool_response_message: ChatMessage = {
    "role": "tool",
    "tool_call_id": "call_weather_london_123",
    "name": "get_weather",
    "content": '{"temperature": 15, "condition": "Cloudy"}'
}
```
