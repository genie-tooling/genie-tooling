# Using the Prompt Management System (`genie.prompts`)

Genie Tooling provides a flexible prompt management system accessible via `genie.prompts`. This allows you to register, retrieve, and render prompt templates, separating prompt engineering concerns from your application logic.

## Core Concepts

*   **`PromptInterface` (`genie.prompts`)**: The facade interface for all prompt-related operations.
*   **`PromptRegistryPlugin`**: Responsible for storing and retrieving raw prompt template content (e.g., from a file system, database).
    *   Built-in: `FileSystemPromptRegistryPlugin` (alias: `file_system_prompt_registry`).
*   **`PromptTemplatePlugin`**: Responsible for rendering a raw template string with provided data.
    *   Built-in: `BasicStringFormatTemplatePlugin` (alias: `basic_string_formatter`), `Jinja2ChatTemplatePlugin` (alias: `jinja2_chat_formatter`).

## Configuration

You configure the default prompt registry and template engine via `FeatureSettings` or explicit `MiddlewareConfig` settings.

**Via `FeatureSettings`:**

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        # ... other features ...
        prompt_registry="file_system_prompt_registry", # Default
        prompt_template_engine="jinja2_chat_formatter"   # Default
    ),
    # Configure the FileSystemPromptRegistryPlugin
    prompt_registry_configurations={
        "file_system_prompt_registry_v1": { # Canonical ID
            "base_path": "./my_application_prompts",
            "template_suffix": ".j2" # If using Jinja2 templates
        }
    }
    # No specific config needed for Jinja2ChatTemplatePlugin by default
)
```

**Explicit Configuration:**

```python
app_config = MiddlewareConfig(
    default_prompt_registry_id="file_system_prompt_registry_v1",
    default_prompt_template_plugin_id="jinja2_chat_template_v1",
    prompt_registry_configurations={
        "file_system_prompt_registry_v1": {"base_path": "prompts"}
    }
)
```

## Using `genie.prompts`

### 1. Getting Raw Template Content

```python
# Assuming 'my_application_prompts/my_task.j2' exists
template_content = await genie.prompts.get_prompt_template_content(
    name="my_task", 
    # version="v1", # Optional version
    # registry_id="custom_registry" # Optional, if not using default
)
if template_content:
    print(f"Raw template: {template_content}")
```

### 2. Rendering a String Prompt

This is useful for prompts that result in a single string, often for completion LLMs.

```python
from genie_tooling.prompts.types import PromptData

prompt_data: PromptData = {"topic": "AI ethics", "length": "short"}

# Uses default registry and default template engine
rendered_string = await genie.prompts.render_prompt(
    name="summarization_prompt", # e.g., 'my_application_prompts/summarization_prompt.txt'
    data=prompt_data
)
if rendered_string:
    print(f"Rendered string prompt: {rendered_string}")
    # Example: "Summarize AI ethics in a short paragraph."
```

### 3. Rendering a Chat Prompt

This is used for prompts that result in a list of `ChatMessage` dictionaries, suitable for chat-based LLMs. The `Jinja2ChatTemplatePlugin` is particularly useful here as Jinja2 can easily render structured JSON/YAML.

**Example Jinja2 template (`my_application_prompts/chat_style_prompt.j2`):**
```jinja2
[
    {"role": "system", "content": "You are a helpful {{ persona }}."},
    {"role": "user", "content": "Tell me about {{ subject }}."}
]
```

**Python code:**
```python
from genie_tooling.prompts.types import PromptData

chat_data: PromptData = {"persona": "historian", "subject": "the Roman Empire"}

# Uses default registry and default template engine (configured to jinja2_chat_formatter)
chat_messages = await genie.prompts.render_chat_prompt(
    name="chat_style_prompt", 
    data=chat_data
)
if chat_messages:
    print(f"Rendered chat messages: {chat_messages}")
    # Output:
    # [
    #   {'role': 'system', 'content': 'You are a helpful historian.'},
    #   {'role': 'user', 'content': 'Tell me about the Roman Empire.'}
    # ]
    # response = await genie.llm.chat(chat_messages)
```

### 4. Listing Available Templates

```python
available_templates = await genie.prompts.list_templates()
# Or for a specific registry:
# available_templates = await genie.prompts.list_templates(registry_id="my_other_registry")

for template_id in available_templates:
    print(f"- Name: {template_id['name']}, Version: {template_id.get('version', 'N/A')}, Desc: {template_id.get('description')}")
```

## Creating Custom Prompt Plugins

*   **`PromptRegistryPlugin`**: Implement `get_template_content` and `list_available_templates`.
*   **`PromptTemplatePlugin`**: Implement `render` (for string output) and `render_chat_messages` (for `List[ChatMessage]` output).

Register your custom plugins via entry points in `pyproject.toml` or ensure they are discoverable via `plugin_dev_dirs`.
