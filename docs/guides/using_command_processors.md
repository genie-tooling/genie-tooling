# Using Command Processors

Command Processors in Genie Tooling are responsible for interpreting a user's natural language command, selecting an appropriate tool, and extracting the necessary parameters for that tool. The primary way to use a command processor is via the `genie.run_command()` method.

## `genie.run_command()`

```python
async def run_command(
    self, 
    command: str, 
    processor_id: Optional[str] = None,
    conversation_history: Optional[List[ChatMessage]] = None
) -> Any:
```

*   `command`: The natural language command string from the user.
*   `processor_id`: Optional. The ID of the `CommandProcessorPlugin` to use. If `None`, the default command processor configured in `MiddlewareConfig` (via `features.command_processor` or `default_command_processor_id`) is used.
*   `conversation_history`: Optional list of previous `ChatMessage` dictionaries to provide context.

The method returns a dictionary, typically including:
*   `tool_result`: The result from the executed tool, if a tool was chosen and run successfully.
*   `thought_process`: An explanation from the processor (especially LLM-based ones).
*   `error`: An error message if processing or tool execution failed.
*   `message`: A message if, for example, no tool was selected.
*   `hitl_decision`: If HITL was triggered, this contains the approval response.

**Important**: Any tool that a command processor might select must be enabled in `MiddlewareConfig.tool_configurations`.

## Configuring Command Processors

You select and configure command processors using `FeatureSettings` or explicit configurations in `MiddlewareConfig`.

### 1. Simple Keyword Processor (`simple_keyword`)

This processor matches keywords in the user's command against a predefined map to select a tool. It then prompts the user for parameters if the selected tool requires them.

**Configuration via `FeatureSettings`:**

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        command_processor="simple_keyword"
    ),
    # Explicit configuration for the simple_keyword_processor_v1
    command_processor_configurations={
        "simple_keyword_processor_v1": { # Canonical ID
            "keyword_map": {
                "calculate": "calculator_tool",
                "math": "calculator_tool",
                "weather": "open_weather_map_tool"
            },
            "keyword_priority": ["calculate", "math", "weather"] # Order for matching
        }
    },
    tool_configurations={ # Tools must be enabled
        "calculator_tool": {},
        "open_weather_map_tool": {}
    }
)
# genie = await Genie.create(config=app_config)
# result = await genie.run_command("calculate 10 + 5") 
# The processor will prompt for num1, num2, operation for calculator_tool.
```

### 2. LLM-Assisted Processor (`llm_assisted`)

This processor uses an LLM to understand the command, select the most appropriate tool from a list of available tools, and extract its parameters.

**Configuration via `FeatureSettings`:**

```python
app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="ollama", # The LLM to be used by the processor
        llm_ollama_model_name="mistral:latest",

        command_processor="llm_assisted",

        # Configure how tools are presented to the LLM
        command_processor_formatter_id_alias="compact_text_formatter", 

        # Configure tool lookup to help the LLM (optional but recommended)
        tool_lookup="embedding", # Use embedding-based lookup
        tool_lookup_formatter_id_alias="compact_text_formatter", # Formatter for indexing tools
        tool_lookup_embedder_id_alias="st_embedder" # Embedder for tool descriptions
    ),
    # Optionally, provide specific settings for the LLM-assisted processor
    command_processor_configurations={
        "llm_assisted_tool_selection_processor_v1": { # Canonical ID
            "tool_lookup_top_k": 3, # Show top 3 tools from lookup to the LLM
            # "system_prompt_template": "Your custom system prompt..." # Override default prompt
        }
    },
    tool_configurations={ # Any tool the LLM might pick must be enabled
        "calculator_tool": {},
        "open_weather_map_tool": {} 
        # Add other tools the LLM might be expected to use
    }
)
# genie = await Genie.create(config=app_config)
# result = await genie.run_command("What's the weather like in Berlin tomorrow?")
```

**Key aspects for `llm_assisted` processor:**
*   **LLM Dependency**: It requires a configured LLM provider (`features.llm`).
*   **Tool Formatting**: It uses a `DefinitionFormatterPlugin` (specified by `command_processor_formatter_id_alias` or an explicit `tool_formatter_id` in its configuration) to format tool definitions for the LLM prompt.
*   **Tool Lookup (Optional but Recommended)**: If `features.tool_lookup` is enabled (e.g., `"embedding"` or `"keyword"`), the processor first uses the `ToolLookupService` to find a smaller set of relevant tools. Only these candidate tools are then presented to the LLM. This improves efficiency and accuracy. The `tool_lookup_top_k` parameter in its configuration controls how many tools from the lookup are passed to the LLM.

See the [Tool Lookup Guide](tool_lookup.md) for more on configuring tool lookup.
