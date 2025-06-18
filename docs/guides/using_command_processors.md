# Using Command Processors

Command Processors in Genie Tooling are responsible for interpreting a user's natural language command, selecting an appropriate tool, and extracting the necessary parameters for that tool. The primary way to use a command processor is via the `genie.run_command()` method.

## `genie.run_command()`

```python
async def run_command(
    self,
    command: str,
    processor_id: Optional[str] = None,
    conversation_history: Optional[List[ChatMessage]] = None,
    context_for_tools: Optional[Dict[str, Any]] = None
) -> Any:
```

*   `command`: The natural language command string from the user.
*   `processor_id`: Optional. The ID of the `CommandProcessorPlugin` to use. If `None`, the default command processor configured in `MiddlewareConfig` is used.
*   `conversation_history`: Optional list of previous `ChatMessage` dictionaries to provide context.
*   `context_for_tools`: Optional dictionary passed to the `context` parameter of the executed tool.

The method returns a dictionary, typically including `tool_result`, `thought_process`, `error`, `message`, and `hitl_decision`.

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
        }
    },
    tool_configurations={ # Any tool the LLM might pick must be enabled
        "calculator_tool": {},
        "open_weather_map_tool": {} 
    }
)
# genie = await Genie.create(config=app_config)
# result = await genie.run_command("What's the weather like in Berlin tomorrow?")
```

### 3. ReWOO Agent as a Processor (`rewoo`)

The ReWOO (Reason-Act) processor is a powerful agentic loop that can be invoked as a command processor. It takes a complex user goal, creates a multi-step plan of tool calls, executes them, and then synthesizes a final answer from the collected evidence.

**Configuration via `FeatureSettings`:**
```python
app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="llama_cpp_internal", # ReWOO needs a capable LLM
        command_processor="rewoo",

        # Ensure plugins ReWOO might use are configured
        prompt_template_engine="jinja2_chat_formatter",
        default_llm_output_parser="pydantic_output_parser",
        tool_lookup="hybrid",
    ),
    # Any tools the agent might plan to use must be enabled
    tool_configurations={
        "intelligent_search_aggregator_v1": {},
        "content_retriever_tool_v1": {},
        # ... other tools ...
    }
)
# genie = await Genie.create(config=app_config)
# result = await genie.run_command("What were the key findings of the Llama 2 paper and how do they compare to GPT-4?")
```
The ReWOO processor is ideal for complex queries that cannot be answered by a single tool call and require a chain of reasoning and information gathering.
