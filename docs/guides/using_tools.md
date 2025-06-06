# Using Tools

Genie Tooling provides robust mechanisms for defining, discovering, and executing tools. Tools represent discrete actions an AI agent can perform.

The primary way to interact with tools is through the `Genie` facade.

## Enabling Tools

**Crucially, for a tool to be available for execution (either directly or via `run_command`), its plugin ID must be included as a key in the `tool_configurations` dictionary within your `MiddlewareConfig`.** If a tool requires no specific configuration, an empty dictionary `{}` as its value is sufficient to enable it.

```python
from genie_tooling.config.models import MiddlewareConfig

app_config = MiddlewareConfig(
    # ... other settings ...
    tool_configurations={
        "calculator_tool": {}, # Enables the built-in calculator tool
        "sandboxed_fs_tool_v1": { # Enables and configures the FS tool
            "sandbox_base_path": "./my_agent_sandbox"
        },
        "my_custom_tool_id": {} # Enables your custom tool
    }
)
```

## Executing Tools Directly with `genie.execute_tool()`

If you know which tool you want to use (and it's enabled in `tool_configurations`) and have its parameters, you can execute it directly:

```python
import asyncio
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings
from genie_tooling.genie import Genie

async def main():
    app_config = MiddlewareConfig(
        features=FeatureSettings(llm="none", command_processor="none"),
        tool_configurations={
            "calculator_tool": {}, # Enable calculator
            "sandboxed_fs_tool_v1": {"sandbox_base_path": "./my_agent_sandbox"}
        }
    )
    genie = await Genie.create(config=app_config)

    # Example: Using the built-in CalculatorTool
    calc_result = await genie.execute_tool(
        "calculator_tool", 
        num1=10, 
        num2=5, 
        operation="multiply"
    )
    print(f"Calculator Result: {calc_result}")
    # Output: Calculator Result: {'result': 50.0, 'error_message': None}

    # Example: Using the SandboxedFileSystemTool (if configured)
    await genie.execute_tool(
        "sandboxed_fs_tool_v1",
        operation="write_file",
        path="example.txt",
        content="Hello from Genie!"
    )
    print("Wrote to sandbox via execute_tool.")

    await genie.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Processing Natural Language Commands with `genie.run_command()`

For more agentic behavior, where the system needs to interpret a natural language command to select and parameterize an *enabled* tool, use `genie.run_command()`. This method leverages a configured **Command Processor** plugin.

```python
import asyncio
import json
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings
from genie_tooling.genie import Genie

async def main():
    # Configure Genie to use an LLM-assisted command processor
    # Ensure Ollama is running and mistral model is pulled for this example
    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", 
            llm_ollama_model_name="mistral:latest",
            command_processor="llm_assisted",
            tool_lookup="embedding" # Enable tool lookup for the LLM processor
        ),
        tool_configurations={
            "calculator_tool": {} # Ensure calculator is enabled
        }
    )
    genie = await Genie.create(config=app_config)

    command_text = "What is the result of 75 divided by 3?"
    print(f"Running command: '{command_text}'")
    
    command_output = await genie.run_command(command_text)
    print(f"Command Output: {json.dumps(command_output, indent=2)}")
    # Expected output might include:
    # {
    #   "tool_result": { "result": 25.0, "error_message": null },
    #   "thought_process": "The user wants to divide 75 by 3. I will use the calculator_tool..."
    # }

    await genie.close()

if __name__ == "__main__":
    asyncio.run(main())
```

Refer to the [Using Command Processors](using_command_processors.md) guide for more details on configuring different command processors.

## Defining Tools

Genie supports two main ways to define tools:

1.  **Plugin-based Tools**: Create a class that inherits from `genie_tooling.tools.abc.Tool` and implements the required methods (`identifier`, `get_metadata`, `execute`). These tools are discovered via entry points or plugin development directories. See [Creating Tool Plugins](creating_tool_plugins.md). **Remember to enable them via `tool_configurations`.**
2.  **Decorator-based Tools**: Use the `@genie_tooling.tool` decorator on your Python functions. Genie can then register these decorated functions as tools using `await genie.register_tool_functions([...])`. **After registration, their identifiers (typically the function names) must also be added to `tool_configurations` to be active for `genie.execute_tool` or `genie.run_command`.**

    ```python
    from genie_tooling import tool
    from genie_tooling.config.models import MiddlewareConfig # For app_config
    from genie_tooling.genie import Genie # For Genie
    import asyncio # For async main

    @tool
    async def my_custom_utility(text: str, uppercase: bool = False) -> str:
        """
        A custom utility function.
        Args:
            text (str): The input text.
            uppercase (bool): Whether to convert the text to uppercase.
        Returns:
            str: The processed text.
        """
        if uppercase:
            return text.upper()
        return text
    
    async def run_decorated_tool_example():
        app_config = MiddlewareConfig(
            tool_configurations={
                "my_custom_utility": {} # Enable the decorated tool
            }
        )
        genie = await Genie.create(config=app_config)
        await genie.register_tool_functions([my_custom_utility])
        
        result = await genie.execute_tool("my_custom_utility", text="hello", uppercase=True)
        print(result) # Output: {'result': 'HELLO'}
        await genie.close()

    # if __name__ == "__main__":
    #     asyncio.run(run_decorated_tool_example())
    ```

## Configuring Tools

Specific tools might require configuration (e.g., the base path for `SandboxedFileSystemTool`). This is done via the `tool_configurations` dictionary in `MiddlewareConfig`, which also serves to enable the tool:

```python
app_config = MiddlewareConfig(
    # ... other settings ...
    tool_configurations={
        "sandboxed_fs_tool_v1": { # Canonical plugin ID of the tool
            "sandbox_base_path": "./agent_data_sandbox"
        },
        "my_custom_api_tool_v1": { # Example for a hypothetical custom tool
            "api_endpoint": "https://api.example.com/custom",
            "default_timeout": 30
        },
        "calculator_tool": {} # Enable calculator with no specific config
    }
)
```
The configuration dictionary provided here will be passed to the tool's `setup()` method.

## Advanced Usage

For more fine-grained control, you can interact directly with the `ToolManager` and `ToolInvoker` components, which are accessible via the `Genie` instance (e.g., `genie._tool_manager`) if needed, though direct interaction is typically for advanced scenarios or extending Genie's core behavior.
