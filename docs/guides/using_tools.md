# Using Tools

Genie Tooling provides robust mechanisms for defining, discovering, and executing tools. Tools represent discrete actions an AI agent can perform.

The primary way to interact with tools is through the `Genie` facade.

## Enabling Tools: Automatic vs. Explicit

Genie supports two modes for enabling tools, controlled by the `auto_enable_registered_tools` flag in `MiddlewareConfig`. This is a critical security and configuration concept.

### Automatic Mode (Default for Development)

By default, `auto_enable_registered_tools` is `True`. This is designed for a frictionless developer experience, especially during prototyping.

*   **How it works**: Any function decorated with `@tool` and registered via `await genie.register_tool_functions([...])` is **automatically enabled** and available for use by `genie.execute_tool` and `genie.run_command`.
*   **Configuration**: You only need to add an entry to `tool_configurations` if a tool requires specific settings for its `setup()` method. Class-based tools (like the built-in `calculator_tool`) generally still need to be listed to be enabled.

```python
# In your main application file:
from genie_tooling import tool, Genie
from genie_tooling.config.models import MiddlewareConfig

@tool
def my_simple_tool():
    return "It works!"

@tool
def my_configurable_tool(api_endpoint: str):
    # ... uses api_endpoint ...
    return "Configured tool works!"

# Development configuration
app_config = MiddlewareConfig(
    auto_enable_registered_tools=True, # This is the default
    tool_configurations={
        # Only tools needing specific config are listed here
        "my_configurable_tool": {"api_endpoint": "https://api.example.com"},
        # Built-in class-based tools still need to be listed to be enabled
        "calculator_tool": {},
    }
)
# genie = await Genie.create(config=app_config)
# await genie.register_tool_functions([my_simple_tool, my_configurable_tool])

# All three tools are now active:
# await genie.execute_tool("my_simple_tool")
# await genie.execute_tool("my_configurable_tool")
# await genie.execute_tool("calculator_tool", ...)
```

### Explicit Mode (Recommended for Production)

For production environments, it is **strongly recommended** to set `auto_enable_registered_tools=False`. This provides a clear, secure, and auditable manifest of the agent's capabilities.

*   **How it works**: A tool is only active if its identifier is present as a key in the `tool_configurations` dictionary. This applies to **both** class-based plugins and `@tool` decorated functions.
*   **Security**: This prevents accidental exposure of development or debugging tools in a production setting. It enforces the principle of least privilege.

```python
# In your production configuration:
app_config = MiddlewareConfig(
    auto_enable_registered_tools=False, # Explicitly disable auto-enablement
    tool_configurations={
        # Only tools listed here will be active, regardless of what's registered.
        "my_simple_tool": {}, # Enable with no config
        "my_configurable_tool": {"api_endpoint": "https://api.example.com"},
        "calculator_tool": {}
        # A hypothetical 'dev_debug_tool' would NOT be active even if registered.
    }
)
```

## Executing Tools Directly with `genie.execute_tool()`

If you know which tool you want to use and have its parameters, you can execute it directly:

```python
# Assuming 'genie' is initialized and 'calculator_tool' is enabled.
calc_result = await genie.execute_tool(
    "calculator_tool",
    num1=10,
    num2=5,
    operation="multiply"
)
print(f"Calculator Result: {calc_result}")
# Output: Calculator Result: {'result': 50.0, 'error_message': None}
```

## Processing Natural Language Commands with `genie.run_command()`

For more agentic behavior, where the system needs to interpret a natural language command to select and parameterize an enabled tool, use `genie.run_command()`. This method leverages a configured **Command Processor** plugin.

```python
# Assuming 'genie' is initialized with an LLM-assisted command processor
# and 'calculator_tool' is enabled.
command_text = "What is the result of 75 divided by 3?"
command_output = await genie.run_command(command_text)
print(f"Command Output: {command_output}")
```
Refer to the [Using Command Processors](using_command_processors.md) guide for more details.

## Defining Tools

Genie supports two main ways to define tools:

1.  **Plugin-based Tools**: Create a class that inherits from `genie_tooling.tools.abc.Tool`. See [Creating Tool Plugins](creating_tool_plugins.md).
2.  **Decorator-based Tools**: Use the `@genie_tooling.tool` decorator on your Python functions.

Remember that regardless of how a tool is defined, its enablement is controlled by the `auto_enable_registered_tools` flag and the `tool_configurations` dictionary.
