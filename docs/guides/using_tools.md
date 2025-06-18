# Using Tools

Tools are fundamental to Genie Tooling, representing discrete actions an agent can perform.

## Enabling Tools: The Production-Safe Way

For security and clarity, Genie requires tools to be explicitly enabled in the configuration. This prevents accidental exposure of development or unused tools in a production environment.

The enablement is controlled by two `MiddlewareConfig` fields:

*   **`auto_enable_registered_tools: bool`**
    *   **Default**: `True`. This is for convenience during development. When `True`, any function decorated with `@tool` and registered via `genie.register_tool_functions()` is automatically enabled.
    *   **Production**: **Set this to `False`**.

*   **`tool_configurations: Dict[str, Dict[str, Any]]`**
    *   This dictionary is the **single source of truth for which tools are active** when `auto_enable_registered_tools` is `False`.
    *   The **keys** are the tool identifiers (e.g., `"calculator_tool"`, `"my_custom_tool"`).
    *   The **values** are configuration dictionaries passed to the tool's `setup()` method. Use an empty dictionary `{}` if no configuration is needed.

**Production Configuration Example:**
```python
# @tool decorated functions
# async def my_tool_one(): ...
# async def my_tool_two(config_param: str): ...

app_config = MiddlewareConfig(
    # Set to False for production safety!
    auto_enable_registered_tools=False,
    
    tool_configurations={
        # Only tools listed here will be active.
        "my_tool_one": {}, # Enable with no config.
        "my_tool_two": {"config_param": "value"}, # Enable AND configure.
        "calculator_tool": {}, # Enable the built-in calculator.
        
        # A hypothetical 'dev_debug_tool' would NOT be active even if
        # it was registered with genie.register_tool_functions().
    }
)

# genie = await Genie.create(config=app_config)
# await genie.register_tool_functions([my_tool_one, my_tool_two])
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
# and 'calculator_tool' is enabled in tool_configurations.
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
