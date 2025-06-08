# Tutorial: Simple Agent CLI (E01)

This tutorial corresponds to the example file `examples/E01_simple_agent_cli.py`.

It demonstrates a basic command-line agent that does **not** use an LLM for tool selection. Instead, it shows how to:
- Configure the `simple_keyword` command processor, which maps keywords like "calculate" or "weather" to specific tool IDs.
- Explicitly enable the `calculator_tool` and `open_weather_map_tool` in `tool_configurations`.
- Use `genie.run_command()` to process user input. The keyword processor will find the tool and then prompt the user for the necessary parameters.

## Example Code

--8<-- "examples/E01_simple_agent_cli.py"
