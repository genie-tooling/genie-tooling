# Tutorial: Simple Keyword Command (E06)

This tutorial corresponds to the example file `examples/E06_run_command_simple_keyword_example.py`.

It demonstrates how to configure and use the `simple_keyword` command processor. This processor does not use an LLM. It shows how to:
- Define a `keyword_map` to associate keywords (like "calculate", "sum") with a specific tool (`calculator_tool`).
- Use `genie.run_command()`, which will trigger the processor to find the tool and then interactively prompt the user for the tool's parameters.

## Example Code

--8<-- "examples/E06_run_command_simple_keyword_example.py"
