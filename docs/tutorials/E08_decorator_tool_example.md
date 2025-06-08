# Tutorial: @tool Decorator (E08)

This tutorial corresponds to the example file `examples/E08_decorator_tool_example.py`.

It demonstrates the easiest way to create custom tools in Genie. It shows how to:
- Define a standard Python function (sync or async).
- Apply the `@genie_tooling.tool` decorator to it, which automatically generates the necessary metadata and schema from the function's signature and docstring.
- Register the decorated function with `genie.register_tool_functions()`.
- Execute the tool directly using `genie.execute_tool()`.

## Example Code

--8<-- "examples/E08_decorator_tool_example.py"
