# Tutorial: LLM-Assisted Command (E07)

This tutorial corresponds to the example file `examples/E07_run_command_llm_assisted_example.py`.

It demonstrates the core agentic capability of using an LLM to interpret a natural language command. It shows how to:
- Configure the `llm_assisted` command processor.
- Configure an embedding-based `tool_lookup` service to help the LLM find relevant tools.
- Use `genie.run_command()`, which will leverage the LLM to select the correct tool and extract its parameters from the user's sentence, all in one step.

## Example Code

--8<-- "examples/E07_run_command_llm_assisted_example.py"
