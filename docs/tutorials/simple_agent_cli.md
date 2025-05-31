# Tutorial: Simple Agent CLI

This tutorial walks you through building a simple command-line interface (CLI) agent using the `Genie` facade.

The complete code for this tutorial can be found in [examples/simple_agent_cli/main.py](https://github.com/genie-tooling/genie-tooling/blob/main/examples/simple_agent_cli/main.py). 
<!-- TODO: Update link when repo is public -->

## Prerequisites

*   Genie Tooling installed (`poetry install --all-extras`).
*   (Optional) `OPENWEATHERMAP_API_KEY` environment variable set if you want to test the weather tool.

## Core Logic

The agent will:
1.  Initialize the `Genie` facade with a configuration that enables a simple keyword-based command processor and some basic tools (like a calculator).
2.  Enter a loop to accept user input.
3.  Pass the user's query to `genie.run_command()`.
4.  The command processor will attempt to match keywords in the query to select a tool.
5.  If a tool is selected and requires parameters, the `SimpleKeywordToolSelectorProcessorPlugin` will prompt the user for them.
6.  The tool is executed, and the result is displayed.

Refer to the example script for the full implementation details. It showcases how to:
*   Define `MiddlewareConfig` using `FeatureSettings`.
*   Configure the `simple_keyword` command processor.
*   Use `Genie.create()` and `genie.run_command()`.
*   Handle basic CLI interaction.
