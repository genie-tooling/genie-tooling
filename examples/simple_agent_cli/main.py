# examples/simple_agent_cli/main.py
"""
Example: Simple Agent CLI (Completed)
--------------------------------------
This example demonstrates a basic command-line agent that uses the Genie Tooling
middleware to interact with tools.

To Run:
1. Ensure you have Genie Tooling installed (`poetry install --all-extras`).
2. Set an environment variable for OpenWeatherMap API key if you want to test that tool:
   `export OPENWEATHERMAP_API_KEY="your_actual_key"`
3. Run from the root of the project:
   `poetry run python examples/simple_agent_cli/main.py`

The agent will:
- Ask for your input.
- Try to "guess" a tool based on keywords (calculator, weather).
- If a tool is chosen, it will ask for parameters.
- Execute the tool using Genie Tooling.
- Display the result.
"""
import asyncio
import os
import json # For pretty printing dicts
from typing import Dict, Any, Optional, List

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.tools.manager import ToolManager
from genie_tooling.invocation.invoker import ToolInvoker
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.core.types import Plugin # For KeyProvider to inherit
from genie_tooling.config.models import MiddlewareConfig

# --- 1. Basic KeyProvider Implementation (Application-Side) ---
class EnvironmentKeyProvider(KeyProvider, Plugin): # Inherit Plugin for manager compatibility
    plugin_id: str = "env_key_provider_v1_example"
    description: str = "Provides API keys from environment variables."

    async def get_key(self, key_name: str) -> Optional[str]:
        print(f"[KeyProvider] Requesting key: {key_name}")
        key_value = os.environ.get(key_name)
        if not key_value:
            print(f"[KeyProvider] Key '{key_name}' not found in environment variables.")
        return key_value

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        print(f"[{self.plugin_id}] Setup complete.")

    async def teardown(self) -> None:
        print(f"[{self.plugin_id}] Teardown complete.")


# --- 2. Simple Agent Logic ---
class SimpleAgent:
    def __init__(self, tool_manager: ToolManager, tool_invoker: ToolInvoker, key_provider: KeyProvider):
        self.tool_manager = tool_manager
        self.tool_invoker = tool_invoker
        self.key_provider = key_provider

    async def list_available_tools(self) -> None:
        print("\nAvailable Tools:")
        summaries, _ = await self.tool_manager.list_tool_summaries()
        if not summaries:
            print("  No tools available.")
            return
        for summary in summaries:
            print(f"  - {summary['name']} (ID: {summary['identifier']}): {summary['short_description']}")

    async def _get_llm_tool_choice_and_params(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        SIMULATES an LLM choosing a tool and extracting parameters.
        In a real agent, this would involve an LLM call with tool definitions.
        """
        query_lower = user_query.lower()
        
        # Simple keyword-based "LLM"
        if "calculate" in query_lower or "math" in query_lower or "+" in query_lower or "multiply" in query_lower:
            print("[Agent] LLM simulation: Looks like a calculation task.")
            try:
                num1_str = input("  Enter first number (num1): ")
                num2_str = input("  Enter second number (num2): ")
                operation = input("  Enter operation (add, subtract, multiply, divide): ").strip().lower()
                return {
                    "tool_identifier": "calculator_tool",
                    "params": {
                        "num1": float(num1_str),
                        "num2": float(num2_str),
                        "operation": operation
                    }
                }
            except ValueError:
                print("[Agent] Invalid number input for calculator.")
                return None
        elif "weather" in query_lower:
            print("[Agent] LLM simulation: Looks like a weather request.")
            city = input("  Enter city (e.g., 'London, UK' or 'Tokyo'): ").strip()
            units = input("  Enter units (metric/imperial, default metric): ").strip().lower() or "metric"
            if not city:
                print("[Agent] City name is required for weather.")
                return None
            return {
                "tool_identifier": "open_weather_map_tool",
                "params": {
                    "city": city,
                    "units": units
                }
            }
        else:
            print("[Agent] LLM simulation: I'm not sure which tool to use for that query.")
            return None

    async def handle_query(self, user_query: str) -> None:
        if not user_query.strip():
            return

        tool_choice = await self._get_llm_tool_choice_and_params(user_query)

        if not tool_choice:
            print("[Agent] Could not determine a tool or parameters for your request.")
            return

        tool_id = tool_choice["tool_identifier"]
        params = tool_choice["params"]

        print(f"\n[Agent] Attempting to invoke tool '{tool_id}' with params: {params}")
        
        # Use the ToolInvoker
        result = await self.tool_invoker.invoke(
            tool_identifier=tool_id,
            params=params,
            key_provider=self.key_provider
            # context, strategy_id, etc. can be specified if needed
        )

        print("\n[Agent] Tool Invocation Result:")
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(result)

    async def run_cli_loop(self):
        print("--- Simple Agent CLI ---")
        print("Type 'tools' to list available tools, 'quit' to exit.")
        await self.list_available_tools()

        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n> Your query: ") # Run input in thread to not block loop
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            
            if user_input.lower() == "quit":
                print("Exiting...")
                break
            elif user_input.lower() == "tools":
                await self.list_available_tools()
                continue
            
            await self.handle_query(user_input)


async def main():
    print("Initializing Simple Agent CLI...")

    # --- Middleware Configuration (Minimal) ---
    # Normally, this might come from a file or environment config
    middleware_cfg = MiddlewareConfig(
        # plugin_dev_dirs=["path/to/my/custom_plugins"] # Add if you have custom plugins
    )

    # --- Initialize Core Middleware Components ---
    plugin_manager = PluginManager(plugin_dev_dirs=middleware_cfg.plugin_dev_dirs)
    
    # Register our custom KeyProvider plugin programmatically if not in dev_dirs
    # This is one way to do it if it's defined within the app.
    # Alternatively, place EnvironmentKeyProvider in a dev_plugins directory.
    if EnvironmentKeyProvider.plugin_id not in plugin_manager.list_discovered_plugin_classes():
         plugin_manager._discovered_plugin_classes[EnvironmentKeyProvider.plugin_id] = EnvironmentKeyProvider # type: ignore
         plugin_manager._plugin_source_map[EnvironmentKeyProvider.plugin_id] = __file__ # type: ignore
         print(f"[CLI] Programmatically registered {EnvironmentKeyProvider.plugin_id}")
    
    # Discover plugins (including built-ins and those in dev_dirs, and our EnvKeyProvider if added above)
    await plugin_manager.discover_plugins()
    print(f"[CLI] Discovered plugins: {list(plugin_manager.list_discovered_plugin_classes().keys())}")


    tool_manager = ToolManager(plugin_manager=plugin_manager)
    # Configure which tools to initialize (empty means all discovered tools with default configs)
    # Example: tool_configurations={"open_weather_map_tool": {"some_config_key": "value"}}
    await tool_manager.initialize_tools(tool_configurations={})

    tool_invoker = ToolInvoker(tool_manager=tool_manager, plugin_manager=plugin_manager)
    
    # Get an instance of our KeyProvider
    # This assumes EnvironmentKeyProvider is registered or discoverable.
    key_provider_instance = await plugin_manager.get_plugin_instance(EnvironmentKeyProvider.plugin_id)
    if not key_provider_instance or not isinstance(key_provider_instance, KeyProvider):
        print(f"[CLI_ERROR] Failed to load the KeyProvider ({EnvironmentKeyProvider.plugin_id}). Exiting.")
        await plugin_manager.teardown_all_plugins()
        return
    
    key_provider = key_provider_instance # Use the resolved instance

    # --- Initialize and Run the Agent ---
    agent = SimpleAgent(tool_manager, tool_invoker, key_provider)
    
    try:
        await agent.run_cli_loop()
    finally:
        # --- Teardown Middleware Components ---
        print("\n[CLI] Tearing down middleware components...")
        # Teardown plugins managed by PluginManager (KeyProvider, Tools, etc.)
        await plugin_manager.teardown_all_plugins()
        print("[CLI] Teardown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CLI] Agent interrupted. Exiting.")