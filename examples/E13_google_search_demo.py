# examples/E13_google_search_demo.py
"""
Example: GoogleSearchTool Demo
-----------------------------
This example demonstrates configuring and using the GoogleSearchTool
via the Genie facade.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Set Environment Variables:
   - `export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"`
   - `export GOOGLE_CSE_ID="YOUR_CUSTOM_SEARCH_ENGINE_ID"`
   (Replace with your actual key and CSE ID from Google Cloud Console
    and Programmable Search Engine setup)
3. Run from the root of the project:
   `poetry run python examples/E13_google_search_demo.py`

The demo will:
- Initialize Genie with the GoogleSearchTool.
- Perform a search query using the tool.
- Print the search results or an error message.
"""
import asyncio
import json
import logging
import os

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie

async def run_google_search_demo():
    print("--- GoogleSearchTool Demo ---")

    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
        print("\nERROR: Please set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables to run this demo.")
        print("Example: ")
        print('  export GOOGLE_API_KEY="your_actual_api_key"')
        print('  export GOOGLE_CSE_ID="your_actual_cse_id"')
        return

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="none",
            command_processor="none"
        ),
        tool_configurations={
            "google_search_tool_v1": {} # Enable the Google Search tool
        }
    )

    genie: Genie | None = None
    try:
        genie = await Genie.create(config=app_config)
        print("Genie facade initialized.")

        search_query = "What is Retrieval Augmented Generation?"
        num_results_to_fetch = 3

        print(f"\nAttempting to search for: '{search_query}' (max {num_results_to_fetch} results)...")
        search_result = await genie.execute_tool(
            "google_search_tool_v1",
            query=search_query,
            num_results=num_results_to_fetch
        )

        print("\nSearch Result:")
        print(json.dumps(search_result, indent=2))

        if search_result.get("error"):
            print(f"\nSearch failed: {search_result['error']}")
        elif search_result.get("results"):
            print(f"\nSuccessfully retrieved {len(search_result['results'])} results.")
            for i, item in enumerate(search_result["results"]):
                print(f"  Result {i+1}:")
                print(f"    Title: {item.get('title')}")
                print(f"    Link: {item.get('link')}")
                print(f"    Snippet: {item.get('snippet')[:100]}...")
        else:
            print("\nSearch returned no results and no error.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if genie:
            await genie.close()
            print("\nGenie facade torn down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_google_search_demo())
