#!/usr/bin/env python3
"""
Developer Utility: Debug Tool Discoverability

This script initializes a minimal Genie environment to test how the ToolLookupService
ranks tools for a given natural language query. It helps developers iterate on
tool descriptions to improve their discoverability.

Usage:
    poetry run python scripts/debug_tool_lookup.py --query "your natural language query"

Optional Arguments:
    --config-path: Path to a YAML or JSON file containing a simplified MiddlewareConfig.
                   Mainly used to specify `plugin_dev_dirs` to find custom tools.
    --provider-id: The ID of the ToolLookupProvider to test (e.g., 'hybrid_lookup', 'embedding_lookup').
                   Defaults to the one configured in the config file or the system default.
    --top-k:       Number of results to return. Defaults to 5.
    --log-level:   Set the logging level (e.g., DEBUG, INFO). Defaults to WARNING.
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


def load_config_from_file(path: str) -> Dict[str, Any]:
    """Loads a config dictionary from a YAML or JSON file."""
    p = Path(path)
    if not p.exists():
        print(f"Error: Config file not found at '{path}'")
        return {}
    with p.open("r", encoding="utf-8") as f:
        if p.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif p.suffix == ".json":
            return json.load(f)
    print(f"Error: Unsupported config file format for '{path}'. Use .yaml, .yml, or .json.")
    return {}


async def main():
    parser = argparse.ArgumentParser(description="Debug Genie Tooling's Tool Lookup Service.")
    parser.add_argument("--query", required=True, help="The natural language query to test.")
    parser.add_argument("--config-path", help="Path to a YAML/JSON file with MiddlewareConfig settings.")
    parser.add_argument("--provider-id", help="Override the ToolLookupProvider plugin ID to use.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to show.")
    parser.add_argument("--log-level", default="WARNING", help="Set logging level (e.g., DEBUG, INFO).")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())
    logging.getLogger("genie_tooling").setLevel(args.log_level.upper())

    print("--- Tool Lookup Debugger ---")
    print(f"Query: '{args.query}'")
    print(f"Top K: {args.top_k}")

    config_data = {}
    if args.config_path:
        print(f"Loading config from: {args.config_path}")
        config_data = load_config_from_file(args.config_path)

    # Create a MiddlewareConfig instance from the loaded data
    app_config = MiddlewareConfig(**config_data)

    # Set the default lookup provider if overridden via CLI
    if args.provider_id:
        print(f"Overriding lookup provider to: '{args.provider_id}'")
        app_config.default_tool_lookup_provider_id = args.provider_id

    genie = None
    try:
        genie = await Genie.create(config=app_config)
        print("Genie facade initialized. Discovering and indexing tools...")

        if not genie._tool_lookup_service:
            print("\nError: ToolLookupService was not initialized. Check configuration.")
            return

        # The first call to find_tools will trigger indexing automatically.
        results = await genie._tool_lookup_service.find_tools(
            natural_language_query=args.query,
            top_k=args.top_k,
            provider_id_override=args.provider_id
        )

        print("\n--- Lookup Results ---")
        if not results:
            print("No tools found for the given query.")
            return

        for i, result in enumerate(results):
            print(f"\n{i+1}. Tool: {result.tool_identifier} (Score: {result.score:.4f})")
            if result.matched_keywords:
                print(f"   Matched Keywords: {', '.join(result.matched_keywords)}")
            if result.similarity_score_details:
                print(f"   Similarity Details: {result.similarity_score_details}")
            if result.description_snippet:
                print(f"   Snippet/Reason: {result.description_snippet}")
            # print(f"   Matched Data: {result.matched_tool_data}") # Uncomment for very verbose output

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        logging.exception("Debugger error details:")
    finally:
        if genie:
            await genie.close()
            print("\n--- Teardown Complete ---")


if __name__ == "__main__":
    asyncio.run(main())
