# tst4.py
#!/usr/bin/env python3
"""
Genie Tooling - DeepResearchAgent E2E Test
"""
import argparse
import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv
from genie_tooling.agents import DeepResearchAgent
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.tools.impl.community_google_search_tool import (
    community_google_search,
)
from genie_tooling.tools.impl.custom_text_parameter_extractor import (
    custom_text_parameter_extractor,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("genie_tooling").setLevel(logging.INFO)
logging.getLogger("genie_tooling.agents.deep_research_agent").setLevel(logging.DEBUG)


async def run_deep_research_test():
    parser = argparse.ArgumentParser(
        description="Run the DeepResearchAgent E2E test with a configurable LLM provider.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--provider", required=True, choices=["ollama", "llama_cpp_internal", "gemini"], help="The LLM provider to use.")
    parser.add_argument("--model", required=True, help="The model name or path.")
    parser.add_argument("--ctx-length", type=int, default=8096, help="Context length for llama_cpp_internal.")
    args = parser.parse_args()

    load_dotenv()
    print("--- Genie Tooling: DeepResearchAgent E2E Test ---")
    print(f"Provider: {args.provider}, Model: {args.model}")

    feature_settings_args = {
        "command_processor": "none",
        "tool_lookup": "hybrid",
        "observability_tracer": "console_tracer",
        "logging_adapter": "pyvider_log_adapter",
        "token_usage_recorder": "in_memory_token_recorder",
        "default_llm_output_parser": "pydantic_output_parser",
    }

    if args.provider == "ollama":
        feature_settings_args["llm"] = "ollama"
        feature_settings_args["llm_ollama_model_name"] = args.model
    elif args.provider == "gemini":
        feature_settings_args["llm"] = "gemini"
        feature_settings_args["llm_gemini_model_name"] = args.model
    elif args.provider == "llama_cpp_internal":
        feature_settings_args["llm"] = "llama_cpp_internal"
        feature_settings_args["llm_llama_cpp_internal_model_path"] = args.model
        feature_settings_args["llm_llama_cpp_internal_n_ctx"] = args.ctx_length
        feature_settings_args["llm_llama_cpp_internal_n_gpu_layers"] = -1

    app_config = MiddlewareConfig(
        features=FeatureSettings(**feature_settings_args),
        auto_enable_registered_tools=False,
        tool_configurations={
            "intelligent_search_aggregator_v1": {},
            "arxiv_search_tool": {},
            "content_retriever_tool_v1": {},
            "custom_text_parameter_extractor": {},
            "community_google_search": {},
            "pdf_text_extractor_tool_v1": {},
            "web_page_scraper_tool_v1": {},
        },
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie facade...")
        genie = await Genie.create(config=app_config)

        # Register all necessary function-based tools
        await genie.register_tool_functions([
            community_google_search,
            custom_text_parameter_extractor,
        ])
        print("Genie facade initialized and function-based tools registered.")

        agent_config = {
            "web_search_tool_id": "intelligent_search_aggregator_v1",
            "academic_search_tool_id": "arxiv_search_tool",
            "content_extraction_tool_id": "content_retriever_tool_v1",
            "data_extraction_tool_id": "custom_text_parameter_extractor",
            "min_high_quality_sources": 2,
            "max_replanning_loops": 1,
        }
        research_agent = DeepResearchAgent(genie=genie, agent_config=agent_config)
        print("DeepResearchAgent initialized.")

        query = "Summarize the key contributions of the 'Transformer' architecture as introduced in 'Attention Is All You Need' (ArXiv:1706.03762) and find one recent (2023-2024) application of Transformers in natural language processing."

        print(f"\nConducting research for query: '{query}'")
        agent_output = await research_agent.run(goal=query)

        print("\n\n" + "=" * 25 + " AGENT RUN COMPLETE " + "=" * 25)
        print(f"Agent Status: {agent_output['status']}")
        print("-" * 60)
        print("\n### FINAL REPORT ###\n")
        print(agent_output["output"])
        print("-" * 60)

    except Exception as e:
        print(f"\nE2E Test FAILED: An unexpected error occurred: {e}")
        logging.exception("DeepResearchAgent E2E Test Error")
    finally:
        if genie:
            print("\nTearing down Genie facade...")
            await genie.close()
            print("Genie facade torn down.")


if __name__ == "__main__":
    asyncio.run(run_deep_research_test())
