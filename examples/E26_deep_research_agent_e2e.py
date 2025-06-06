# examples/E26_deep_research_agent_e2e.py
"""
Genie Tooling - DeepResearchAgent E2E Test with Deep Retrieval Tools
This script initializes the Genie facade, configures the DeepResearchAgent
to use DuckDuckGo for web searches, ArXiv search, and the deep retrieval tools
(WebPageScraperTool, ArxivPDFTextExtractorTool), and performs a research query.

This example uses the Llama.cpp server provider.

Prerequisites:
1. `genie-tooling` installed with relevant extras:
   `poetry install --all-extras`
   (Specifically needs: llama_cpp_server, ddg_search_tool, arxiv_tool, web_tools, pdf_tools)
2. A Llama.cpp server running (see E23 for example command), accessible via LLAMA_CPP_BASE_URL.
   Ensure the model alias (LLAMA_CPP_MODEL_ALIAS) matches your server setup.
"""
import asyncio
import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from genie_tooling.agents.deep_research_agent import DeepResearchAgent
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie

logging.basicConfig(level=logging.INFO)
logging.getLogger("genie_tooling.agents.deep_research_agent").setLevel(logging.DEBUG)
# logging.getLogger("genie_tooling").setLevel(logging.DEBUG) 

LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080")
LLAMA_CPP_MODEL_ALIAS = os.getenv("LLAMA_CPP_MODEL_ALIAS", "mistral:latest")
LLAMA_CPP_API_KEY_NAME = "LLAMA_CPP_API_KEY" 

async def run_deep_research_test_with_deep_dive():
    print("--- Genie Tooling: DeepResearchAgent E2E Test (Llama.cpp Server) ---")
    load_dotenv() 

    print(f"Using Llama.cpp server at: {LLAMA_CPP_BASE_URL} with model alias {LLAMA_CPP_MODEL_ALIAS}")
    if os.getenv(LLAMA_CPP_API_KEY_NAME):
        print(f"Using API Key from environment variable: {LLAMA_CPP_API_KEY_NAME}")

    app_features = FeatureSettings(
        llm="llama_cpp", # Target Llama.cpp server
        llm_llama_cpp_base_url=LLAMA_CPP_BASE_URL,
        llm_llama_cpp_model_name=LLAMA_CPP_MODEL_ALIAS,
        llm_llama_cpp_api_key_name=LLAMA_CPP_API_KEY_NAME if os.getenv(LLAMA_CPP_API_KEY_NAME) else None,
        
        default_llm_output_parser="pydantic_output_parser",
        prompt_registry="file_system_prompt_registry",
        prompt_template_engine="jinja2_chat_formatter",
        tool_lookup="embedding", 
        observability_tracer="console_tracer", 
    )

    app_config = MiddlewareConfig(
        features=app_features,
        tool_configurations={
            "duckduckgo_search_tool_v1": {},
            "arxiv_search_tool": {},
            "web_page_scraper_tool_v1": {},
            "arxiv_pdf_extractor_tool_v1": {},
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie facade...")
        genie = await Genie.create(config=app_config)
        print("Genie facade initialized successfully!")

        dra_agent_config = {
            "web_search_tool_config": {
                "tool_id": "duckduckgo_search_tool_v1", "num_results": 2,
                "description_for_llm": "Searches the web using DuckDuckGo for general information."
            },
            "arxiv_search_tool_config": {
                "tool_id": "arxiv_search_tool", "max_results": 1,
                "description_for_llm": "Searches ArXiv for academic papers."
            },
            "web_page_scraper_tool_config": {
                "tool_id": "web_page_scraper_tool_v1",
                "description_for_llm": "Fetches and extracts text from a web URL."
            },
            "arxiv_pdf_extractor_tool_config": {
                "tool_id": "arxiv_pdf_extractor_tool_v1",
                "description_for_llm": "Downloads and extracts text from an ArXiv PDF."
            },
            "rag_search_tool_config": { 
                "collection_name": "e26_research_docs", "top_k": 1,
                "tool_id": "internal_rag_search_tool", 
                "description_for_llm": "Searches an internal knowledge base."
            },
            "max_total_gathering_cycles": 3, 
            "max_sub_questions_per_plan": 2,  
            "max_plan_refinement_cycles": 1,  
            "iterative_synthesis_interval": 2, 
        }
        research_agent = DeepResearchAgent(genie=genie, agent_config=dra_agent_config)
        print("DeepResearchAgent initialized with deep retrieval capabilities.")

        query = "Summarize the key contributions of the 'Transformer' architecture as introduced in 'Attention Is All You Need' (ArXiv:1706.03762) and find one recent (2023-2024) application of Transformers in natural language processing."

        print(f"\nConducting research for query: '{query}'")
        agent_output = await research_agent.run(goal=query)

        print("\n\n--- Deep Research Agent Output ---")
        print(f"Status: {agent_output['status']}")
        print(f"\nFinal Report:\n{agent_output['output']}")

        print("\n--- Research Plan (Final) ---")
        if agent_output.get("plan"):
            print(json.dumps(agent_output["plan"], indent=2))
        else:
            print("No plan information available in output.")

        print("\n--- Research History (Snippets - first 2 if any) ---")
        if agent_output.get("history"):
            snippet_count = 0
            for item in agent_output["history"]: 
                if item.get("type") == "snippet_gathered":
                    print(f"  Source Type: {item.get('snippet_source_type', item.get('source_type'))}, Identifier: {item.get('snippet_source', item.get('source_identifier'))}")
                    print(f"  Sub-query: {item.get('sub_query')}")
                    print(f"  Extracted Info (first 150 chars): {str(item.get('extracted_info', ''))[:150]}...")
                    snippet_count += 1
                    if snippet_count >= 2:
                        break
            if snippet_count == 0:
                print("  No snippets were gathered in this run (check agent DEBUG logs for details).")

    except Exception as e:
        print(f"\nE2E Test FAILED: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if genie:
            print("\nTearing down Genie facade...")
            await genie.close()
            print("Genie facade torn down.")

if __name__ == "__main__":
    asyncio.run(run_deep_research_test_with_deep_dive())
