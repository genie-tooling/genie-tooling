import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from genie_tooling import tool
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.tools.impl.community_google_search_tool import (
    community_google_search,
)

logger = logging.getLogger(__name__)

try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None
    TRAFILATURA_AVAILABLE = False


# --- Configuration ---
HISTORICAL_LLM_DATA_CSV = """model_name,release_date,param_size_category,release_type_tags,family_keywords,arxiv_mentions_post_3m_increase_pct
Llama-2-7B,2023-07-18,medium,"instruct_tuned;open_weights","Llama;LLaMA",30.5
Mistral-7B-Instruct-v0.1,2023-09-27,small,"instruct_tuned;open_weights","Mistral",45.2
"""
HISTORICAL_CSV_PATH = Path("./temp_oss_llm_history_cli_test.csv")
HISTORICAL_RAG_COLLECTION_NAME = "historical_oss_llm_impact_cli_test"

# --- Tool Definitions ---

# --- REVISED AND ROBUST TOOL IMPLEMENTATION ---
@tool
async def web_page_content_extractor(url: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    WebScraper: Fetches the full text content from a given web URL. It includes retries for transient errors.
    Use this tool *after* a search tool provides a relevant URL.
    Output: {"url": "str", "title": "str", "content": "str", "error": "Optional[str]"}
    Args:
        url (str): The URL of the web page to scrape.
        context (Dict[str, Any]): Invocation context.
    """
    if not url or not isinstance(url, str):
        return {"error": "URL must be a non-empty string."}

    normalized_url = url
    if not url.lower().startswith(("http://", "https://")):
        normalized_url = f"https://{url}"
        logger.debug("web_page_content_extractor: Prepended 'https://' to URL.")

    # More robust headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "DNT": "1",
        "Sec-GPC": "1",
        "Upgrade-Insecure-Requests": "1",
    }

    current_context = context if context is not None else {}
    genie_instance: Optional[Genie] = current_context.get("genie_framework_instance")
    correlation_id = current_context.get("correlation_id", str(asyncio.current_task()))

    max_retries = 2
    initial_delay = 1.0

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=headers) as client:
        for attempt in range(max_retries + 1):
            try:
                if genie_instance:
                    await genie_instance.observability.trace_event(
                        "tool.web_page_content_extractor.fetch.start",
                        {"url": normalized_url, "attempt": attempt + 1},
                        "web_page_content_extractor",
                        correlation_id,
                    )
                response = await client.get(normalized_url)
                response.raise_for_status()

                html_content = response.text
                page_title = urlparse(normalized_url).netloc
                if TRAFILATURA_AVAILABLE and trafilatura:
                    extracted_text = trafilatura.extract(
                        html_content, include_comments=False, include_tables=True
                    )
                    text_content = extracted_text if extracted_text else ""
                    if BeautifulSoup:
                        soup_title = BeautifulSoup(html_content, "html.parser")
                        if soup_title.title and soup_title.title.string:
                            page_title = soup_title.title.string.strip()
                elif BeautifulSoup:
                    soup = BeautifulSoup(html_content, "html.parser")
                    if soup.title and soup.title.string:
                        page_title = soup.title.string.strip()
                    for el_type in [
                        "script", "style", "nav", "footer", "aside", "header", "form", "button", "input",
                        "textarea", "select", "option", "iframe", "embed", "object", "video", "audio",
                        "source", "track", "canvas", "map", "area", "svg", "math",
                    ]:
                        for el in soup(el_type):
                            el.decompose()
                    text_content = soup.get_text(separator=" ", strip=True)
                else:
                    text_content = html_content

                limited_content = text_content[:15000]
                if genie_instance:
                    await genie_instance.observability.trace_event(
                        "tool.web_page_content_extractor.fetch.success",
                        {"url": normalized_url, "title_found": bool(page_title), "content_length": len(limited_content)},
                        "web_page_content_extractor",
                        correlation_id,
                    )
                return {"url": normalized_url, "title": page_title, "content": limited_content, "error": None}

            except httpx.HTTPStatusError as e:
                # For client errors (4xx), don't retry. For server errors (5xx), do.
                if 400 <= e.response.status_code < 500:
                    logger.error(f"Client error {e.response.status_code} fetching {normalized_url}. Not retrying.")
                    return {"url": normalized_url, "title": None, "content": None, "error": str(e)}
                elif attempt < max_retries:
                    delay = initial_delay * (2**attempt)
                    logger.warning(
                        f"Server error {e.response.status_code} fetching {normalized_url}. "
                        f"Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Server error fetching {normalized_url} after {max_retries} retries.")
                    return {"url": normalized_url, "title": None, "content": None, "error": str(e)}
            except httpx.RequestError as e:
                if attempt < max_retries:
                    delay = initial_delay * (2**attempt)
                    logger.warning(
                        f"Request error fetching {normalized_url}: {e}. Retrying in {delay:.2f}s... "
                        f"(Attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request error fetching {normalized_url} after {max_retries} retries.")
                    return {"url": normalized_url, "title": None, "content": None, "error": str(e)}
            except Exception as e:
                logger.error(f"Unexpected error fetching/parsing {normalized_url}: {e}", exc_info=True)
                return {"url": normalized_url, "title": None, "content": None, "error": f"Unexpected error: {e!s}"}

    return {"url": normalized_url, "title": None, "content": None, "error": "Exhausted all retries without success."}
# --- END OF REVISED TOOL ---


@tool
async def custom_text_parameter_extractor(
    text_content: str,
    parameter_names: List[str],
    regex_patterns: List[str],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    DataExtractor: Extracts specific named pieces of information (e.g., scores, numbers, names, dates)
    from a larger block of text using regular expressions. Use this *after* obtaining full text content.
    `parameter_names` and `regex_patterns` are parallel lists; each `regex_patterns[i]` is used to find the value for `parameter_names[i]`.
    Output: A dictionary where keys are from `parameter_names`, and values are the extracted strings/numbers, or None if not found/error.
    Args:
        text_content (str): The block of text to extract parameters from.
        parameter_names (List[str]): A list of output keys for the extracted values.
        regex_patterns (List[str]): A list of Python regex patterns. Each pattern must have one capturing group `(...)` for the value.
        context (Optional[Dict[str, Any]]): Invocation context.
    """
    if len(parameter_names) != len(regex_patterns):
        err_msg = f"Mismatched argument lengths: {len(parameter_names)} names provided, but {len(regex_patterns)} patterns. They must be parallel lists."
        logger.warning(f"{custom_text_parameter_extractor._tool_metadata_['identifier']}: {err_msg}")
        return {"error": err_msg}

    extracted_values: Dict[str, Any] = {}
    if not text_content or not isinstance(text_content, str):
        logger.warning(f"{custom_text_parameter_extractor._tool_metadata_['identifier']}: Input 'text_content' is empty or not a string.")
        for name in parameter_names:
            extracted_values[name] = None
        return extracted_values

    for name, pattern_str in zip(parameter_names, regex_patterns, strict=False):
        if not name or not pattern_str:
            unnamed_key = f"extraction_error_param_{len(extracted_values)}"
            extracted_values[name or unnamed_key] = None
            logger.warning(f"Invalid parameter specification. Name: '{name}', Pattern: '{pattern_str}'.")
            continue

        try:
            match = re.search(pattern_str, text_content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                value_str = match.group(1) if match.groups() else match.group(0)
                value_str = value_str.strip()
                if value_str.replace(".", "", 1).replace("-","",1).isdigit():
                    if "." in value_str:
                        try:
                            extracted_values[name] = float(value_str)
                        except ValueError:
                            extracted_values[name] = value_str
                    else:
                        try:
                            extracted_values[name] = int(value_str)
                        except ValueError:
                            extracted_values[name] = value_str
                else:
                    extracted_values[name] = value_str
            else:
                extracted_values[name] = None
        except re.error as e_re:
            logger.error(f"{custom_text_parameter_extractor._tool_metadata_['identifier']}: Regex error for parameter '{name}' with pattern '{pattern_str}': {e_re}", exc_info=False)
            extracted_values[name] = None
        except Exception as e:
            logger.error(f"{custom_text_parameter_extractor._tool_metadata_['identifier']}: Unexpected error extracting parameter '{name}': {e}", exc_info=True)
            extracted_values[name] = None

    logger.debug(f"{custom_text_parameter_extractor._tool_metadata_['identifier']}: Extracted values: {extracted_values}")
    return extracted_values

@tool
async def oss_model_release_impact_analyzer(model_family_keywords: List[str], release_type_tags: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes the typical impact of open-source LLM releases on academic paper mentions (ArXiv),
    based on historical data retrieved via RAG.
    Output: {"average_arxiv_mention_increase_pct": Optional[float], "num_comparable_releases_analyzed": int, "notes": str, "error": Optional[str]}
    Args:
        model_family_keywords (List[str]): Keywords for model families (e.g., ["Llama", "Mistral"]).
        release_type_tags (List[str]): Tags describing release types (e.g., ["instruct_tuned", "open_weights"]).
        context (Dict[str, Any]): Invocation context.
    """
    genie_instance: Optional[Genie] = context.get("genie_framework_instance")
    if not genie_instance:
        logger.error("OSSAnalyzer: Genie instance not found.")
        return {"error": "Genie instance missing."}
    query_parts = ["Historical impact of open-source LLM release"]
    if model_family_keywords:
        query_parts.append(f"for model families like {', '.join(model_family_keywords)}")
    if release_type_tags:
        query_parts.append(f"with release types such as {', '.join(release_type_tags)}")
    query_parts.append("on ArXiv academic paper mentions")
    search_query = " ".join(query_parts)
    try:
        rag_results = await genie_instance.rag.search(search_query, collection_name=HISTORICAL_RAG_COLLECTION_NAME, top_k=10)
        if not rag_results:
            return {"average_arxiv_mention_increase_pct": None, "num_comparable_releases_analyzed": 0, "notes": "No comparable releases found.", "error": None}
        parsed_impacts = []
        for chunk in rag_results:
            parts = chunk.content.split(",")
            if len(parts) >= 6:
                matches_family = not model_family_keywords or any(kw.lower() in parts[4].lower() for kw in model_family_keywords)
                matches_type = not release_type_tags or any(tag.lower() in parts[3].lower() for tag in release_type_tags)
                if matches_family and matches_type:
                    try:
                        parsed_impacts.append(float(parts[-1].strip()))
                    except ValueError:
                        pass
        if not parsed_impacts:
            return {"average_arxiv_mention_increase_pct": None, "num_comparable_releases_analyzed": 0, "notes": "No matching data parsed.", "error": None}
        avg_impact = sum(parsed_impacts) / len(parsed_impacts)
        return {"average_arxiv_mention_increase_pct": round(avg_impact, 2), "num_comparable_releases_analyzed": len(parsed_impacts), "notes": f"Based on {len(parsed_impacts)} releases.", "error": None}
    except Exception as e:
        logger.error(f"OSSAnalyzer: Error: {e}", exc_info=True)
        return {"error": f"Failed: {e!s}"}

# --- Main Execution Function ---
async def main():
    parser = argparse.ArgumentParser(
        description="Run the advanced ReWOO agent test with a configurable LLM provider.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=["ollama", "llama_cpp_internal", "gemini"],
        help="The LLM provider to use."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="The model name (for ollama/gemini) or path to the GGUF file (for llama_cpp_internal)."
    )
    parser.add_argument(
        "--ctx-length",
        type=int,
        default=8096,
        help="Context length for the model (primarily for llama_cpp_internal). Defaults to 8096."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    logging.getLogger("genie_tooling.command_processors.impl.rewoo_processor").setLevel(logging.DEBUG)
    load_dotenv()

    print("--- Genie Tooling: Advanced ReWOO Agent Test ---")
    print(f"Provider: {args.provider}, Model: {args.model}, Context Length: {args.ctx_length}")

    feature_settings_args = {
        "command_processor": "rewoo",
        "default_llm_output_parser": "pydantic_output_parser",
        "tool_lookup": "hybrid",
        "tool_lookup_embedder_id_alias": "st_embedder",
        "rag_embedder": "sentence_transformer",
        "rag_vector_store": "faiss",
        "prompt_template_engine": "jinja2_chat_formatter", # Ensure this is set
        "observability_tracer": "console_tracer",
        "logging_adapter": "pyvider_log_adapter",
        "token_usage_recorder": "in_memory_token_recorder"
    }

    if args.provider == "ollama":
        feature_settings_args["llm"] = "ollama"
        feature_settings_args["llm_ollama_model_name"] = args.model
    elif args.provider == "gemini":
        feature_settings_args["llm"] = "gemini"
        feature_settings_args["llm_gemini_model_name"] = args.model
    elif args.provider == "llama_cpp_internal":
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"ERROR: Model path for llama_cpp_internal does not exist: {args.model}")
            return
        feature_settings_args["llm"] = "llama_cpp_internal"
        feature_settings_args["llm_llama_cpp_internal_model_path"] = str(model_path.resolve())
        feature_settings_args["llm_llama_cpp_internal_n_ctx"] = args.ctx_length
        feature_settings_args["llm_llama_cpp_internal_n_gpu_layers"] = -1

    app_features = FeatureSettings(**feature_settings_args)

    HISTORICAL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORICAL_CSV_PATH.write_text(HISTORICAL_LLM_DATA_CSV)

    app_config = MiddlewareConfig(
        features=app_features,
        auto_enable_registered_tools=False,
        tool_configurations={
            "intelligent_search_aggregator_v1": {
                "google_search_tool_id": "community_google_search",
                "arxiv_search_tool_id": "arxiv_search_tool",
                "embedder_id": "sentence_transformer_embedder_v1"
            },
            # *** REFACTORED CONFIGURATION ***
            "content_retriever_tool_v1": {
                "pdf_extractor_id": "pdf_text_extractor_tool_v1",
                "web_scraper_id": "web_page_scraper_tool_v1"
            },
            # Enable all required tools
            "community_google_search": {},
            "arxiv_search_tool": {},
            "pdf_text_extractor_tool_v1": {},
            "web_page_scraper_tool_v1": {}, # This tool is now explicitly enabled
            "calculator_tool": {},
            "custom_text_parameter_extractor": {},
            "discussion_sentiment_summarizer": {},
            "oss_model_release_impact_analyzer": {},
        },
        embedding_generator_configurations={
             "sentence_transformer_embedder_v1": {"model_name": "all-MiniLM-L6-v2"}
        },
        log_adapter_configurations={
            "pyvider_telemetry_log_adapter_v1": {"default_level": "DEBUG", "enable_key_name_redaction": False}
        },
        document_loader_configurations={
            "file_system_loader_v1": {"glob_pattern": HISTORICAL_CSV_PATH.name}
        },
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie...")
        genie = await Genie.create(config=app_config)

        await genie.register_tool_functions([
            community_google_search,
            web_page_content_extractor,
            custom_text_parameter_extractor,
            oss_model_release_impact_analyzer,
        ])
        print("Decorated tools registered.")

        await genie.rag.index_directory(
            path=str(HISTORICAL_CSV_PATH.parent.resolve()),
            collection_name=HISTORICAL_RAG_COLLECTION_NAME,
            loader_id="file_system_loader_v1",
            loader_config={"glob_pattern": HISTORICAL_CSV_PATH.name}
        )
        print(f"Indexed historical LLM release data into RAG collection: '{HISTORICAL_RAG_COLLECTION_NAME}'.")

        goal = """Explain the function of CRISPR-Cas9. Specifically, identify the roles of the Cas9 protein and the guide RNA (gRNA), and find one recent (2023-2024) application of this technology mentioned in a scientific paper on ArXiv."""

        print(f"\n--- Starting ReWOO Agent for Goal ---\n{goal}\n------------------------------------")

        context_for_tool_calls = {"initial_goal": goal}
        agent_result = await genie.run_command(goal, context_for_tools=context_for_tool_calls)

        print("\n--- ReWOO Agent Final Report ---")
        if agent_result.get("error"):
            print(f"Error: {agent_result['error']}")
        if agent_result.get("final_answer"):
            print(f"Final Answer:\n{agent_result['final_answer']}")

        print("\n--- ReWOO Agent Thought Process (Plan & Evidence if available) ---")
        if agent_result.get("llm_thought_process"):
            try:
                thought_data = json.loads(agent_result["llm_thought_process"])
                print(json.dumps(thought_data, indent=2, default=str))
            except (json.JSONDecodeError, TypeError):
                print(str(agent_result["llm_thought_process"]))

        print("\n--- Token Usage Summary ---")
        usage_summary = await genie.usage.get_summary()
        print(json.dumps(usage_summary, indent=2))

    except Exception as e:
        print(f"\nAn unexpected error occurred in the main script: {e}")
        logging.exception("Main script E2E test error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie facade torn down.")
        if HISTORICAL_CSV_PATH.exists():
            HISTORICAL_CSV_PATH.unlink()
        historical_parent_dir = HISTORICAL_CSV_PATH.parent
        if historical_parent_dir.exists() and not any(historical_parent_dir.iterdir()):
            try:
                historical_parent_dir.rmdir()
            except OSError as e_rmdir:
                logging.warning(f"Could not remove temp dir {historical_parent_dir}: {e_rmdir}")

if __name__ == "__main__":
    asyncio.run(main())
