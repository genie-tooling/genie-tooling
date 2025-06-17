### run_math_bot.py
import argparse
import asyncio
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from genie_tooling.agents.math_proof_assistant_agent import MathProofAssistantAgent
from genie_tooling.config import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings
from genie_tooling.genie import Genie, RAGInterface

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO)
# Quieter library logs for a cleaner TUI experience
logging.getLogger("genie_tooling").setLevel(logging.WARNING)
# Enable debug logs for the agent itself to see its state transitions
logging.getLogger("src.genie_tooling.agents.math_proof_assistant_agent").setLevel(
    logging.INFO
)
logger = logging.getLogger(__name__)


# --- Monkey-patch a convenience method onto RAGInterface ---
# This is a clean way to add the helper without modifying the library source directly.
async def index_text_impl(
    self: RAGInterface, text: str, collection_name: str, metadata: Optional[Dict] = None
) -> Dict:
    """Convenience method to index a single string of text."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "content.txt"
        temp_file.write_text(text, encoding="utf-8")
        return await self.index_directory(
            path=temp_dir,
            collection_name=collection_name,
            loader_config={"glob_pattern": "*.txt"},
            splitter_config={"chunk_size": 2000, "chunk_overlap": 200},
            embedder_config={"key_provider": self._key_provider},
            vector_store_config={"collection_name": collection_name},
        )


RAGInterface.index_text = index_text_impl
# --- End of monkey-patch ---


async def main(model_path_str: str):
    """Main function to configure and run the MathProofAssistantAgent in a TUI loop."""
    load_dotenv()

    model_path = Path(model_path_str)
    if not model_path.exists():
        print("\n\n!!! CRITICAL ERROR !!!")
        print(f"Model file not found at '{model_path_str}'.")
        print(
            "Please provide a valid path via the --model-path argument "
            "or set the LLAMA_CPP_INTERNAL_MODEL_PATH environment variable."
        )
        return

    # 1. Configure Genie Tooling
    # This configuration enables all tools and subsystems required by the MathProofAssistantAgent.
    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="llama_cpp_internal",
            llm_llama_cpp_internal_model_path=str(model_path.resolve()),
            llm_llama_cpp_internal_n_ctx=8092,  # Using a large context window for complex proofs
            llm_llama_cpp_internal_n_gpu_layers=-1,  # Offload all layers to GPU
            logging_adapter="pyvider_log_adapter",
            tool_lookup="hybrid",
            rag_embedder="sentence_transformer",
            rag_vector_store="chroma",
            prompt_template_engine="jinja2_chat_formatter",
            conversation_state_provider="redis_convo_provider",
            default_llm_output_parser="pydantic_output_parser",
            observability_tracer="console_tracer",
        ),
        auto_enable_registered_tools=False,
        tool_configurations={
            # Tools for MathProofAssistantAgent direct use
            "generic_code_execution_tool": {"executor_id": "pysandbox_executor_stub_v1"},
            "symbolic_math_tool": {},
            # Tools that the DeepResearchAgent sub-system needs enabled
            "intelligent_search_aggregator_v1": {},
            "arxiv_search_tool": {},
            "content_retriever_tool_v1": {},
            "custom_text_parameter_extractor": {},
            "community_google_search": {},
            "pdf_text_extractor_tool_v1": {},
            "web_page_scraper_tool_v1": {},
        },
        command_processor_configurations={
            # Config for our "Agent as Processor" bridge for deep research.
            "deep_research_agent_v1": {
                "min_high_quality_sources": 2,
                "max_replanning_loops": 1,
            }
        },
        vector_store_configurations={
            "chromadb_vector_store_v1": {"path": "./math_proof_memory"}
        },
        conversation_state_provider_configurations={
            "redis_conversation_state_v1": {"redis_url": "redis://localhost:6379/1"}
        },
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie framework and Math Proof Assistant...")
        genie = await Genie.create(config=app_config)
        print(f"Genie initialized successfully using model: {model_path.name}")
        print("-" * 50)

        # 2. Get user's initial goal
        print("Welcome to the Math Proof Assistant TUI!")
        print("Please state your primary research goal or theorem to explore.")
        print(
            "Example: I want to explore the potential relationship between the Collatz conjecture "
            "and the core principles of Shor's algorithm."
        )
        
        # --- MODIFICATION START ---
        # Replace standard input() with a robust byte-reading and decoding method
        # to prevent UnicodeDecodeError from pasted text.
        print("\nYour Goal > ", end="", flush=True)
        try:
            initial_goal_bytes = await asyncio.to_thread(sys.stdin.buffer.readline)
            initial_goal = initial_goal_bytes.decode('utf-8', errors='replace').strip()
        except Exception as e:
            logger.critical(f"Failed to read user input: {e}", exc_info=True)
            print(f"\nCRITICAL: Could not read your input due to an error: {e}")
            initial_goal = ""
        # --- MODIFICATION END ---

        if not initial_goal.strip():
            print("No goal provided. Exiting.")
            return

        # 3. Instantiate and run the agent
        # The agent's `run` method contains the main interactive TUI loop.
        math_agent = MathProofAssistantAgent(genie=genie)
        await math_agent.run(initial_goal=initial_goal)

    except Exception as e:
        logger.critical(
            f"A critical error occurred in the main application: {e}", exc_info=True
        )
    finally:
        if genie:
            print("\nShutting down Genie framework...")
            await genie.close()
            print("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Math Proof Assistant Agent in an interactive TUI.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv(
            "LLAMA_CPP_INTERNAL_MODEL_PATH",
            "/path/to/your/model.gguf",  # A placeholder to force user action
        ),
        help="Path to the GGUF model file for the Llama.cpp internal provider.\n"
        "Can also be set with the LLAMA_CPP_INTERNAL_MODEL_PATH environment variable.",
    )

    args = parser.parse_args()

    # Check if the path is still the placeholder
    if "/path/to/your/model.gguf" in args.model_path:
        print("ERROR: Please specify a model path.")
        print("Usage: python run_math_bot.py --model-path /path/to/your/model.gguf")
        print(
            "Or set the 'LLAMA_CPP_INTERNAL_MODEL_PATH' environment variable."
        )
    else:
        asyncio.run(main(args.model_path))