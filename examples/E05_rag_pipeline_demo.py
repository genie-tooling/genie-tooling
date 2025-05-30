# examples/rag_pipeline_demo/main.py
"""
Example: RAG Pipeline Demo using Genie Facade (Updated)
-------------------------------------------------------
This example demonstrates setting up and using a RAG pipeline to index
local text files and perform similarity searches using the Genie facade
and FeatureSettings for simplified configuration.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
   You'll need dependencies for local RAG, e.g., sentence-transformers, faiss-cpu.
2. Create the data files (if they don't exist from previous runs):
   - examples/rag_pipeline_demo/data/doc1.txt (Content: The quick brown fox.)
   - examples/rag_pipeline_demo/data/doc2.txt (Content: Genie Tooling is great.)
   - examples/rag_pipeline_demo/data/doc3.txt (Content: Python powers AI.)
3. Run from the root of the project:
   `poetry run python examples/rag_pipeline_demo/main.py`

The demo will:
- Initialize the Genie facade using FeatureSettings for RAG components.
- Index documents from the 'examples/rag_pipeline_demo/data/' directory.
- Perform a search query against the indexed documents.
- Print the search results.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from genie_tooling.config.features import FeatureSettings  # Import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie

# Genie uses EnvironmentKeyProvider by default if no key_provider_instance is given
# and key_provider_id is not set or set to "env_keys".
# Local RAG components (ST embedder, FAISS) typically don't need API keys.

async def main():
    print("--- RAG Pipeline Demo using Genie Facade (FeatureSettings) ---")

    current_file_dir = Path(__file__).parent
    data_dir = current_file_dir / "data"
    data_dir.mkdir(exist_ok=True) # Ensure data directory exists

    # Create dummy files if they don't exist for the demo
    if not (data_dir / "doc1.txt").exists(): (data_dir / "doc1.txt").write_text("The quick brown fox jumps over the lazy dog.")
    if not (data_dir / "doc2.txt").exists(): (data_dir / "doc2.txt").write_text("Genie Tooling provides a hyper-pluggable middleware for AI agents.")
    if not (data_dir / "doc3.txt").exists(): (data_dir / "doc3.txt").write_text("Python is a versatile programming language for AI development.")


    # 1. Configure Middleware using FeatureSettings for RAG
    app_config = MiddlewareConfig(
        features=FeatureSettings(
            # LLM and Command Processor not strictly needed for this RAG-focused demo
            llm="none",
            command_processor="none",
            # Configure RAG components
            rag_embedder="sentence_transformer", # Use SentenceTransformerEmbedder
            # rag_embedder_st_model_name="all-MiniLM-L6-v2", # Default, can be omitted
            rag_vector_store="faiss", # Use FAISSVectorStore
            # For FAISS, path is handled by the plugin's default logic if not specified here
            # or in vector_store_configurations.
        ),
        # Example: If you wanted to specify a custom path for FAISS:
        # vector_store_configurations={
        #     "faiss_vector_store_v1": { # Canonical ID
        #         "index_file_path": str(current_file_dir / "my_faiss_index.faiss"),
        #         "doc_store_file_path": str(current_file_dir / "my_faiss_docs.pkl")
        #     }
        # }
    )

    genie: Optional[Genie] = None
    try:
        # 2. Initialize Genie
        print("\nInitializing Genie facade...")
        genie = await Genie.create(config=app_config) # Uses default EnvironmentKeyProvider
        print("Genie facade initialized.")

        # 3. Indexing Data
        collection_name_for_demo = "my_local_rag_collection"
        print(f"\nIndexing documents from '{data_dir}' into collection '{collection_name_for_demo}'...")

        # genie.rag.index_directory will use the RAG components configured by FeatureSettings
        index_result = await genie.rag.index_directory(
            str(data_dir),
            collection_name=collection_name_for_demo
        )
        print(f"Indexing result: {index_result}")
        if index_result.get("status") != "success":
            print(f"ERROR: Indexing failed: {index_result.get('message')}")
            return

        # 4. Performing a Search
        query = "What is Genie Tooling?"
        print(f"\nPerforming search for query: '{query}' in collection '{collection_name_for_demo}'")

        search_results = await genie.rag.search(
            query,
            collection_name=collection_name_for_demo,
            top_k=2
        )

        if not search_results:
            print("No search results found.")
        else:
            print("\nSearch Results:")
            for i, result_chunk in enumerate(search_results):
                print(f"  --- Result {i+1} (Score: {result_chunk.score:.4f}, ID: {result_chunk.id}) ---")
                print(f"  Content: {result_chunk.content[:200]}...")
                print(f"  Metadata: {result_chunk.metadata}")
                print("  ------------------------------------")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        logging.exception("Error details:")
    finally:
        if genie:
            print("\n--- Tearing down Genie facade ---")
            await genie.close()
            print("Genie facade teardown complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(main())
