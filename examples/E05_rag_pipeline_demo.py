# examples/E05_rag_pipeline_demo.py
"""
Example: RAG Pipeline Demo using Genie Facade (Updated)
-------------------------------------------------------
This example demonstrates setting up and using a RAG pipeline to index
local text files and perform similarity searches using the Genie facade
and FeatureSettings for simplified configuration.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
   You'll need dependencies for local RAG, e.g., sentence-transformers, faiss-cpu.
2. The script will create dummy data files in examples/data/ if they don't exist.
3. Run from the root of the project:
   `poetry run python examples/E05_rag_pipeline_demo.py`

The demo will:
- Initialize the Genie facade using FeatureSettings for RAG components.
- Index documents from the 'examples/data/' directory.
- Perform a search query against the indexed documents.
- Print the search results.
"""
import asyncio
import logging
import shutil # For cleanup
from pathlib import Path
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie

async def main():
    print("--- RAG Pipeline Demo using Genie Facade (FeatureSettings) ---")

    current_file_dir = Path(__file__).parent
    data_dir = current_file_dir / "data"
    data_dir.mkdir(exist_ok=True) # Ensure data directory exists

    # Create dummy files if they don't exist for the demo
    doc1_path = data_dir / "doc1.txt"
    doc2_path = data_dir / "doc2.txt"
    doc3_path = data_dir / "doc3.txt"

    if not doc1_path.exists(): doc1_path.write_text("The quick brown fox jumps over the lazy dog.\nLarge language models are transforming AI.")
    if not doc2_path.exists(): doc2_path.write_text("Genie Tooling provides a hyper-pluggable middleware.\nRetrieval Augmented Generation enhances LLM responses.")
    if not doc3_path.exists(): doc3_path.write_text("Python is a versatile programming language.\nAsync programming is key for I/O bound tasks.")


    # 1. Configure Middleware using FeatureSettings for RAG
    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="none",
            command_processor="none",
            rag_embedder="sentence_transformer", 
            rag_vector_store="faiss", 
        )
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie facade...")
        genie = await Genie.create(config=app_config)
        print("Genie facade initialized.")

        collection_name_for_demo = "my_local_rag_collection_e05"
        print(f"\nIndexing documents from '{data_dir}' into collection '{collection_name_for_demo}'...")

        index_result = await genie.rag.index_directory(
            str(data_dir),
            collection_name=collection_name_for_demo
        )
        print(f"Indexing result: {index_result}")
        if index_result.get("status") != "success":
            print(f"ERROR: Indexing failed: {index_result.get('message')}")
            return

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
        
        # Clean up dummy files (optional, but good for repeated test runs)
        # for p in [doc1_path, doc2_path, doc3_path]:
        #     p.unlink(missing_ok=True)
        # if data_dir.exists() and not any(data_dir.iterdir()): # Only remove if empty
        #     data_dir.rmdir()
        # For simplicity, we'll leave the data dir and files for now.
        # To fully clean up FAISS index files if they were persisted to default location:
        # default_faiss_path = Path("./.genie_data/faiss")
        # if default_faiss_path.exists():
        #     shutil.rmtree(default_faiss_path, ignore_errors=True)
        #     print(f"Cleaned up FAISS data at {default_faiss_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(main())
