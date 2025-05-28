# examples/rag_pipeline_demo/main.py
"""
Example: RAG Pipeline Demo using Genie Facade
---------------------------------------------
This example demonstrates setting up and using a RAG pipeline to index
local text files and perform similarity searches using the Genie facade.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
   You'll need dependencies for local RAG, e.g., sentence-transformers, faiss-cpu.
2. Create the data files:
   - examples/rag_pipeline_demo/data/doc1.txt
   - examples/rag_pipeline_demo/data/doc2.txt
   - examples/rag_pipeline_demo/data/doc3.txt
   (Content for these files is provided in the prompt response)
3. Run from the root of the project:
   `poetry run python examples/rag_pipeline_demo/main.py`

The demo will:
- Initialize the Genie facade.
- Index documents from the 'examples/rag_pipeline_demo/data/' directory.
- Perform a search query against the indexed documents.
- Print the search results.
- Clean up by tearing down the Genie facade.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.types import Plugin as CorePluginType # For KeyProvider
from genie_tooling.genie import Genie
from genie_tooling.security.key_provider import KeyProvider

# --- 1. Basic KeyProvider Implementation (Application-Side) ---
# Even if not directly used by this specific RAG setup (local embedders),
# Genie.create expects it, and some embedders might need it.
class EnvironmentKeyProvider(KeyProvider, CorePluginType):
    plugin_id: str = "env_key_provider_rag_demo_v1"
    description: str = "Provides API keys from environment variables for RAG demo."

    async def get_key(self, key_name: str) -> Optional[str]:
        print(f"[KeyProvider - RAG Demo] Requesting key: {key_name}")
        return os.environ.get(key_name)

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        print(f"[{self.plugin_id}] Setup complete.")

    async def teardown(self) -> None:
        print(f"[{self.plugin_id}] Teardown complete.")


async def main():
    print("--- RAG Pipeline Demo using Genie Facade ---")

    # --- Configuration ---
    # Relative path to the data directory for this example
    current_file_dir = Path(__file__).parent
    data_dir = current_file_dir / "data"
    if not data_dir.exists():
        print(f"ERROR: Data directory '{data_dir}' not found. Please create it and add sample files.")
        return

    # For this demo, we'll use local RAG components which don't require API keys typically.
    # However, Genie.create requires a KeyProvider.
    # If you were using an OpenAI embedder, you'd set OPENAI_API_KEY environment variable.
    key_provider_instance = EnvironmentKeyProvider()
    
    # Middleware config - largely relying on defaults for RAG components
    # The RAGInterface in Genie will use its own defaults or these if specified.
    # For local RAG, defaults like "sentence_transformer_embedder_v1" and "faiss_vector_store_v1" are good.
    middleware_cfg = MiddlewareConfig(
        # plugin_dev_dirs=["path/to/my/custom_rag_plugins"], # If you had custom plugins
        default_rag_embedder_id="sentence_transformer_embedder_v1", # Example: ensure local embedder
        default_rag_vector_store_id="faiss_vector_store_v1", # Example: ensure local vector store
        # default_rag_loader_id="file_system_loader_v1", # Default in RAGInterface already
        # default_rag_splitter_id="character_recursive_text_splitter_v1", # Default in RAGInterface
    )

    genie: Optional[Genie] = None
    try:
        # --- Initialize Genie ---
        print("\nInitializing Genie facade...")
        genie = await Genie.create(
            config=middleware_cfg,
            key_provider_instance=key_provider_instance
        )
        print("Genie facade initialized.")

        # --- Indexing Data ---
        collection_name_for_demo = "my_rag_demo_collection"
        print(f"\nIndexing documents from '{data_dir}' into collection '{collection_name_for_demo}'...")
        
        # You can override component IDs and their configs here if needed:
        # index_result = await genie.rag.index_directory(
        #     str(data_dir),
        #     collection_name=collection_name_for_demo,
        #     # loader_id="custom_loader", loader_config={"param": "value"},
        #     # embedder_id="openai_embedding_generator_v1" # If API key is set
        # )
        index_result = await genie.rag.index_directory(
            str(data_dir), 
            collection_name=collection_name_for_demo
        )

        print(f"Indexing result: {index_result}")
        if index_result.get("status") != "success":
            print(f"ERROR: Indexing failed: {index_result.get('message')}")
            return

        # --- Performing a Search ---
        query = "What is Genie Tooling?"
        print(f"\nPerforming search for query: '{query}' in collection '{collection_name_for_demo}'")
        
        search_results = await genie.rag.search(
            query,
            collection_name=collection_name_for_demo,
            top_k=2
            # retriever_id="custom_retriever", retriever_config={...}
        )

        if not search_results:
            print("No search results found.")
        else:
            print("\nSearch Results:")
            for i, result_chunk in enumerate(search_results):
                print(f"  --- Result {i+1} (Score: {result_chunk.score:.4f}, ID: {result_chunk.id}) ---")
                print(f"  Content: {result_chunk.content[:200]}...") # Print snippet
                print(f"  Metadata: {result_chunk.metadata}")
                print("  ------------------------------------")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if genie:
            print("\n--- Tearing down Genie facade ---")
            await genie.close()
            print("Genie facade teardown complete.")

if __name__ == "__main__":
    asyncio.run(main())