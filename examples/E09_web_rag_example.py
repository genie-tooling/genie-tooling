# examples/E09_web_rag_example.py
"""
Example: RAG with WebPageLoader using Genie Facade
--------------------------------------------------
This example demonstrates indexing content from a web page and performing
a similarity search using the Genie facade and FeatureSettings.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
   You'll need dependencies for web loading (beautifulsoup4, trafilatura),
   local RAG (sentence-transformers, faiss-cpu).
2. Run from the root of the project:
   `poetry run python examples/E09_web_rag_example.py`
"""
import asyncio
import logging
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def run_web_rag_demo():
    print("--- Web Page RAG Demo ---")

    # URL to index (Python's Getting Started page as an example)
    web_page_url = "https://www.python.org/about/gettingstarted/"
    collection_name = "python_docs_web_collection_e09"

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="none", # Not needed for RAG indexing/search focus
            command_processor="none",

            rag_loader="web_page", # Use WebPageLoader by default
            rag_embedder="sentence_transformer",
            rag_vector_store="faiss", # In-memory FAISS for this demo
        ),
        # Optionally configure WebPageLoader (e.g., to use trafilatura)
        document_loader_configurations={
            "web_page_loader_v1": { # Canonical ID
                "use_trafilatura": True # Attempt to use trafilatura for better content extraction
            }
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie for Web RAG...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        # Index the web page
        print(f"\nIndexing web page: {web_page_url} into collection '{collection_name}'...")
        index_result = await genie.rag.index_web_page(
            web_page_url,
            collection_name=collection_name
        )
        print(f"Indexing result: {index_result}")
        if index_result.get("status") != "success":
            print(f"ERROR: Indexing failed: {index_result.get('message')}")
            return

        # Perform a search
        query = "What are Python libraries?"
        print(f"\nPerforming search for: '{query}' in '{collection_name}'")
        search_results = await genie.rag.search(
            query,
            collection_name=collection_name,
            top_k=2
        )

        if not search_results:
            print("No search results found.")
        else:
            print("\nSearch Results:")
            for i, chunk in enumerate(search_results):
                print(f"  --- Result {i+1} (Score: {chunk.score:.4f}) ---")
                print(f"  Content: {chunk.content[:300]}...") # Print snippet
                print(f"  Source: {chunk.metadata.get('url')}")
                print("  ------------------------------------")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Web RAG demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG) # For detailed logs
    asyncio.run(run_web_rag_demo())
