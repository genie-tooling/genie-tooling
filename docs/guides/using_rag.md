# Using Retrieval Augmented Generation (RAG)

Genie Tooling provides a flexible RAG system through the `genie.rag` interface, allowing you to index data from various sources and perform semantic searches.

## Core RAG Operations via `genie.rag`

The `genie.rag` interface simplifies common RAG tasks:

*   **`genie.rag.index_directory(path, collection_name, ...)`**: Indexes all supported files within a local directory.
*   **`genie.rag.index_web_page(url, collection_name, ...)`**: Fetches content from a web URL, extracts text, and indexes it.
*   **`genie.rag.search(query, collection_name, top_k, ...)`**: Performs a semantic search against an indexed collection.

## Configuring RAG Components

RAG components (Document Loaders, Text Splitters, Embedding Generators, Vector Stores, Retrievers) are configured primarily using `FeatureSettings` within your `MiddlewareConfig`.

```python
import asyncio
from pathlib import Path
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings
from genie_tooling.genie import Genie

async def main():
    app_config = MiddlewareConfig(
        features=FeatureSettings(
            # RAG Embedder (e.g., for creating embeddings of your documents)
            rag_embedder="sentence_transformer", # Alias for "sentence_transformer_embedder_v1"
            # rag_embedder_st_model_name="all-MiniLM-L6-v2", # Default ST model

            # RAG Vector Store (e.g., for storing and searching embeddings)
            rag_vector_store="faiss", # Alias for "faiss_vector_store_v1" (in-memory)
            # Or for persistent ChromaDB:
            # rag_vector_store="chroma",
            # rag_vector_store_chroma_path="./my_rag_db",
            # rag_vector_store_chroma_collection_name="my_documents",

            # Default RAG Loader (used by index_directory if not specified)
            # rag_loader="file_system", # Alias for "file_system_loader_v1" (default for index_directory)
            # Or for index_web_page, it defaults to "web_page_loader_v1" internally.
        )
    )
    genie = await Genie.create(config=app_config)

    # Create a dummy directory and file for indexing
    data_path = Path("./rag_data_example")
    data_path.mkdir(exist_ok=True)
    (data_path / "sample.txt").write_text("Genie makes RAG easy and configurable.")

    # Index the directory
    collection = "my_sample_collection"
    await genie.rag.index_directory(str(data_path), collection_name=collection)
    print(f"Indexed documents from '{data_path}' into '{collection}'.")

    # Search the collection
    query = "How is RAG with Genie?"
    results = await genie.rag.search(query, collection_name=collection, top_k=1)
    if results:
        print(f"Search for '{query}':")
        for res in results:
            print(f"  - Score: {res.score:.4f}, Content: {res.content[:100]}...")
    else:
        print(f"No results found for '{query}'.")
    
    # Clean up dummy data
    (data_path / "sample.txt").unlink()
    data_path.rmdir()

    await genie.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Overriding RAG Component Configurations

You can override configurations for specific RAG component plugins (Document Loaders, Text Splitters, Embedding Generators, Vector Stores, Retrievers) using their respective `*_configurations` dictionaries in `MiddlewareConfig`.

For example, to configure the `WebPageLoader` to use Trafilatura for better content extraction when `genie.rag.index_web_page()` is called (and `rag_loader` feature is set to `"web_page"` or not set, allowing the default):

```python
app_config = MiddlewareConfig(
    features=FeatureSettings(
        rag_loader="web_page", # Explicitly or implicitly uses web_page_loader_v1
        rag_embedder="sentence_transformer",
        rag_vector_store="faiss"
    ),
    document_loader_configurations={
        "web_page_loader_v1": { # Canonical ID of the WebPageLoader
            "use_trafilatura": True,
            # "trafilatura_include_comments": False # Other loader-specific settings
        }
    },
    embedding_generator_configurations={
        "sentence_transformer_embedder_v1": {
            "model_name": "paraphrase-multilingual-MiniLM-L12-v2" # Use a different ST model
        }
    }
)
```

When calling `genie.rag.index_directory()` or `genie.rag.search()`, you can also pass `loader_id`, `splitter_id`, `embedder_id`, `vector_store_id`, `retriever_id`, and their corresponding `*_config` dictionaries to override the defaults for that specific call.

## Advanced Usage

The `genie.rag` interface internally uses the `RAGManager`. For advanced scenarios requiring direct interaction with the `RAGManager` or individual RAG component plugins, you can access it via `genie._rag_manager` (though this is generally not needed for typical use cases).

Refer to [Creating RAG Plugins](creating_rag_plugins.md) for information on developing your own RAG components.
