# Creating RAG Plugins

Genie Tooling's Retrieval Augmented Generation (RAG) system is composed of several pluggable components. You can create custom implementations for each to tailor the RAG pipeline to your specific data sources and needs.

The core RAG component protocols are:

*   **`DocumentLoaderPlugin`**: Loads raw data from a source (e.g., files, web pages, databases) into `Document` objects.
    *   Located in: `genie_tooling.document_loaders.abc`
    *   Key method: `async def load(self, source_uri: str, config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Document]`
*   **`TextSplitterPlugin`**: Splits `Document` objects into smaller `Chunk` objects.
    *   Located in: `genie_tooling.text_splitters.abc`
    *   Key method: `async def split(self, documents: AsyncIterable[Document], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Chunk]`
*   **`EmbeddingGeneratorPlugin`**: Generates embedding vectors for `Chunk` objects.
    *   Located in: `genie_tooling.embedding_generators.abc`
    *   Key method: `async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]`
*   **`VectorStorePlugin`**: Stores, manages, and searches `Chunk` embeddings.
    *   Located in: `genie_tooling.vector_stores.abc`
    *   Key methods: `async def add(...)`, `async def search(...)`, `async def delete(...)`.
*   **`RetrieverPlugin`**: Orchestrates the process of taking a query, embedding it, and searching a vector store to retrieve relevant chunks. Often composes an `EmbeddingGeneratorPlugin` and a `VectorStorePlugin`.
    *   Located in: `genie_tooling.retrievers.abc`
    *   Key method: `async def retrieve(self, query: str, top_k: int, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]`

## Steps to Create a RAG Plugin

1.  **Identify the Component**: Determine which part of the RAG pipeline you want to customize (e.g., a new document loader for a specific API, a different text splitting strategy).
2.  **Implement the Protocol**:
    *   Create a Python class that inherits from the relevant plugin protocol (e.g., `DocumentLoaderPlugin`).
    *   Your class must also implicitly or explicitly adhere to the base `genie_tooling.core.types.Plugin` protocol (by having a `plugin_id` and optional `setup`/`teardown` methods).
    *   Implement all required methods from the chosen RAG component protocol.
    ```python
    # Example: Custom Document Loader
    from genie_tooling.core.types import Document
    from genie_tooling.document_loaders.abc import DocumentLoaderPlugin
    from typing import AsyncIterable, Dict, Any, Optional

    class MyCustomAPILoader(DocumentLoaderPlugin):
        plugin_id: str = "my_custom_api_loader_v1"
        description: str = "Loads documents from MyCustomAPI."

        async def setup(self, config: Optional[Dict[str, Any]] = None):
            # Initialize API clients, etc.
            self.api_endpoint = (config or {}).get("api_endpoint", "https://api.example.com/data")
            # self.key_provider = (config or {}).get("key_provider") # If API key needed

        async def load(self, source_uri: str, config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Document]:
            # source_uri might be an entity ID or query for your API
            # api_key = await self.key_provider.get_key("MY_CUSTOM_API_KEY")
            # ... fetch data from API ...
            # for item in fetched_data:
            #     yield Document(content=item['text'], metadata={"source": "MyCustomAPI", "id": item['id']})
            if False: # Make it an async generator
                 yield
            pass # Replace with actual implementation
    ```
3.  **Register Your Plugin**:
    *   Use entry points in `pyproject.toml` or place it in a `plugin_dev_dirs` directory.
4.  **Configure Genie to Use Your Plugin**:
    *   In `MiddlewareConfig`, update the relevant `features` settings (e.g., `features.rag_loader = "my_custom_api_loader_v1"`) or explicitly set the default ID (e.g., `default_rag_loader_id = "my_custom_api_loader_v1"`).
    *   Provide any necessary configuration for your plugin in the corresponding `*_configurations` dictionary (e.g., `document_loader_configurations["my_custom_api_loader_v1"] = {"api_endpoint": "..."}`).

## Example: Using a Custom RAG Component

```python
# In your MiddlewareConfig
app_config = MiddlewareConfig(
    features=FeatureSettings(
        rag_loader="my_custom_api_loader_v1", # Use your custom loader
        rag_embedder="st_embedder",
        rag_vector_store="faiss_vs"
    ),
    document_loader_configurations={
        "my_custom_api_loader_v1": {
            "api_endpoint": "https://my.service.com/data_source",
            # "key_provider": my_key_provider_instance (if needed and passed to Genie.create)
        }
    }
)

# genie = await Genie.create(config=app_config, key_provider_instance=my_key_provider_instance)
# await genie.rag.index_directory(source_uri="query_for_my_api", collection_name="custom_data")
```
The `RAGManager` (used internally by `genie.rag`) will then pick up and use your custom plugin based on the configuration.
