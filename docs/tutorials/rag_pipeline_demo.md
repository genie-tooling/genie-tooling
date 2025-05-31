# Tutorial: RAG Pipeline Demo

This tutorial demonstrates setting up a Retrieval Augmented Generation (RAG) pipeline to index local text files and perform similarity searches using the `Genie` facade.

The complete code for this tutorial can be found in [examples/rag_pipeline_demo/main.py](https://github.com/genie-tooling/genie-tooling/blob/main/examples/rag_pipeline_demo/main.py).
<!-- TODO: Update link when repo is public -->

## Prerequisites

*   Genie Tooling installed (`poetry install --all-extras`). This will include dependencies for local RAG like `sentence-transformers` and `faiss-cpu`.
*   Sample text files in a directory (e.g., `examples/rag_pipeline_demo/data/`).

## Core Logic

The demo will:
1.  Initialize the `Genie` facade. `FeatureSettings` will be used to configure RAG components (e.g., `SentenceTransformerEmbedder` and `FAISSVectorStore`).
2.  Use `genie.rag.index_directory()` to load documents from a specified local path, split them into chunks, generate embeddings, and store them in the FAISS vector store.
3.  Use `genie.rag.search()` to take a user query, embed it, and search the FAISS index for the most similar document chunks.
4.  Display the search results.

Refer to the example script for the full implementation details. It showcases how to:
*   Configure RAG components using `FeatureSettings`.
*   Use `genie.rag.index_directory()` for local file indexing.
*   Use `genie.rag.search()` for similarity search.
