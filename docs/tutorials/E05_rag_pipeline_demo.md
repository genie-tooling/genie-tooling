# Tutorial: RAG Pipeline Demo (E05)

This tutorial corresponds to the example file `examples/E05_rag_pipeline_demo.py`.

It demonstrates how to set up and use a complete, local Retrieval Augmented Generation (RAG) pipeline. It shows how to:
- Configure a local RAG pipeline using `FeatureSettings` (Sentence Transformers for embeddings and FAISS for an in-memory vector store).
- Index documents from a local directory using `genie.rag.index_directory()`.
- Perform a similarity search on the indexed documents using `genie.rag.search()`.

## Example Code

--8<-- "examples/E05_rag_pipeline_demo.py"
