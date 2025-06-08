# Tutorial: Web RAG (E09)

This tutorial corresponds to the example file `examples/E09_web_rag_example.py`.

It demonstrates how to use the RAG pipeline to ingest data directly from the web. It shows how to:
- Configure the `WebPageLoader` plugin (aliased as `web_page`).
- Use `genie.rag.index_web_page()` to fetch content from a URL, extract its main text, and index it into a vector store.
- Perform a semantic search over the ingested web content.

## Example Code

--8<-- "examples/E09_web_rag_example.py"
