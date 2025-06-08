# Tutorial: ChromaDB Tool Lookup (E10)

This tutorial corresponds to the example file `examples/E10_chroma_tool_lookup_showcase.py`.

It demonstrates how to make the embedding-based tool lookup service persistent. It shows how to:
- Configure the `embedding` tool lookup provider to use ChromaDB as its backend via `FeatureSettings`.
- Specify a local path for the ChromaDB database, allowing tool embeddings to persist between application runs.
- Use the `llm_assisted` command processor, which will now benefit from this persistent tool index.

## Example Code

--8<-- "examples/E10_chroma_tool_lookup_showcase.py"
