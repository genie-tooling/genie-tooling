# Tutorial: Llama.cpp Internal GBNF (E25)

This tutorial corresponds to the example file `examples/E25_llama_cpp_internal_gbnf_parsing.py`.

It demonstrates a key feature of the Llama.cpp providers: using GBNF grammar for constrained, structured output. It shows how to:
- Define a Pydantic model for the desired output structure.
- Pass this model to `genie.llm.chat()` via the `output_schema` parameter.
- Observe how the Llama.cpp internal provider generates a JSON string that perfectly matches the Pydantic model, which can then be parsed reliably.

## Example Code

--8<-- "examples/E25_llama_cpp_internal_gbnf_parsing.py"
