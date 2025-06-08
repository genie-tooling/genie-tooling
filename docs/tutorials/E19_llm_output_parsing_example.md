# Tutorial: LLM Output Parsing (E19)

This tutorial corresponds to the example file `examples/E19_llm_output_parsing_example.py`.

It demonstrates how to reliably extract structured data from an LLM's text response. It shows how to:
- Use `genie.llm.parse_output()` to convert LLM text into Python objects.
- Configure and use `JSONOutputParserPlugin` to extract JSON.
- Configure and use `PydanticOutputParserPlugin` to parse JSON directly into a Pydantic model instance for validation and type safety.

## Example Code

--8<-- "examples/E19_llm_output_parsing_example.py"
