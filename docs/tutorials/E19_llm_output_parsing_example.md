# Tutorial: LLM Output Parsing Example

This tutorial corresponds to the example file `examples/E19_llm_output_parsing_example.py`.

It demonstrates how to:
- Use `genie.llm.parse_output()` to convert LLM text responses into structured data.
- Parse LLM output into Python dictionaries (from JSON).
- Parse LLM output directly into Pydantic model instances.
- Configure and use `JSONOutputParserPlugin` and `PydanticOutputParserPlugin`.

```python
# Full code from examples/E19_llm_output_parsing_example.py
```

**Key Takeaways:**
- `genie.llm.parse_output()` simplifies extracting structured data from LLMs.
- Provide a Pydantic model class to the `schema` argument for direct model instantiation.
