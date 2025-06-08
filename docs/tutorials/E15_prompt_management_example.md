# Tutorial: Prompt Management (E15)

This tutorial corresponds to the example file `examples/E15_prompt_management_example.py`.

It demonstrates how to use the prompt management system (`genie.prompts`) to separate prompt templates from application code. It shows how to:
- Configure `FileSystemPromptRegistryPlugin` to load templates from a directory.
- Use different template engines (`BasicStringFormatTemplatePlugin`, `Jinja2ChatTemplatePlugin`).
- Render simple string prompts and complex, structured chat prompts from templates.

## Example Code

--8<-- "examples/E1p_prompt_management_example.py"
