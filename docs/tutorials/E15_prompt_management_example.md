# Tutorial: Prompt Management Example

This tutorial corresponds to the example file `examples/E15_prompt_management_example.py`.

It demonstrates how to:
- Configure `FileSystemPromptRegistryPlugin` and `Jinja2ChatTemplatePlugin`.
- List available prompt templates.
- Retrieve raw template content.
- Render string-based prompts.
- Render chat-based prompts suitable for LLM conversation.

```python
# Full code from examples/E15_prompt_management_example.py
# (This will be auto-filled by your documentation generation process if configured,
# or you can paste the example code here manually.)
```

**Key Takeaways:**
- Use `genie.prompts.get_prompt_template_content()` to fetch raw templates.
- Use `genie.prompts.render_prompt()` for simple string templates.
- Use `genie.prompts.render_chat_prompt()` for structured chat message lists, especially with Jinja2 templates.
