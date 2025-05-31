# Tutorial: Conversation State Example

This tutorial corresponds to the example file `examples/E16_conversation_state_example.py`.

It demonstrates how to:
- Configure a `ConversationStateProviderPlugin` (e.g., `InMemoryStateProviderPlugin`).
- Load existing conversation state for a session ID.
- Add new user and assistant messages to the conversation history.
- Observe how metadata like `created_at` and `last_updated` is managed.
- Delete conversation state.

```python
# Full code from examples/E16_conversation_state_example.py
```

**Key Takeaways:**
- `genie.conversation.load_state()` retrieves history.
- `genie.conversation.add_message()` appends to history and updates/saves the state.
- `genie.conversation.delete_state()` removes a session's history.
