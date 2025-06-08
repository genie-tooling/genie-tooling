# Tutorial: Conversation State (E16)

This tutorial corresponds to the example file `examples/E16_conversation_state_example.py`.

It demonstrates how to manage conversation history using `genie.conversation`. It shows how to:
- Configure a `ConversationStateProviderPlugin` (e.g., `in_memory_convo_provider`).
- Load, add messages to, and delete conversation state for a given session ID.
- Observe how metadata like `created_at` and `last_updated` is automatically managed.

## Example Code

--8<-- "examples/E16_conversation_state_example.py"
