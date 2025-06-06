# Managing Conversation State (`genie.conversation`)

Genie Tooling provides an interface for managing conversation state, primarily the history of messages in a chat interaction. This is accessible via `genie.conversation`.

## Core Concepts

*   **`ConversationInterface` (`genie.conversation`)**: The facade interface for all conversation state operations.
*   **`ConversationStateProviderPlugin`**: Responsible for storing, retrieving, and deleting conversation states. Each state is typically identified by a unique `session_id`.
    *   Built-in: `InMemoryStateProviderPlugin` (alias: `in_memory_convo_provider`), `RedisStateProviderPlugin` (alias: `redis_convo_provider`).
*   **`ConversationState` (TypedDict)**: The structure for conversation data:
    ```python
    from typing import List, Dict, Optional, Any
    from genie_tooling.llm_providers.types import ChatMessage # Assuming ChatMessage is defined

    class ConversationState(TypedDict):
        session_id: str
        history: List[ChatMessage]
        metadata: Optional[Dict[str, Any]] 
    ```

## Configuration

You configure the default conversation state provider via `FeatureSettings` or explicit `MiddlewareConfig` settings.

**Via `FeatureSettings`:**

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        # ... other features ...
        conversation_state_provider="in_memory_convo_provider" # Default
        # OR for Redis:
        # conversation_state_provider="redis_convo_provider" 
    ),
    # Example: Configure the RedisStateProviderPlugin if chosen
    # conversation_state_provider_configurations={
    #     "redis_conversation_state_v1": { # Canonical ID
    #         "redis_url": "redis://localhost:6379/2",
    #         "key_prefix": "my_app_cs:",
    #         "default_ttl_seconds": 7200 # 2 hours
    #     }
    # }
)
```

**Explicit Configuration:**

```python
app_config = MiddlewareConfig(
    default_conversation_state_provider_id="in_memory_conversation_state_v1",
    # conversation_state_provider_configurations={ ... }
)
```

## Using `genie.conversation`

### 1. Loading Conversation State

```python
# Assuming 'genie' is an initialized Genie instance
session_id = "user123_chat_abc"
state = await genie.conversation.load_state(session_id)
# state = await genie.conversation.load_state(session_id, provider_id="custom_store") # Optional

if state:
    print(f"Loaded history for {session_id}: {state['history']}")
else:
    print(f"No existing state for {session_id}.")
```

### 2. Adding a Message to a Session

This method handles loading the existing state (or creating a new one if it doesn't exist), appending the message, and saving the updated state.

```python
from genie_tooling.llm_providers.types import ChatMessage

session_id = "user123_chat_abc"
new_user_message: ChatMessage = {"role": "user", "content": "What's new?"}

await genie.conversation.add_message(session_id, new_user_message)
# await genie.conversation.add_message(session_id, new_user_message, provider_id="custom_store")

# If you then get an assistant response:
# assistant_response: ChatMessage = {"role": "assistant", "content": "Not much, how about you?"}
# await genie.conversation.add_message(session_id, assistant_response)
```
The `add_message` method automatically updates a `last_updated` timestamp in the state's metadata. If it's a new session, it also adds a `created_at` timestamp.

### 3. Saving Conversation State (Explicitly)

While `add_message` handles saving, you can explicitly save a state if you've modified it directly (e.g., adding custom metadata).

```python
from genie_tooling.prompts.conversation.types import ConversationState # Correct import

session_id = "user123_chat_abc"
current_state = await genie.conversation.load_state(session_id)
if not current_state:
    current_state = ConversationState(session_id=session_id, history=[], metadata={})

# Modify state
current_state["history"].append({"role": "system", "content": "Context updated."})
if current_state.get("metadata") is None: current_state["metadata"] = {}
current_state["metadata"]["custom_flag"] = True 

await genie.conversation.save_state(current_state)
```

### 4. Deleting Conversation State

```python
session_id = "user123_chat_to_delete"
was_deleted = await genie.conversation.delete_state(session_id)
if was_deleted:
    print(f"State for session {session_id} deleted.")
else:
    print(f"State for session {session_id} not found or delete failed.")
```

## Creating Custom Conversation State Providers

Implement the `ConversationStateProviderPlugin` protocol, defining `load_state`, `save_state`, and `delete_state` methods to interact with your chosen backend (database, file, etc.). Register your plugin via entry points or `plugin_dev_dirs`.
