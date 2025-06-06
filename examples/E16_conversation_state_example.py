# examples/E16_conversation_state_example.py
"""
Example: Using Conversation State Management (`genie.conversation`)
------------------------------------------------------------------
This example demonstrates how to load, add messages to, and save
conversation state using `genie.conversation`. It uses the in-memory
state provider by default.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Run from the root of the project:
   `poetry run python examples/E16_conversation_state_example.py`
"""
import asyncio
import logging
import uuid
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import ChatMessage


async def run_conversation_state_demo():
    print("--- Conversation State Example ---")
    logging.basicConfig(level=logging.INFO)

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="mistral:latest",
            conversation_state_provider="in_memory_convo_provider" 
        ),
    )

    genie: Optional[Genie] = None
    session_id = f"demo_session_e16_{uuid.uuid4().hex[:8]}"
    print(f"Using Session ID: {session_id}")

    try:
        print("\nInitializing Genie for conversation state...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        print(f"\n1. Loading state for session '{session_id}'...")
        initial_state = await genie.conversation.load_state(session_id)
        if initial_state:
            print(f"  Found existing state: {initial_state}")
        else:
            print("  No existing state found (as expected for new session).")

        print("\n2. Adding user message...")
        user_msg: ChatMessage = {"role": "user", "content": "Hello Genie, how are you?"}
        await genie.conversation.add_message(session_id, user_msg)
        print(f"  Added: {user_msg}")

        state_after_user = await genie.conversation.load_state(session_id)
        if state_after_user:
            print(f"  Current history: {state_after_user['history']}")
            print(f"  Metadata: {state_after_user.get('metadata')}")
        else:
            print("  Error: State not found after adding user message.")
            return

        print("\n3. Adding assistant message...")
        assistant_msg: ChatMessage = {"role": "assistant", "content": "I am doing well, thank you for asking!"}
        await genie.conversation.add_message(session_id, assistant_msg)
        print(f"  Added: {assistant_msg}")

        final_state = await genie.conversation.load_state(session_id)
        if final_state:
            print("\n--- Final Conversation State ---")
            print(f"Session ID: {final_state['session_id']}")
            print("History:")
            for msg in final_state["history"]:
                print(f"  - {msg['role']}: {msg['content']}")
            print(f"Metadata: {final_state.get('metadata')}")
        else:
            print("  Error: Final state not found.")

        print(f"\n4. Deleting state for session '{session_id}'...")
        deleted = await genie.conversation.delete_state(session_id)
        print(f"  Deletion successful: {deleted}")

        state_after_delete = await genie.conversation.load_state(session_id)
        assert state_after_delete is None, "State should be None after deletion."
        print("  State confirmed deleted.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Conversation state demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    asyncio.run(run_conversation_state_demo())
