# examples/E21_guardrails_example.py
"""
Example: Using Guardrails
-------------------------
This example demonstrates how to configure and use Guardrails in Genie Tooling.
It uses the built-in `KeywordBlocklistGuardrailPlugin`.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Ensure Ollama is running and 'mistral:latest' is pulled (or configure a different LLM).
3. Run from the root of the project:
   `poetry run python examples/E21_guardrails_example.py`
   Observe how inputs/outputs containing 'secret' or 'forbidden' are handled.
"""
import asyncio
import logging
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def run_guardrails_demo():
    print("--- Guardrails Example ---")
    logging.basicConfig(level=logging.INFO)

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="mistral:latest",
            command_processor="llm_assisted",
            tool_lookup="embedding",
            input_guardrails=["keyword_blocklist_guardrail"],
            output_guardrails=["keyword_blocklist_guardrail"],
        ),
        guardrail_configurations={
            "keyword_blocklist_guardrail_v1": {
                "blocklist": ["secret", "forbidden_word", "sensitive_operation"],
                "case_sensitive": False,
                "action_on_match": "block"
            }
        },
        tool_configurations={
            "calculator_tool": {}
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie with Guardrails enabled...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        print("\n--- Testing Input Guardrail (LLM Chat) ---")
        blocked_input_prompt = "Tell me a secret about the project."
        print(f"Sending potentially blocked input: '{blocked_input_prompt}'")
        try:
            chat_response_blocked = await genie.llm.chat([{"role": "user", "content": blocked_input_prompt}])
            print(f"LLM Response (should not be reached if blocked): {chat_response_blocked['message']['content']}")
        except PermissionError as e_perm:
            print(f"Input Guardrail Blocked: {e_perm}")
        except Exception as e_llm_in:
            print(f"LLM chat error (possibly unrelated to guardrail): {e_llm_in}")

        allowed_input_prompt = "Tell me a fun fact."
        print(f"\nSending allowed input: '{allowed_input_prompt}'")
        try:
            chat_response_allowed = await genie.llm.chat([{"role": "user", "content": allowed_input_prompt}])
            print(f"LLM Response: {chat_response_allowed['message']['content'][:100]}...")
        except Exception as e_llm_allowed:
            print(f"LLM chat error for allowed input: {e_llm_allowed}")


        print("\n--- Testing Output Guardrail (LLM Generate) ---")
        prompt_for_blocked_output = "Write a sentence that includes the word 'secret'."
        print(f"Sending prompt for potentially blocked output: '{prompt_for_blocked_output}'")
        try:
            gen_response = await genie.llm.generate(prompt_for_blocked_output)
            print(f"LLM Generated Text: {gen_response['text']}")
            if "[RESPONSE BLOCKED" in gen_response["text"]:
                print("Output Guardrail successfully blocked the response.")
            elif "secret" in gen_response["text"].lower():
                 print("Warning: Output guardrail did not block 'secret' as expected. LLM output was: ", gen_response["text"])
        except Exception as e_gen:
            print(f"LLM generate error: {e_gen}")

        print("\n--- (Illustrative) Tool Usage Guardrail ---")
        print("To test tool usage guardrails, enable 'tool_usage_guardrails' in FeatureSettings")
        print("and try a command that would pass a blocked keyword as a tool parameter.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Guardrails demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    asyncio.run(run_guardrails_demo())
