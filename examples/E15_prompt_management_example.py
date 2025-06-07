# examples/E15_prompt_management_example.py
"""
Example: Using the Prompt Management System (`genie.prompts`)
-------------------------------------------------------------
This example demonstrates how to use `genie.prompts` to load and render
prompt templates. It assumes you have a `FileSystemPromptRegistryPlugin`
configured and some template files.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. The script will create a temporary prompt directory and files.
3. Run from the root of the project:
   `poetry run python examples/E15_prompt_management_example.py`
"""
import asyncio
import logging
import shutil  # For cleanup
from pathlib import Path
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.prompts.types import PromptData


async def run_prompt_management_demo():
    print("--- Prompt Management Example ---")
    logging.basicConfig(level=logging.INFO)

    prompt_dir = Path(__file__).parent / "my_app_prompts_e15" # Use example-local dir
    prompt_dir.mkdir(exist_ok=True)
    (prompt_dir / "greeting.txt").write_text("Hello, {name}! Welcome to {place}.")
    (prompt_dir / "chat_intro.j2").write_text(
        '[\n'
        '  {"role": "system", "content": "You are a {{ bot_role }} assistant."},\n'
        '  {"role": "user", "content": "Please tell me about {{ topic }}."}\n'
        ']'
    )

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="mistral:latest",
            prompt_registry="file_system_prompt_registry",
        ),
        prompt_registry_configurations={
            "file_system_prompt_registry_v1": {
                "base_path": str(prompt_dir.resolve()),
            }
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie for prompt management...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        print("\n--- Available Templates ---")
        templates = await genie.prompts.list_templates()
        if templates:
            for t_id in templates:
                print(f"- Name: {t_id['name']}, Version: {t_id.get('version', 'N/A')}")
        else:
            print("No templates found (check base_path and suffix).")

        print("\n--- Raw Template Content (greeting.txt) ---")
        raw_greeting = await genie.prompts.get_prompt_template_content(name="greeting")
        print(raw_greeting)

        print("\n--- Rendered String Prompt (greeting.txt) ---")
        greeting_data: PromptData = {"name": "Explorer", "place": "Genieville"}
        rendered_greeting = await genie.prompts.render_prompt(
            name="greeting",
            data=greeting_data,
            template_engine_id="basic_string_formatter" # Explicitly use basic formatter
        )
        print(rendered_greeting)

        print("\n--- Rendered Chat Prompt (chat_intro.j2) ---")
        chat_data: PromptData = {"bot_role": "helpful", "topic": "AI agents"}
        rendered_chat_messages = await genie.prompts.render_chat_prompt(
            name="chat_intro",
            data=chat_data,
            template_engine_id="jinja2_chat_formatter" # Explicitly use Jinja2 formatter
        )
        print(rendered_chat_messages)

        if rendered_chat_messages:
            print("\n--- Sending rendered chat to LLM ---")
            try:
                chat_response = await genie.llm.chat(rendered_chat_messages)
                print(f"LLM Response: {chat_response['message']['content'][:150]}...")
            except Exception as e_llm:
                print(f"LLM chat error: {e_llm}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Prompt management demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")
        if prompt_dir.exists():
            shutil.rmtree(prompt_dir)
            print(f"Cleaned up demo prompt directory: {prompt_dir}")

if __name__ == "__main__":
    asyncio.run(run_prompt_management_demo())
