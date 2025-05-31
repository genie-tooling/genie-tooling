# examples/E15_prompt_management_example.py
"""
Example: Using the Prompt Management System (`genie.prompts`)
-------------------------------------------------------------
This example demonstrates how to use `genie.prompts` to load and render
prompt templates. It assumes you have a `FileSystemPromptRegistryPlugin`
configured and some template files.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Create a directory for prompts, e.g., `./my_app_prompts`.
3. Create template files inside it:
   - `./my_app_prompts/greeting.txt` (Content: "Hello, {name}! Welcome to {place}.")
   - `./my_app_prompts/chat_intro.j2` (Content for Jinja2:
     ```jinja2
     [
       {"role": "system", "content": "You are a {{ bot_role }} assistant."},
       {"role": "user", "content": "Please tell me about {{ topic }}."}
     ]
     ```
     )
4. Run from the root of the project:
   `poetry run python examples/E15_prompt_management_example.py`
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.prompts.types import PromptData


async def run_prompt_management_demo():
    print("--- Prompt Management Example ---")
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG) # For detailed logs

    # Setup prompt directory and files for the demo
    prompt_dir = Path("./my_app_prompts_e15")
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
            # LLM needed if we want to use the rendered chat prompt
            llm="ollama",
            llm_ollama_model_name="mistral:latest",

            prompt_registry="file_system_prompt_registry",
            # Use basic for string, Jinja2 for chat to show both
            # Default template engine will be used if not specified in render calls
            # For this demo, we'll specify per call to be explicit.
        ),
        prompt_registry_configurations={
            "file_system_prompt_registry_v1": {
                "base_path": str(prompt_dir.resolve()),
                # Suffix will be auto-detected or can be specified
            }
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie for prompt management...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        # 1. List available templates
        print("\n--- Available Templates ---")
        templates = await genie.prompts.list_templates()
        if templates:
            for t_id in templates:
                print(f"- Name: {t_id['name']}, Version: {t_id.get('version', 'N/A')}")
        else:
            print("No templates found (check base_path and suffix).")

        # 2. Get raw template content
        print("\n--- Raw Template Content (greeting.txt) ---")
        raw_greeting = await genie.prompts.get_prompt_template_content(name="greeting") # Suffix .txt implied
        print(raw_greeting)

        # 3. Render a string prompt (using BasicStringFormatTemplatePlugin)
        print("\n--- Rendered String Prompt (greeting.txt) ---")
        greeting_data: PromptData = {"name": "Explorer", "place": "Genieville"}
        rendered_greeting = await genie.prompts.render_prompt(
            name="greeting", # Suffix .txt implied
            data=greeting_data,
            template_engine_id="basic_string_format_template_v1" # Explicitly use basic formatter
        )
        print(rendered_greeting)

        # 4. Render a chat prompt (using Jinja2ChatTemplatePlugin)
        print("\n--- Rendered Chat Prompt (chat_intro.j2) ---")
        chat_data: PromptData = {"bot_role": "helpful", "topic": "AI agents"}
        rendered_chat_messages = await genie.prompts.render_chat_prompt(
            name="chat_intro", # Suffix .j2 implied
            data=chat_data,
            template_engine_id="jinja2_chat_template_v1" # Explicitly use Jinja2 formatter
        )
        print(rendered_chat_messages)

        # Optional: Use the rendered chat messages with an LLM
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
        # Clean up demo prompt files
        if prompt_dir.exists():
            for item in prompt_dir.iterdir(): item.unlink()
            prompt_dir.rmdir()

if __name__ == "__main__":
    asyncio.run(run_prompt_management_demo())
