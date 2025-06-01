## Genie Tooling Refactoring & Migration Guide for LLMs

**Primary LLM Directive:** Your goal is to analyze an existing Python codebase and systematically refactor its components to utilize the `genie-tooling` middleware. The primary interaction point will be the `genie_tooling.genie.Genie` facade. Prioritize modularity, configurability, and leveraging Genie's built-in plugin ecosystem.

**Core Reference Documents for LLM:**
1.  **`docs/LLM_guide.md`**: This is your **primary technical schema** for Genie Tooling's API, plugin types, configuration structure, and core concepts. Internalize this deeply.
2.  **`README.md`**: For high-level vision, core concepts (Genie facade, plugins, `@tool`), and quick start examples.
3.  **`pyproject.toml` (`[tool.poetry.plugins."genie_tooling.plugins"]` section)**: Canonical source for plugin IDs and their Python module paths.
4.  **`src/genie_tooling/config/resolver.py` (`PLUGIN_ID_ALIASES` dict)**: Mapping of user-friendly aliases to canonical plugin IDs.
5.  **`examples/` directory**: Practical usage patterns and configuration examples.
6.  **`docs/guides/`**: Detailed human-readable guides for specific features (e.g., RAG, LLM Providers, Configuration).

---

### Phase 0: Deep Ingestion & Semantic Understanding of Genie Tooling

**LLM Instructions:**

1.  **Master `docs/LLM_guide.md`**: This document is specifically structured for your consumption. Treat it as the ground truth for Genie's API, available plugin types, configuration parameters (via `MiddlewareConfig` and `FeatureSettings`), and core data structures (e.g., `ChatMessage`, `ToolCall`, `AgentOutput`).
2.  **Internalize Core Concepts (`README.md`)**:
    *   **`Genie` Facade**: Understand its role as the central interaction point.
    *   **Plugin Architecture**: Grasp that most functionalities are swappable plugins.
    *   **Managers**: Note their role in orchestrating plugin types (though direct interaction is rare when using the facade).
    *   **`@tool` Decorator**: Recognize its utility for quickly turning Python functions into tools.
    *   **Configuration (`MiddlewareConfig`, `FeatureSettings`, `ConfigResolver`)**: Understand the layered configuration approach, from high-level features to specific plugin settings.
3.  **Plugin Identification & Aliasing**:
    *   Correlate plugin IDs from `pyproject.toml` with aliases in `PLUGIN_ID_ALIASES` from `resolver.py`. This is vital for interpreting `FeatureSettings` and user configurations.
4.  **Facade & Interface Study**:
    *   Analyze `src/genie_tooling/genie.py` to see how the `Genie` class initializes and exposes its sub-interfaces.
    *   Thoroughly review `src/genie_tooling/interfaces.py` to understand the methods available under `genie.llm`, `genie.rag`, `genie.prompts`, etc.
5.  **Practical Application Patterns**:
    *   Examine all files in the `examples/` directory. Pay close attention to how `MiddlewareConfig` and `FeatureSettings` are used to initialize `Genie` and how various facade methods are invoked.
    *   Note how different plugin types are configured and utilized in these examples.

---

### Phase 1: Target Project Analysis & Genie Capability Mapping

**LLM Instructions:**

1.  **Deconstruct Target Project**: Analyze the existing codebase to identify components and patterns that can be mapped to Genie Tooling features. Create a mental (or actual, if helpful for your process) inventory:
    *   **LLM Interactions**: Identify direct calls to `openai`, `google.generativeai`, `httpx` (for Ollama/Llama.cpp), etc. Note models used, parameters (temp, max_tokens), and error handling.
    *   **Tool/Function Logic**: Pinpoint functions or classes that perform discrete, reusable actions (e.g., calculations, API calls, file operations).
    *   **Command Interpretation**: Find any logic that parses natural language into function calls or tool selections.
    *   **RAG Components**: Locate code for document loading, text splitting, embedding generation, vector storage, and similarity search.
    *   **Configuration**: Identify how API keys, model names, URLs, and other settings are managed.
    *   **Prompt Management**: Find hardcoded prompts, f-string templates, or custom templating systems.
    *   **Conversation History**: Analyze how chat history is stored and passed to LLMs.
    *   **Agentic Loops**: If present, identify patterns like ReAct or Plan-and-Execute.
    *   **Safety/Validation**: Note any input/output validation or content filtering.
    *   **Observability**: Current logging or tracing for LLM/tool interactions.
2.  **Create a Refactoring Map**: For each identified component in the target project, map it to a corresponding Genie Tooling feature or plugin type. Refer heavily to `docs/LLM_guide.md` and `README.md` for this.
    *   *Example Mapping:*
        *   `requests.get("some_api")` -> Potential `@tool` function, or a custom `ToolPlugin`.
        *   Manual OpenAI API calls -> `genie.llm.chat(provider_id="openai", ...)`
        *   Custom vector search logic -> `genie.rag.search()` with configured RAG plugins.
        *   Environment variables for API keys -> `EnvironmentKeyProvider` (default) or custom `KeyProvider`.
        *   Hardcoded system prompts -> `FileSystemPromptRegistryPlugin` + `genie.prompts.render_chat_prompt()`.
        *   Existing ReAct loop -> `genie_tooling.agents.ReActAgent`.
3.  **Assess API Key Handling**: Determine the current strategy for API key management. Plan to integrate with Genie's `KeyProvider` system. If keys are in environment variables, the default `EnvironmentKeyProvider` might suffice. Otherwise, a custom `KeyProvider` implementation will be necessary.
4.  **Identify Custom Plugin Candidates**: Beyond simple `@tool` functions, determine if any complex components from the target project would be better implemented as full Genie plugins (e.g., a custom `VectorStorePlugin` for an unsupported database, a specialized `CommandProcessorPlugin`).

---

### Phase 2: Core Genie Integration & Configuration Setup

**LLM Instructions:**

1.  **Add Dependency**: Ensure `genie-tooling` is added to the target project's dependencies (e.g., in `pyproject.toml` or `requirements.txt`).
2.  **Design `MiddlewareConfig`**:
    *   Create a central configuration point for Genie, typically where the application initializes.
    *   Instantiate `genie_tooling.config.models.MiddlewareConfig`.
    *   Prioritize using `genie_tooling.config.features.FeatureSettings` for high-level configuration based on the analysis in Phase 1.
        *   Example: If the target project uses Ollama for LLM and FAISS for RAG:
            ```python
            from genie_tooling.config.models import MiddlewareConfig
            from genie_tooling.config.features import FeatureSettings

            app_features = FeatureSettings(
                llm="ollama",
                llm_ollama_model_name="identified_ollama_model", # From target project
                rag_embedder="sentence_transformer", # Assuming local embeddings
                rag_embedder_st_model_name="identified_st_model", # From target project
                rag_vector_store="faiss",
                # ... other features based on Phase 1 analysis ...
            )
            app_config = MiddlewareConfig(features=app_features)
            ```
    *   For functionalities not covered by `FeatureSettings` or requiring specific overrides, populate the relevant `*_configurations` dictionaries in `MiddlewareConfig` (e.g., `tool_configurations`, `command_processor_configurations`). Use canonical plugin IDs or recognized aliases (refer to `PLUGIN_ID_ALIASES`).
    *   If custom plugins (developed in Phase 3) will reside in project-specific directories, add these paths to `app_config.plugin_dev_dirs`.
3.  **Implement/Configure `KeyProvider`**:
    *   If using environment variables that match Genie's defaults (e.g., `OPENAI_API_KEY`), no explicit `KeyProvider` instance needs to be passed to `Genie.create()`.
    *   If custom key names or sources are used, implement a class inheriting from `genie_tooling.security.key_provider.KeyProvider` and plan to pass an instance to `Genie.create()`.
4.  **Instantiate `Genie` Facade**:
    *   At an appropriate point in the application's startup sequence (e.g., in `main()` or an initialization function), create the `Genie` instance:
        ```python
        # from genie_tooling.genie import Genie
        # from my_project_key_provider import MyCustomKeyProvider # If applicable

        # key_provider_instance = MyCustomKeyProvider() if using_custom_kp else None
        # global_genie_instance = await Genie.create(
        #     config=app_config,
        #     key_provider_instance=key_provider_instance
        # )
        ```
    *   Plan for how this `genie_instance` will be accessed by other parts of the refactored application (e.g., passed via dependency injection, global singleton if appropriate for the project structure).
5.  **Integrate Teardown**: Ensure `await global_genie_instance.close()` is called when the application shuts down to release resources held by plugins.

---

### Phase 3: Iterative Refactoring of Mapped Components

**LLM Instructions:** Systematically replace existing functionalities with their Genie Tooling equivalents, using the `genie_instance` created in Phase 2. Refer to `docs/LLM_guide.md` for precise method signatures and `examples/` for usage patterns.

1.  **LLM Interactions (`genie.llm`)**:
    *   Replace direct SDK calls (e.g., `openai.ChatCompletion.create(...)`) with `await genie.llm.chat(...)` or `await genie.llm.generate(...)`.
    *   Pass `provider_id` if needing to switch between multiple configured LLMs.
    *   Migrate LLM parameters (temperature, max_tokens) to the `**kwargs` of Genie's methods or configure them as defaults in `MiddlewareConfig.llm_provider_configurations`.

2.  **Tool Definition & Execution (`@tool`, `genie.execute_tool`, `genie.run_command`)**:
    *   **Refactor Functions to Tools**: Apply `@genie_tooling.tool` to identified Python functions. Ensure type hints are accurate and docstrings are descriptive (especially `Args:` section for parameter descriptions).
    *   **Register Tools**: After `Genie` instantiation, call `await genie.register_tool_functions([list_of_decorated_functions])`.
    *   **Replace Direct Calls**: Change existing direct function calls (that are now tools) to `await genie.execute_tool("tool_name_as_string", arg1=val1, ...)`.
    *   **Refactor Command Parsing**: If the old code parsed natural language to call tools, replace this logic with `await genie.run_command(user_query_string)`. This requires configuring a `CommandProcessorPlugin` (see next point).

3.  **Command Processing (`genie.run_command`)**:
    *   Configure `features.command_processor` (e.g., `"llm_assisted"`).
    *   If using `"llm_assisted"`:
        *   Ensure `features.llm` is set.
        *   Configure `features.tool_lookup` (e.g., `"embedding"`) and its associated `tool_lookup_formatter_id_alias` and `tool_lookup_embedder_id_alias`.
        *   Set `command_processor_configurations` for `llm_assisted_tool_selection_processor_v1` if non-default `tool_lookup_top_k` or system prompt is needed.
    *   If using `"simple_keyword"`, configure `command_processor_configurations["simple_keyword_processor_v1"]["keyword_map"]`.

4.  **RAG Pipeline (`genie.rag`)**:
    *   **Indexing**: Replace custom document loading, splitting, embedding, and vector store ingestion with `await genie.rag.index_directory(...)` or `await genie.rag.index_web_page(...)`.
    *   **Search**: Replace custom similarity search logic with `await genie.rag.search(...)`.
    *   **Configuration**:
        *   Set `features.rag_embedder` and `features.rag_vector_store`.
        *   Provide necessary paths, collection names, API keys (for cloud vector stores via `KeyProvider`), or embedding dimensions via `features` (e.g., `features.rag_vector_store_qdrant_url`) or `MiddlewareConfig`'s `embedding_generator_configurations` and `vector_store_configurations`.
        *   If custom RAG components are needed, implement `DocumentLoaderPlugin`, `TextSplitterPlugin`, `EmbeddingGeneratorPlugin`, or `VectorStorePlugin` and register/configure them.

5.  **Prompt Management (`genie.prompts`)**:
    *   Move hardcoded prompts or templates to external files (e.g., `.txt`, `.j2`).
    *   Configure `features.prompt_registry` (e.g., `"file_system_prompt_registry"`) and set `prompt_registry_configurations` in `MiddlewareConfig` (e.g., `base_path`).
    *   Configure `features.prompt_template_engine` (e.g., `"basic_string_formatter"`, `"jinja2_chat_formatter"`).
    *   Replace old templating logic with `await genie.prompts.render_prompt(...)` or `await genie.prompts.render_chat_prompt(...)`.

6.  **Conversation State (`genie.conversation`)**:
    *   Replace custom chat history management with `await genie.conversation.load_state(...)`, `await genie.conversation.add_message(...)`, etc.
    *   Configure `features.conversation_state_provider` (e.g., `"in_memory_convo_provider"`, `"redis_convo_provider"`) and its settings in `conversation_state_provider_configurations`.

7.  **Observability (`genie.observability`)**:
    *   Configure `features.observability_tracer` (e.g., `"console_tracer"`).
    *   Rely on automatic tracing from Genie facade methods.
    *   Insert `await genie.observability.trace_event(...)` for custom application-specific events.

8.  **Human-in-the-Loop (`genie.human_in_loop`)**:
    *   If `genie.run_command()` is used, HITL for tool execution is automatic if `features.hitl_approver` is configured (e.g., to `"cli_hitl_approver"`).
    *   For other custom approval points, replace existing logic with `await genie.human_in_loop.request_approval(...)`.

9.  **Token Usage Tracking (`genie.usage`)**:
    *   Configure `features.token_usage_recorder`.
    *   Remove custom token counting logic; rely on automatic recording by `genie.llm` calls.
    *   Use `await genie.usage.get_summary()` for reporting.

10. **Guardrails**:
    *   Configure `features.input_guardrails`, `output_guardrails`, `tool_usage_guardrails` with aliases/IDs of guardrail plugins (e.g., `keyword_blocklist_guardrail`).
    *   Set specific configurations for these guardrails in `MiddlewareConfig.guardrail_configurations`.
    *   Replace existing validation/filtering logic with Genie's guardrail system where appropriate.

11. **LLM Output Parsing (`genie.llm.parse_output`)**:
    *   Replace custom JSON/structured data extraction from LLM string responses with `await genie.llm.parse_output(llm_response, schema=MyPydanticModelOrJsonSchema)`.
    *   Configure `features.default_llm_output_parser` or specify `parser_id` in the call.

12. **Agentic Loops (`genie_tooling.agents`)**:
    *   If the target project has ReAct or Plan-and-Execute style agents, refactor them to use `genie_tooling.agents.ReActAgent` or `genie_tooling.agents.PlanAndExecuteAgent`.
    *   Pass the `genie_instance` to the agent's constructor.
    *   Configure agent-specific parameters (like system prompt IDs, max iterations) via the `agent_config` dictionary passed to the agent constructor. Ensure these prompts are managed by `genie.prompts`.

---

### Phase 4: Testing, Validation, and Refinement

**LLM Instructions:**

1.  **Unit Tests**:
    *   For any new custom plugins (`Tool`, `KeyProvider`, `RAGPlugin`, etc.), write comprehensive unit tests.
    *   For application logic now using `Genie`, mock the relevant `genie.facade.method()` calls to test the logic in isolation.
2.  **Integration Tests**:
    *   Set up integration tests that initialize a `Genie` instance with a minimal but functional configuration (e.g., using Ollama, in-memory RAG components, console tracer).
    *   Test key end-to-end flows:
        *   `genie.run_command()` -> tool selection -> tool execution -> result.
        *   `genie.rag.index_directory()` followed by `genie.rag.search()`.
        *   `genie.llm.chat()` and `genie.llm.generate()` with different providers if configured.
        *   Agent runs (if `ReActAgent` or `PlanAndExecuteAgent` are used).
3.  **Behavioral Verification**: Compare the behavior and outputs of the refactored system against the original system for critical user scenarios. Identify and address any regressions or unintended changes.
4.  **Configuration Robustness**: Test the application with varied `MiddlewareConfig` settings (e.g., switching LLM providers via `features`, enabling/disabling tool lookup) to ensure the refactored system adapts correctly.
5.  **Observability Review**: If tracing is enabled, review the trace outputs to ensure they provide clear and useful insights into the application's operation with Genie.

---

### Phase 5: Finalization - Documentation & Code Cleanup

**LLM Instructions:**

1.  **Update Project Documentation**:
    *   Modify `README.md` and any other developer documentation to reflect the use of Genie Tooling.
    *   Explain the new configuration process using `MiddlewareConfig` and `FeatureSettings`.
    *   Document how to add new tools (using `@tool` or custom plugins).
    *   Describe how to switch between different LLMs, RAG backends, etc., via configuration.
2.  **Code Cleanup**:
    *   Remove all old code modules, classes, and utility functions that have been made redundant by Genie Tooling.
    *   Ensure all new code (custom plugins, application logic using Genie) adheres to the project's coding standards and style guides (e.g., run `black .` and `ruff check . --fix` if used).
3.  **Configuration Review**: Perform a final review of the primary `MiddlewareConfig` setup in the application. Ensure it's clear, well-commented, and leverages `FeatureSettings` effectively for defaults, with explicit overrides only where necessary for custom behavior.
4.  **Update `LLM_guide.md` (If Genie Tooling Itself Was Modified)**:
    *   **CRITICAL LLM SELF-REFLECTION**: If, during the refactoring process, *you* (the LLM) identified a need to modify or extend the Genie Tooling library itself (e.g., adding a new core feature, changing a plugin interface), you *must* update `docs/LLM_guide.md` to accurately reflect these changes. This ensures future LLM interactions with Genie Tooling remain consistent.

---
