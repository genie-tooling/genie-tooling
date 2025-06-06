## Genie Tooling Refactoring & Migration Guide for LLMs

**Primary LLM Directive:** Your goal is to analyze an existing Python codebase and systematically refactor its components to utilize the `genie-tooling` middleware. The primary interaction point will be the `genie_tooling.genie.Genie` facade. Prioritize modularity, configurability, and leveraging Genie's built-in plugin ecosystem. **Key Change**: Tools are now explicitly enabled via `MiddlewareConfig.tool_configurations`.

**Core Reference Documents for LLM:**
1.  **`docs/LLM_code_assist_instructions/LLM_api_guide.md`**: This is your **primary technical schema** for Genie Tooling's API, plugin types, configuration structure, and core concepts. Internalize this deeply, noting the change in tool enablement and new features like the internal Llama.cpp provider and distributed task queues.
2.  **`README.md`**: For high-level vision, core concepts (Genie facade, plugins, `@tool`, explicit tool enablement), and quick start examples.
3.  **`pyproject.toml` (`[tool.poetry.plugins."genie_tooling.plugins"]` section)**: Canonical source for plugin IDs and their Python module paths.
4.  **`src/genie_tooling/config/resolver.py` (`PLUGIN_ID_ALIASES` dict)**: Mapping of user-friendly aliases to canonical plugin IDs.
5.  **`examples/` directory**: Practical usage patterns and configuration examples, now reflecting explicit tool enablement and new features.
6.  **`docs/guides/`**: Detailed human-readable guides for specific features (e.g., RAG, LLM Providers, Configuration, Using Tools, Distributed Tasks, Observability).

---

### Phase 0: Deep Ingestion & Semantic Understanding of Genie Tooling

**LLM Instructions:**

1.  **Master `docs/LLM_code_assist_instructions/LLM_api_guide.md`**: This document is specifically structured for your consumption. Treat it as the ground truth for Genie's API, available plugin types, configuration parameters (via `MiddlewareConfig` and `FeatureSettings`), and core data structures (e.g., `ChatMessage`, `ToolCall`, `AgentOutput`). **Pay special attention to the `tool_configurations` field in `MiddlewareConfig` as the method for enabling tools, and note the new `llama_cpp_internal` LLM provider and `task_queue` interface.**
2.  **Internalize Core Concepts (`README.md`)**:
    *   **`Genie` Facade**: Understand its role as the central interaction point.
    *   **Plugin Architecture**: Grasp that most functionalities are swappable plugins.
    *   **Explicit Tool Enablement**: Recognize that tools are only loaded if their ID is a key in `MiddlewareConfig.tool_configurations`.
    *   **Managers**: Note their role in orchestrating plugin types (though direct interaction is rare when using the facade).
    *   **`@tool` Decorator**: Recognize its utility for quickly turning Python functions into tools. These decorated tools also need to be enabled via `tool_configurations` after registration.
    *   **Configuration (`MiddlewareConfig`, `FeatureSettings`, `ConfigResolver`)**: Understand the layered configuration approach.
3.  **Plugin Identification & Aliasing**:
    *   Correlate plugin IDs from `pyproject.toml` with aliases in `PLUGIN_ID_ALIASES` from `resolver.py`.
4.  **Facade & Interface Study**:
    *   Analyze `src/genie_tooling/genie.py` to see how the `Genie` class initializes and exposes its sub-interfaces, including the new `TaskQueueInterface`.
    *   Thoroughly review `src/genie_tooling/interfaces.py`.
5.  **Practical Application Patterns**:
    *   Examine all files in the `examples/` directory. Pay close attention to how `MiddlewareConfig` (especially `tool_configurations`) and `FeatureSettings` are used, including examples for new features like `E17_observability_tracing_example.py`, `E20_token_usage_example.py`, and `E22_distributed_task_example.py`.

---

### Phase 1: Target Project Analysis & Genie Capability Mapping

**LLM Instructions:**

1.  **Deconstruct Target Project**: (Same as before)
2.  **Create a Refactoring Map**: (Same as before, but when mapping to a Genie Tool, note that it will need to be added to `tool_configurations`).
3.  **Assess API Key Handling**: (Same as before)
4.  **Identify Custom Plugin Candidates**: (Same as before)

---

### Phase 2: Core Genie Integration & Configuration Setup

**LLM Instructions:**

1.  **Add Dependency**: (Same as before)
2.  **Design `MiddlewareConfig`**:
    *   Create a central configuration point.
    *   Instantiate `genie_tooling.config.models.MiddlewareConfig`.
    *   Prioritize using `genie_tooling.config.features.FeatureSettings`.
    *   **Crucially, populate `app_config.tool_configurations` with entries for *every tool* that needs to be active.** For tools without specific settings, an empty dictionary is sufficient (e.g., `app_config.tool_configurations = {"calculator_tool": {}}`).
        *   Example:
            ```python
            from genie_tooling.config.models import MiddlewareConfig
            from genie_tooling.config.features import FeatureSettings

            app_features = FeatureSettings(
                llm="ollama", 
                # ... other features like task_queue="celery" ...
            )
            app_config = MiddlewareConfig(
                features=app_features,
                tool_configurations={
                    "calculator_tool": {}, # Enable calculator
                    "my_existing_api_tool_id": {"api_base_url": "https://service.com/api"}, # Enable & configure
                    "another_simple_tool_id": {} 
                }
            )
            ```
    *   For functionalities not covered by `FeatureSettings` or requiring specific overrides, populate the relevant `*_configurations` dictionaries (e.g., `llm_provider_configurations`, `distributed_task_queue_configurations`).
    *   If custom plugins reside in project-specific directories, add these paths to `app_config.plugin_dev_dirs`.
3.  **Implement/Configure `KeyProvider`**: (Same as before)
4.  **Instantiate `Genie` Facade**: (Same as before)
5.  **Integrate Teardown**: (Same as before)

---

### Phase 3: Iterative Refactoring of Mapped Components

**LLM Instructions:** Systematically replace existing functionalities with their Genie Tooling equivalents.

1.  **LLM Interactions (`genie.llm`)**: (Same as before. Consider `llama_cpp_internal` for local execution.)
2.  **Tool Definition & Execution (`@tool`, `genie.execute_tool`, `genie.run_command`)**:
    *   **Refactor Functions to Tools**: Apply `@genie_tooling.tool`.
    *   **Register Tools**: Call `await genie.register_tool_functions([...])`.
    *   **Enable Registered Tools**: **Ensure the identifiers of these registered tools are added as keys to `MiddlewareConfig.tool_configurations` (e.g., `{"my_decorated_function_name": {}}`).**
    *   **Replace Direct Calls**: Change to `await genie.execute_tool("tool_name_as_string", ...)`.
    *   **Refactor Command Parsing**: Replace with `await genie.run_command(user_query_string)`. Ensure any tools the command processor might select are enabled in `tool_configurations`.
3.  **Command Processing (`genie.run_command`)**:
    *   (Same as before, but reiterate that any tools the processor might select must be enabled in `tool_configurations`).
4.  **RAG Pipeline (`genie.rag`)**: (Same as before)
5.  **Prompt Management (`genie.prompts`)**: (Same as before)
6.  **Conversation State (`genie.conversation`)**: (Same as before)
7.  **Observability (`genie.observability`)**: (Same as before, noting OpenTelemetry support)
8.  **Human-in-the-Loop (`genie.human_in_loop`)**: (Same as before)
9.  **Token Usage Tracking (`genie.usage`)**: (Same as before, noting OpenTelemetry metrics support)
10. **Guardrails**: (Same as before)
11. **LLM Output Parsing (`genie.llm.parse_output`)**: (Same as before)
12. **Agentic Loops (`genie_tooling.agents`)**:
    *   (Same as before, ensuring any tools used by the agent are enabled in `tool_configurations`).
13. **Distributed Task Offloading (`genie.task_queue`)**:
    *   Replace direct calls to Celery/RQ client libraries with `await genie.task_queue.submit_task(...)`, `await genie.task_queue.get_task_status(...)`, etc.
    *   If the remote task involves executing a Genie tool, that tool must be enabled in the *worker's* Genie configuration.
    *   Consider if the `DistributedTaskInvocationStrategy` can simplify tool offloading, if applicable and configured.

---

### Phase 4: Testing, Validation, and Refinement

**LLM Instructions:**

1.  **Unit Tests**: (Same as before)
2.  **Integration Tests**:
    *   Ensure `MiddlewareConfig` in tests correctly enables all necessary tools via `tool_configurations`.
    *   Test new features like task queue interactions and internal Llama.cpp provider usage.
3.  **Behavioral Verification**: (Same as before)
4.  **Configuration Robustness**: (Same as before)
5.  **Observability Review**: (Same as before, check OTel traces/metrics if used)

---

### Phase 5: Finalization - Documentation & Code Cleanup

**LLM Instructions:**

1.  **Update Project Documentation**:
    *   Clearly document that tools must be enabled via `MiddlewareConfig.tool_configurations`.
    *   Update guides for new features: Llama.cpp internal provider, distributed task queues (Celery & RQ), OpenTelemetry tracing and metrics.
2.  **Code Cleanup**: (Same as before)
3.  **Configuration Review**:
    *   Verify that `tool_configurations` correctly lists all active tools.
4.  **Update `LLM_api_guide.md` (If Genie Tooling Itself Was Modified)**:
    *   **CRITICAL LLM SELF-REFLECTION**: (Same as before. Ensure the API guide reflects the explicit tool enablement via `tool_configurations` and all new features like `llama_cpp_internal` and `task_queue` interface.)

---
