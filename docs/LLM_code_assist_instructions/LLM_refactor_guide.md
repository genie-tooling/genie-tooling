## Genie Tooling Refactoring & Migration Guide for LLMs

**Primary LLM Directive:** Your goal is to analyze an existing Python codebase and systematically refactor its components to utilize the `genie-tooling` middleware. The primary interaction point will be the `genie_tooling.genie.Genie` facade. Prioritize modularity, configurability, and leveraging Genie's built-in plugin ecosystem.

**Critical Change**: All tools, whether class-based or function-based (`@tool`), are now enabled via the `MiddlewareConfig.tool_configurations` dictionary when in production mode (`auto_enable_registered_tools=False`).

**Core Reference Documents for LLM:**
1.  **`docs/LLM_code_assist_instructions/LLM_api_guide.md`**: Your **primary technical schema** for Genie Tooling's API.
2.  **`README.md`**: For high-level vision and the explicit tool enablement concept.
3.  **`docs/guides/extending_genie_tooling.md`**: The official guide for creating extensions using the new `BootstrapPlugin` system.
4.  **`pyproject.toml`**: Source for canonical plugin IDs.
5.  **`src/genie_tooling/config/resolver.py`**: Source for plugin ID aliases.
6.  **`examples/` directory**: Practical usage patterns reflecting the latest code.

---
### Phase 1: Target Project Analysis & Genie Capability Mapping
(This phase remains the same: Deconstruct, Map, Assess Keys, Identify Custom Plugins.)

---
### Phase 2: Core Genie Integration & Configuration Setup

**LLM Instructions:**

1.  **Add Dependency**: (Same)
2.  **Design `MiddlewareConfig`**:
    *   Create a central configuration point using `genie_tooling.config.models.MiddlewareConfig`.
    *   Prioritize using `genie_tooling.config.features.FeatureSettings`.
    *   **Crucially, populate `app_config.tool_configurations` with keys for *every tool* that needs to be active.** For tools without specific settings, an empty dictionary is sufficient (e.g., `{"calculator_tool": {}}`).
    *   Set `auto_enable_registered_tools=False` for production-grade refactoring.
3.  **Implement/Configure `KeyProvider`**: (Same)
4.  **Instantiate `Genie` Facade**: (Same)
5.  **Integrate Teardown**: (Same)

---
### Phase 3: Iterative Refactoring of Mapped Components

**LLM Instructions:** Systematically replace existing functionalities with their Genie Tooling equivalents.

1.  **LLM Interactions (`genie.llm`)**: (Same)
2.  **Tool Definition & Execution**:
    *   Refactor functions to tools using `@genie_tooling.tool`.
    *   Register tools with `await genie.register_tool_functions([...])`.
    *   **Enable Registered Tools**: **Ensure the identifiers of these registered tools are added as keys to `MiddlewareConfig.tool_configurations` (e.g., `{"my_decorated_function_name": {}}`).**
    *   Replace direct calls with `await genie.execute_tool(...)`.
3.  **Command Processing (`genie.run_command`)**:
    *   Refactor command parsing logic to `await genie.run_command(user_query_string)`.
    *   Ensure any tools the processor might select are enabled in `tool_configurations`.
    *   For complex research tasks, consider using the `rewoo` command processor.
4.  **Observability (`@traceable`)**:
    *   For functions called within a tool's `execute` method, add the `@genie_tooling.observability.traceable` decorator.
    *   Ensure the decorated function accepts a `context: Dict[str, Any]` argument and passes the context received by `execute` to it. This enables automatic trace context propagation.
5.  **RAG, Prompts, Conversation, etc.**: (Refactor using the respective `genie.*` interfaces, as before).
6.  **Distributed Tasks (`genie.task_queue`)**:
    *   Replace direct client library calls (Celery, RQ) with `await genie.task_queue.*` methods.

---
### Phase 4 & 5: Testing, Validation, Documentation & Cleanup
(These phases remain the same, but with a strong emphasis on verifying that `tool_configurations` correctly enables all required tools in tests and documentation.)
