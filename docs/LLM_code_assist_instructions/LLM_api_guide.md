# LLM_GUIDE_METADATA_V1.0

# THIS FILE IS A HYPER-COMPRESSED SEMANTIC REPRESENTATION OF GENIE TOOLING FOR LLM CONSUMPTION.
# LLM: IF YOU UPDATE THE GENIE TOOLING PROJECT CODE (CLASSES, METHODS, PARAMS, CORE LOGIC),
# YOU MUST UPDATE THIS FILE TO REFLECT THOSE CHANGES ACCURATELY.
# MAINTAIN THE COMPRESSED FORMAT. HUMAN READABILITY IS NOT A PRIORITY.

**Project**: Genie Tooling. Python Agentic/LLM Middleware. Async. MIT.

**Facade**: `genie_tooling.genie.Genie`
*   `init`: `async Genie.create(config:MiddlewareConfig, key_provider_instance?:KeyProvider, plugin_manager?:PluginManager)`
*   `llm`: `LLMInterface`
    *   `chat(msgs:List[ChatMessage], prov_id?:str, stream?:bool, **kw) -> LLMChatResp|AsyncIter[LLMChatChunk]` (kwargs can include `output_schema` for GBNF with Llama.cpp)
    *   `generate(prompt:str, prov_id?:str, stream?:bool, **kw) -> LLMCompResp|AsyncIter[LLMCompChunk]` (kwargs can include `output_schema` for GBNF with Llama.cpp)
    *   `parse_output(resp:Union[LLMChatResp,LLMCompResp], parser_id?:str, schema?:Any) -> ParsedOutput`
*   `rag`: `RAGInterface`
    *   `index_directory(path:str, collection_name?:str, loader_id?:str, **kw) -> Dict[str,Any]`
    *   `index_web_page(url:str, collection_name?:str, **kw) -> Dict[str,Any]`
    *   `search(query:str, collection_name?:str, top_k?:int, retriever_id?:str, **kw) -> List[RetrievedChunk]`
*   **Tools**:
    *   `@genie_tooling.tool` (Decorator for functions. Auto-generates metadata).
    *   `await genie.register_tool_functions(functions:List[Callable])`
    *   `await genie.execute_tool(tool_identifier:str, context?:Dict, **params) -> Any`
    *   `await genie.run_command(command:str, processor_id?:str, conversation_history?:List, context_for_tools?:Dict) -> Any`
*   `prompts`: `PromptInterface`
*   `conversation`: `ConversationInterface`
*   `observability`: `ObservabilityInterface`
    *   `trace_event(event_name:str, data:Dict, component?:str, correlation_id?:str)`
*   `human_in_loop`: `HITLInterface`
*   `usage`: `UsageTrackingInterface`
*   `task_queue`: `TaskQueueInterface`
    *   `submit_task(task_name:str, args?:Tuple, kwargs?:Dict, queue_id?:str, task_options?:Dict) -> str?`
    *   `get_task_status(task_id:str, queue_id?:str) -> TaskStatus?`
    *   `get_task_result(task_id:str, queue_id?:str, timeout_seconds?:float) -> Any?`
    *   `revoke_task(task_id:str, queue_id?:str, terminate?:bool) -> bool`
*   `teardown`: `await genie.close()`

**Agent Classes** (in `genie_tooling.agents`):
*   `BaseAgent(genie:Genie, agent_cfg?:Dict)`
*   `ReActAgent(BaseAgent)`
*   `PlanAndExecuteAgent(BaseAgent)`
*   `DeepResearchAgent(BaseAgent)`
*   `MathProofAssistantAgent(BaseAgent)`

**Configuration**: `genie_tooling.config.models.MiddlewareConfig` (`MWCfg`)
*   **Key Security Parameter**: `auto_enable_registered_tools: bool` (Default: `True`. Set `False` for production.)
*   **Tool Enablement**: `tool_configurations: Dict[str, Dict]`. An entry in this dict, even with empty settings `{}`, is required to enable a tool when `auto_enable_registered_tools=False`.
*   `features: FeatureSettings`: High-level config toggles. Processed by `ConfigResolver`.
    *   `llm: Literal["ollama", "openai", "gemini", "llama_cpp", "llama_cpp_internal", "none"]`
    *   `command_processor: Literal["llm_assisted", "simple_keyword", "rewoo", "none"]`
    *   `tool_lookup: Literal["embedding", "keyword", "hybrid", "none"]`
    *   `rag_embedder`, `rag_vector_store`, `cache`, `logging_adapter`, `observability_tracer`, `hitl_approver`, `token_usage_recorder`, `guardrails`, `prompt_registry`, `prompt_template_engine`, `conversation_state_provider`, `default_llm_output_parser`, `task_queue`
    *   All features have associated detailed configuration fields (e.g., `llm_ollama_model_name`, `task_queue_celery_broker_url`).
*   `plugin_dev_dirs: List[str]`
*   `key_provider_id: str?` (or `key_provider_instance` to `Genie.create()`)
*   `*_configurations`: Dictionaries for detailed plugin settings (e.g., `llm_provider_configurations`).

**Plugins**: Discovered via `pyproject.toml` entry points (`genie_tooling.plugins` and `genie_tooling.bootstrap`) and `plugin_dev_dirs`.
*   **Bootstrap Plugins**: A new type for extending Genie. Registered under `genie_tooling.bootstrap`. Implements `async def bootstrap(self, genie: Genie)`.
*   **Canonical IDs & Aliases**: `genie_tooling.config.resolver.PLUGIN_ID_ALIASES` maps short names (e.g., `ollama`) to full IDs (e.g., `ollama_llm_provider_v1`).

**Observability Decorator**: `@genie_tooling.observability.traceable`
*   Wraps async or sync functions.
*   Requires a `context: Dict[str, Any]` kwarg to receive trace context.
*   Automatically creates and links OpenTelemetry spans.
