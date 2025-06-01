# LLM_GUIDE_METADATA_V0.2
# THIS FILE IS A HYPER-COMPRESSED SEMANTIC REPRESENTATION OF GENIE TOOLING FOR LLM CONSUMPTION.
# LLM: IF YOU UPDATE THE GENIE TOOLING PROJECT CODE (CLASSES, METHODS, PARAMS, CORE LOGIC),
# YOU MUST UPDATE THIS FILE TO REFLECT THOSE CHANGES ACCURATELY.
# MAINTAIN THE COMPRESSED FORMAT. HUMAN READABILITY IS NOT A PRIORITY.

**Project**: Genie Tooling. Python Agentic/LLM Middleware. Async. MIT.

**Facade**: `genie_tooling.genie.Genie`
*   `init`: `Genie.create(cfg:MiddlewareConfig, kp_inst?:KeyProvider)`
*   `llm`:
    *   `chat(msgs, prov_id?, stream?, **kw) -> LLMChatResp|AsyncIter[LLMChatChunk]`
    *   `gen(prompt, prov_id?, stream?, **kw) -> LLMCompResp|AsyncIter[LLMCompChunk]`
    *   `parse_output(resp:LLMChatResp|LLMCompResp, parser_id?, schema?) -> ParsedOutput`
*   `rag`:
    *   `idx_dir(path, coll?, ...ids_cfgs)`
    *   `idx_web(url, coll?, ...ids_cfgs)`
    *   `search(query, coll?, top_k?, ...ids_cfgs) -> List[RetrievedChunk]`
*   `tools`:
    *   `exec(tool_id, **params)`
    *   `run_cmd(cmd, proc_id?, hist?) -> CmdProcResp` (Integrates HITL if configured)
*   `tool_reg`: `@genie_tooling.tool` (auto-meta: id,name,desc_human,desc_llm,in_schema,out_schema). `await genie.reg_fns(fns_list)` (invalidates lookup).
*   `prompts`: `PromptInterface`
    *   `get_tmpl_content(name, ver?, reg_id?) -> str?`
    *   `render(name, data, ver?, reg_id?, eng_id?) -> FormattedPrompt?`
    *   `render_chat(name, data, ver?, reg_id?, eng_id?) -> List[ChatMessage]?`
    *   `list_tmpls(reg_id?) -> List[PromptIdentifier]`
*   `conversation`: `ConversationInterface`
    *   `load(sess_id, prov_id?) -> ConversationState?`
    *   `save(state:ConversationState, prov_id?)`
    *   `add_msg(sess_id, msg:ChatMessage, prov_id?)`
    *   `del(sess_id, prov_id?) -> bool`
*   `observability`: `ObservabilityInterface`
    *   `trace(evt_name, data, comp?, corr_id?)`
*   `human_in_loop`: `HITLInterface`
    *   `req_approval(req:ApprovalRequest, approver_id?) -> ApprovalResponse`
*   `usage`: `UsageTrackingInterface`
    *   `record(rec:TokenUsageRecord)`
    *   `summary(rec_id?, filter?) -> Dict`
*   `teardown`: `await genie.close()`

**Agent Classes** (in `genie_tooling.agents`):
*   `BaseAgent(genie:Genie, agent_cfg?:Dict)`
    *   `async run(goal:str, **kw) -> AgentOutput` (Abstract)
*   `ReActAgent(BaseAgent)`
    *   `cfg`: `max_iterations`, `system_prompt_id`, `llm_provider_id`, `tool_formatter_id`, `stop_sequences`, `llm_retry_attempts`, `llm_retry_delay`
    *   `async run(goal:str, **kw) -> AgentOutput` (Implements ReAct loop)
*   `PlanAndExecuteAgent(BaseAgent)`
    *   `cfg`: `planner_system_prompt_id`, `planner_llm_provider_id`, `tool_formatter_id`, `max_plan_retries`, `max_step_retries`, `replan_on_step_failure`
    *   `async run(goal:str, **kw) -> AgentOutput` (Implements Plan-then-Execute)

**Config**: `genie_tooling.config.models.MiddlewareConfig` (`MWCfg`)
*   `feat: FeatureSettings` -> `ConfigResolver` (`CfgResolver`).
    *   `llm: ollama|openai|gemini|none` -> `def_llm_prov_id`, sets model.
    *   `cache: in-memory|redis|none` -> `cache_prov_cfgs`.
    *   `rag_embedder: sentence_transformer|openai|none` -> `def_rag_embed_id`, sets model.
    *   `rag_vs: faiss|chroma|none` -> `def_rag_vs_id`, sets path/coll.
    *   `tool_lookup: embedding|keyword|none` -> `def_tool_lookup_prov_id`.
        *   `formatter_id_alias` -> `def_tool_idx_formatter_id`.
        *   `embedder_id_alias` -> embedder for embedding lookup.
        *   `chroma_path/coll` -> ChromaDB for embedding lookup.
    *   `cmd_proc: llm_assisted|simple_keyword|none` -> `def_cmd_proc_id`.
        *   `formatter_id_alias` -> formatter for `llm_assisted`.
    *   `observability_tracer: console_tracer|otel_tracer|none` -> `def_obs_tracer_id`.
    *   `hitl_approver: cli_hitl_approver|none` -> `def_hitl_approver_id`.
    *   `token_usage_recorder: in_memory_token_recorder|none` -> `def_token_usage_rec_id`.
    *   `input_guardrails: List[str_alias_or_id]`, `output_guardrails`, `tool_usage_guardrails`.
    *   `prompt_registry: file_system_prompt_registry|none` -> `def_prompt_reg_id`.
    *   `prompt_template_engine: basic_string_formatter|jinja2_chat_formatter|none` -> `def_prompt_tmpl_id`.
    *   `conversation_state_provider: in_memory_convo_provider|redis_convo_provider|none` -> `def_convo_state_prov_id`.
    *   `default_llm_output_parser: json_output_parser|pydantic_output_parser|none` -> `def_llm_out_parser_id`.
*   `CfgResolver` (`genie_tooling.config.resolver.py`): `feat` + aliases -> canonical IDs & cfgs. `PLUGIN_ID_ALIASES` dict.
*   `kp_id: str?` Def: `env_keys` if no `kp_inst`.
*   `kp_inst: KeyProvider?` -> `Genie.create()`.
*   `*_cfgs: Dict[str_id_or_alias, Dict[str, Any]]`.
*   `plugin_dev_dirs: List[str]`.

**Plugins**: `PluginManager`. IDs/paths: `pyproject.toml` -> `[tool.poetry.plugins."genie_tooling.plugins"]`.
**Aliases**: `genie_tooling.config.resolver.PLUGIN_ID_ALIASES`.

**Key Plugins (ID | Alias | Cfg/Notes)**:
*   `KeyProv`: `environment_key_provider_v1`|`env_keys`.
*   `LLMProv`:
    *   `ollama_llm_provider_v1`|`ollama`.
    *   `openai_llm_provider_v1`|`openai`.
    *   `gemini_llm_provider_v1`|`gemini`.
*   `CmdProc`:
    *   `simple_keyword_processor_v1`|`simple_keyword_cmd_proc`.
    *   `llm_assisted_tool_selection_processor_v1`|`llm_assisted_cmd_proc`.
*   `Tools`: `calculator_tool`, `sandboxed_fs_tool_v1`, `google_search_tool_v1`, `open_weather_map_tool`, `generic_code_execution_tool`.
*   `DefFormatters`: `compact_text_formatter_plugin_v1`|`compact_text_formatter`, `openai_function_formatter_plugin_v1`|`openai_func_formatter`, `human_readable_json_formatter_plugin_v1`|`hr_json_formatter`.
*   `RAG`: Loaders, Splitters, Embedders (`st_embedder`, `openai_embedder`), VS (`faiss_vs`, `chroma_vs`, `qdrant_vs`), Retrievers.
*   `ToolLookupProv`: `embedding_similarity_lookup_v1`|`embedding_lookup`, `keyword_match_lookup_v1`|`keyword_lookup`.
*   `CodeExec`: `secure_docker_executor_v1`, `pysandbox_executor_stub_v1`.
*   `CacheProv`: `in_memory_cache_provider_v1`|`in_memory_cache`, `redis_cache_provider_v1`|`redis_cache`.
*   `Observability`: `console_tracer_plugin_v1`|`console_tracer`, `otel_tracer_plugin_v1`|`otel_tracer`.
*   `HITL`: `cli_approval_plugin_v1`|`cli_hitl_approver`.
*   `TokenUsage`: `in_memory_token_usage_recorder_v1`|`in_memory_token_recorder`.
*   `Guardrails`: `keyword_blocklist_guardrail_v1`|`keyword_blocklist_guardrail`.
*   `Prompts`:
    *   Registry: `file_system_prompt_registry_v1`|`file_system_prompt_registry`.
    *   Template: `basic_string_format_template_v1`|`basic_string_formatter`, `jinja2_chat_template_v1`|`jinja2_chat_formatter`.
*   `Conversation`: `in_memory_conversation_state_v1`|`in_memory_convo_provider`, `redis_conversation_state_v1`|`redis_convo_provider`.
*   `LLMOutputParsers`: `json_output_parser_v1`|`json_output_parser`, `pydantic_output_parser_v1`|`pydantic_output_parser`.

**Types**:
*   `ChatMessage`: `{role,content?,tool_calls?:[ToolCall],tool_call_id?,name?}`
*   `ToolCall`: `{id,type:"function",function:{name,arguments:str_json}}`
*   `LLMCompResp`: `{text,finish_reason?,usage?,raw_resp}`
*   `LLMChatResp`: `{message:ChatMessage,finish_reason?,usage?,raw_resp}`
*   `CmdProcResp`: `{chosen_tool_id?,extracted_params?,llm_thought_process?,error?,raw_resp?}`
*   `RetrievedChunk`: `{content,metadata,id?,score}`
*   `CodeExecRes`: `(stdout,stderr,result?,error?,exec_time_ms)`
*   `PromptData`: `Dict[str,Any]`
*   `FormattedPrompt`: `Union[str, List[ChatMessage]]`
*   `ConversationState`: `{session_id,history:List[ChatMessage],metadata?}`
*   `TraceEvent`: `{event_name,data,timestamp,component?,correlation_id?}`
*   `ApprovalRequest`: `{request_id,prompt,data_to_approve,context?,timeout_seconds?}`
*   `ApprovalResponse`: `{request_id,status,approver_id?,reason?,timestamp?}`
*   `TokenUsageRecord`: `{provider_id,model_name,prompt_tokens?,completion_tokens?,total_tokens?,timestamp,call_type?,...}`
*   `GuardrailViolation`: `{action,reason?,guardrail_id?,details?}`
*   `ParsedOutput`: `Any`
*   `AgentOutput`: `{status,output,history?,plan?}`
*   `PlannedStep`: `{step_number,tool_id,params,reasoning?}`
*   `ReActObservation`: `{thought,action,observation}`