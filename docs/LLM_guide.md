## Genie Tooling - LLM Data Points

**Project**: Genie Tooling. Python Agentic/LLM Middleware. Async. MIT.

**Facade**: `genie_tooling.genie.Genie`
*   `init`: `Genie.create(cfg:MiddlewareConfig, kp_inst?:KeyProvider)`
*   `llm`:
    *   `chat(msgs, prov_id?, stream?, **kw) -> LLMChatResp|AsyncIter[LLMChatChunk]`
    *   `gen(prompt, prov_id?, stream?, **kw) -> LLMCompResp|AsyncIter[LLMCompChunk]`
*   `rag`:
    *   `idx_dir(path, coll?, ...ids_cfgs)`
    *   `idx_web(url, coll?, ...ids_cfgs)`
    *   `search(query, coll?, top_k?, ...ids_cfgs) -> List[RetrievedChunk]`
*   `tools`:
    *   `exec(tool_id, **params)`
    *   `run_cmd(cmd, proc_id?, hist?) -> CmdProcResp`
*   `tool_reg`: `@genie_tooling.tool` (auto-meta: id,name,desc_human,desc_llm,in_schema,out_schema). `await genie.reg_fns(fns_list)` (invalidates lookup).
*   `teardown`: `await genie.close()`

**Config**: `genie_tooling.config.models.MiddlewareConfig` (`MWCfg`)
*   `feat: FeatureSettings` -> `ConfigResolver` (`CfgResolver`).
    *   `llm: ollama|openai|gemini|none` -> `def_llm_prov_id`, sets model (e.g., `llm_ollama_model_name`).
    *   `cache: in-memory|redis|none` -> `cache_prov_cfgs` (e.g., `cache_redis_url`).
    *   `rag_embedder: sentence_transformer|openai|none` -> `def_rag_embed_id`, sets model (e.g., `rag_embedder_st_model_name`).
    *   `rag_vs: faiss|chroma|none` -> `def_rag_vs_id`, sets path/coll (e.g., `rag_vs_chroma_path`).
    *   `tool_lookup: embedding|keyword|none` -> `def_tool_lookup_prov_id`.
        *   `formatter_id_alias` -> `def_tool_idx_formatter_id`.
        *   `embedder_id_alias` -> embedder for embedding lookup.
        *   `chroma_path/coll` -> ChromaDB for embedding lookup.
    *   `cmd_proc: llm_assisted|simple_keyword|none` -> `def_cmd_proc_id`.
        *   `formatter_id_alias` -> formatter for `llm_assisted`.
*   `CfgResolver` (`genie_tooling.config.resolver.py`): `feat` + aliases -> canonical IDs & cfgs. `PLUGIN_ID_ALIASES` dict. Precedence: explicit_cfgs > explicit_defaults > feat.
*   `kp_id: str?` Def: `env_keys` if no `kp_inst`.
*   `kp_inst: KeyProvider?` -> `Genie.create()`.
*   `*_cfgs: Dict[str_id_or_alias, Dict[str, Any]]` (e.g., `llm_prov_cfgs`, `tool_cfgs`). Override feat.
*   `plugin_dev_dirs: List[str]`.

**Plugins**: `PluginManager`. IDs/paths: `pyproject.toml` -> `[tool.poetry.plugins."genie_tooling.plugins"]`.
**Aliases**: `genie_tooling.config.resolver.PLUGIN_ID_ALIASES`.

**Key Plugins (ID | Alias | Cfg/Notes)**:
*   `KeyProv`: `environment_key_provider_v1`|`env_keys`|Env.
*   `LLMProv`:
    *   `ollama_llm_provider_v1`|`ollama`|Cfg:`model_name,base_url`.Opt:`format="json"`.
    *   `openai_llm_provider_v1`|`openai`|Cfg:`model_name,openai_api_base`.Needs `OPENAI_API_KEY`.
    *   `gemini_llm_provider_v1`|`gemini`|Cfg:`model_name`.Needs `GOOGLE_API_KEY`.Func calling.
*   `CmdProc`:
    *   `simple_keyword_processor_v1`|`simple_keyword_cmd_proc`|Cfg:`keyword_map,keyword_priority`.Prompts.
    *   `llm_assisted_tool_selection_processor_v1`|`llm_assisted_cmd_proc`|LLM.Cfg:`tool_formatter_id,tool_lookup_top_k`.Uses `ToolLookupService`.
*   `Tools`:
    *   `calculator_tool`.
    *   `sandboxed_fs_tool_v1`|Cfg:`sandbox_base_path`.
    *   `google_search_tool_v1`|Needs `GOOGLE_API_KEY,GOOGLE_CSE_ID`.
    *   `open_weather_map_tool`|Needs `OPENWEATHERMAP_API_KEY`.
    *   `generic_code_execution_tool`|Uses `CodeExecutorPlugin`.
*   `DefFormatters`:
    *   `compact_text_formatter_plugin_v1`|`compact_text_formatter`.
    *   `openai_function_formatter_plugin_v1`|`openai_func_formatter`.
    *   `human_readable_json_formatter_plugin_v1`|`hr_json_formatter`.
*   `RAG`:
    *   Loaders: `file_system_loader_v1`, `web_page_loader_v1`(Cfg:`use_trafilatura`).
    *   Splitters: `character_recursive_text_splitter_v1`.
    *   Embedders: `sentence_transformer_embedder_v1`|`st_embedder`(Cfg:`model_name`), `openai_embedding_generator_v1`|`openai_embedder`(Needs `OPENAI_API_KEY`).
    *   VS: `faiss_vector_store_v1`|`faiss_vs`, `chromadb_vector_store_v1`|`chroma_vs`(Cfg:`path,collection_name`).
    *   Retrievers: `basic_similarity_retriever_v1`.
*   `ToolLookupProv`:
    *   `embedding_similarity_lookup_v1`|`embedding_lookup`|Uses embedder(`tool_lookup_embedder_id_alias`) & opt VS.
    *   `keyword_match_lookup_v1`|`keyword_lookup`.
*   `CodeExec`:
    *   `secure_docker_executor_v1`|Docker.Cfg:`*_docker_image,pull_images_on_setup`.
    *   `pysandbox_executor_stub_v1`|STUB,INSECURE(`exec()`).
*   `CacheProv`:
    *   `in_memory_cache_provider_v1`|`in_memory_cache`|Cfg:`max_size,default_ttl_seconds`.
    *   `redis_cache_provider_v1`|`redis_cache`|Cfg:`redis_url,default_ttl_seconds`.

**Types**:
*   `ChatMessage`: `{role,content?,tool_calls?:[ToolCall],tool_call_id?,name?}`
*   `ToolCall`: `{id,type:"function",function:{name,arguments:str_json}}`
*   `LLMCompResp`: `{text,finish_reason?,usage?,raw_resp}`
*   `LLMChatResp`: `{message:ChatMessage,finish_reason?,usage?,raw_resp}`
*   `CmdProcResp`: `{chosen_tool_id?,extracted_params?,llm_thought_process?,error?,raw_resp?}`
*   `RetrievedChunk`: `{content,metadata,id?,score}`
*   `CodeExecRes`: `(stdout,stderr,result?,error?,exec_time_ms)`