# GenieTooling LLMGuide vHyperCompressed
Facade:genie.Genie
Init:Genie.create(cfg:MWCfg,kp?:KeyProv)
LLM:
 chat(msgs,provID?,stream?,**kw)->ChatResp|AIter[ChatChunk]
 gen(prompt,provID?,stream?,**kw)->CompResp|AIter[CompChunk]
RAG:
 idx_dir(path,coll?,...ids_cfgs)
 idx_web(url,coll?,...ids_cfgs)
 search(q,coll?,topK?,...ids_cfgs)->[RetChunk]
Tools:
 exec(toolID,**params)
 run_cmd(cmd,procID?,hist?)->CmdProcResp
ToolReg:@tool->meta(id,nm,dsc_h,dsc_l,in_sch,out_sch). genie.reg_fns([fns])->inval_lookup
Teardown:genie.close()

Config:MWCfg(config.models.MiddlewareConfig)
 FeatSets(config.features.FeatureSettings)->CfgResolver(config.resolver.ConfigResolver)
  llm:str(ollama|openai|gemini|none)->def_llm_prov_id,sets_model_name(e.g.llm_ollama_model_name)
  cache:str(in-memory|redis|none)->cache_prov_cfgs(e.g.cache_redis_url)
  rag_emb:str(sentence_transformer|openai|none)->def_rag_emb_id,sets_model(e.g.rag_emb_st_model_name)
  rag_vs:str(faiss|chroma|none)->def_rag_vs_id,sets_path/coll(e.g.rag_vs_chroma_path)
  tool_lkp:str(embedding|keyword|none)->def_tool_lkp_prov_id
   fmt_alias->def_tool_idx_fmt_id
   emb_alias->emb_for_emb_lkp
   chroma_pth/coll->Chroma_for_emb_lkp
  cmd_proc:str(llm_assisted|simple_keyword|none)->def_cmd_proc_id
   fmt_alias->fmt_for_llm_assist
 CfgResolver:feat+aliases->canon_ids&cfgs. PLUGIN_ID_ALIASES dict. Prec:expl_cfg>expl_def>feat.
 kp_id:str? Def:env_keys if no kp_inst.
 kp_inst:KeyProv?->Genie.create().
 *_cfgs:Dict[id|alias,Dict] (e.g.llm_prov_cfgs). Override feat.
 plugin_dev_dirs:[str].

Plugins:PluginMgr.IDs/paths:pyproject.toml->[tool.poetry.plugins."genie_tooling.plugins"]
Aliases:config.resolver.PLUGIN_ID_ALIASES

KeyPlugins(ID|Alias|Cfg/Notes):
 KeyProv:env_key_prov_v1|env_keys|Env.
 LLMProv:
  ollama_llm_prov_v1|ollama|Cfg:model,base_url.Opt:fmt="json".
  openai_llm_prov_v1|openai|Cfg:model,api_base.Needs:OPENAI_API_KEY.
  gemini_llm_prov_v1|gemini|Cfg:model.Needs:GOOGLE_API_KEY.FuncCall.
 CmdProc:
  simple_kw_proc_v1|simple_kw_cmd_proc|Cfg:kw_map,kw_pri.Prompts.
  llm_assist_tool_sel_proc_v1|llm_assist_cmd_proc|LLM.Cfg:tool_fmt_id,tool_lkp_top_k.Uses:ToolLookupSvc.
 Tools:calc_tool. sandbox_fs_v1(Cfg:sandbox_base). google_search_v1(Needs:GOOG_API_KEY,CSE_ID). openweather_tool(Needs:OWM_API_KEY). gen_code_exec(Uses:CodeExecPlugin).
 DefFmt:compact_txt_fmt_v1|compact_txt_fmt. openai_fn_fmt_v1|openai_fn_fmt. hr_json_fmt_v1|hr_json_fmt.
 RAG:
  Load:fs_load_v1,web_load_v1(Cfg:use_traf).
  Split:char_recur_split_v1.
  Emb:st_emb_v1|st_emb(Cfg:model),openai_emb_gen_v1|openai_emb(Needs:OPENAI_API_KEY).
  VS:faiss_vs_v1|faiss_vs,chroma_vs_v1|chroma_vs(Cfg:path,coll).
  Ret:basic_sim_ret_v1.
 ToolLkpProv:emb_sim_lkp_v1|emb_lkp(Uses:emb(tool_lkp_emb_id_alias)&optVS). kw_match_lkp_v1|kw_lkp.
 CodeExec:sec_docker_exec_v1|Docker(Cfg:*_img,pull_setup). pysandbox_stub_v1|STUB,INSECURE(exec()).
 CacheProv:in_mem_cache_v1|in_mem_cache(Cfg:max_sz,def_ttl). redis_cache_v1|redis_cache(Cfg:redis_url,def_ttl).

Types:
 ChatMsg:{role,cont?,tool_calls?:[ToolCall],tool_call_id?,nm?}
 ToolCall:{id,type:"fn",fn:{nm,args:str_json}}
 LLMCompResp:{txt,fin_rsn?,use?,raw}
 LLMChatResp:{msg:ChatMsg,fin_rsn?,use?,raw}
 CmdProcResp:{tool_id?,params?,thought?,err?,raw?}
 RetChunk:{cont,meta,id?,score}
 CodeExecRes:(out,err,res?,error?,time_ms)
