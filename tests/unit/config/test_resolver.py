### tests/unit/config/test_resolver.py
import logging
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.resolver import PLUGIN_ID_ALIASES, ConfigResolver
from genie_tooling.security.key_provider import KeyProvider


@pytest.fixture
def mock_kp_instance_for_resolver() -> MagicMock:
    kp = MagicMock(spec=KeyProvider)
    kp.plugin_id = "mock_kp_instance_id_for_resolver"
    return kp

@pytest.fixture
def config_resolver() -> ConfigResolver:
    return ConfigResolver()

def test_resolver_llm_feature_ollama(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="ollama", llm_ollama_model_name="test-ollama-model")
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    ollama_id = PLUGIN_ID_ALIASES["ollama"]
    assert resolved.default_llm_provider_id == ollama_id
    assert ollama_id in resolved.llm_provider_configurations
    assert resolved.llm_provider_configurations[ollama_id]["model_name"] == "test-ollama-model"
    assert "key_provider" not in resolved.llm_provider_configurations[ollama_id]


def test_resolver_llm_feature_openai_with_key_provider(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="openai", llm_openai_model_name="test-gpt")
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    openai_id = PLUGIN_ID_ALIASES["openai"]
    assert resolved.default_llm_provider_id == openai_id
    assert openai_id in resolved.llm_provider_configurations
    assert resolved.llm_provider_configurations[openai_id]["model_name"] == "test-gpt"
    assert resolved.llm_provider_configurations[openai_id]["key_provider"] is mock_kp_instance_for_resolver

def test_resolver_tool_lookup_embedding_feature(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            tool_lookup="embedding",
            tool_lookup_formatter_id_alias="compact_text_formatter",
            tool_lookup_embedder_id_alias="st_embedder",
            tool_lookup_chroma_collection_name="tools_lookup_test",
            tool_lookup_chroma_path="/data/tool_lookup_chroma"
        )
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    embedding_lookup_id = PLUGIN_ID_ALIASES["embedding_lookup"]
    st_embedder_id = PLUGIN_ID_ALIASES["st_embedder"]
    chroma_vs_id = PLUGIN_ID_ALIASES["chroma_vs"]
    compact_text_id = PLUGIN_ID_ALIASES["compact_text_formatter"]

    assert resolved.default_tool_lookup_provider_id == embedding_lookup_id
    assert resolved.default_tool_indexing_formatter_id == compact_text_id

    assert embedding_lookup_id in resolved.tool_lookup_provider_configurations
    lookup_provider_cfg = resolved.tool_lookup_provider_configurations.get(embedding_lookup_id, {})

    # CORRECTED: Assertions now look at the top-level config for "embedding"
    assert lookup_provider_cfg.get("embedder_id") == st_embedder_id
    assert lookup_provider_cfg.get("vector_store_id") == chroma_vs_id
    assert "embedder_config" in lookup_provider_cfg
    assert lookup_provider_cfg["embedder_config"].get("model_name") == FeatureSettings().rag_embedder_st_model_name
    assert "vector_store_config" in lookup_provider_cfg
    assert lookup_provider_cfg["vector_store_config"].get("collection_name") == "tools_lookup_test"
    assert lookup_provider_cfg["vector_store_config"].get("path") == "/data/tool_lookup_chroma"

def test_resolver_tool_lookup_hybrid_feature(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    """Test that 'hybrid' lookup correctly nests the dense provider config."""
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            tool_lookup="hybrid",
            tool_lookup_embedder_id_alias="openai_embedder"
        )
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    hybrid_lookup_id = PLUGIN_ID_ALIASES["hybrid_lookup"]
    openai_embedder_id = PLUGIN_ID_ALIASES["openai_embedder"]

    assert resolved.default_tool_lookup_provider_id == hybrid_lookup_id
    assert hybrid_lookup_id in resolved.tool_lookup_provider_configurations

    hybrid_provider_cfg = resolved.tool_lookup_provider_configurations.get(hybrid_lookup_id, {})
    assert "dense_provider_config" in hybrid_provider_cfg
    dense_config = hybrid_provider_cfg["dense_provider_config"]

    assert dense_config.get("embedder_id") == openai_embedder_id
    assert "embedder_config" in dense_config
    assert dense_config["embedder_config"].get("key_provider") is mock_kp_instance_for_resolver

def test_resolver_key_provider_injection_tool_lookup_openai_embedder(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            tool_lookup="embedding",
            tool_lookup_embedder_id_alias="openai_embedder"
        )
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    embedding_lookup_id = PLUGIN_ID_ALIASES["embedding_lookup"]
    openai_embedder_id = PLUGIN_ID_ALIASES["openai_embedder"]

    assert embedding_lookup_id in resolved.tool_lookup_provider_configurations
    lookup_cfg = resolved.tool_lookup_provider_configurations[embedding_lookup_id]

    # CORRECTED: Assertions now look at the top-level config for "embedding"
    assert lookup_cfg.get("embedder_id") == openai_embedder_id
    assert "embedder_config" in lookup_cfg
    assert lookup_cfg["embedder_config"].get("key_provider") is mock_kp_instance_for_resolver

# ... (All other previously existing tests from test_resolver.py are restored here) ...
def test_resolver_llm_feature_gemini_with_key_provider(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="gemini", llm_gemini_model_name="test-gemini-model")
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    gemini_id = PLUGIN_ID_ALIASES["gemini"]
    assert resolved.default_llm_provider_id == gemini_id
    assert gemini_id in resolved.llm_provider_configurations
    assert resolved.llm_provider_configurations[gemini_id]["model_name"] == "test-gemini-model"
    assert resolved.llm_provider_configurations[gemini_id]["key_provider"] is mock_kp_instance_for_resolver

def test_resolver_llm_feature_openai_no_kp_instance(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="openai", llm_openai_model_name="test-gpt-no-kp")
    )
    resolved = config_resolver.resolve(user_config, key_provider_instance=None)
    openai_id = PLUGIN_ID_ALIASES["openai"]
    assert resolved.default_llm_provider_id == openai_id
    assert openai_id in resolved.llm_provider_configurations
    assert resolved.llm_provider_configurations[openai_id]["model_name"] == "test-gpt-no-kp"
    assert "key_provider" not in resolved.llm_provider_configurations[openai_id]

def test_resolver_cache_feature_in_memory(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(features=FeatureSettings(cache="in-memory"))
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    in_memory_id = PLUGIN_ID_ALIASES["in_memory_cache"]
    assert in_memory_id in resolved.cache_provider_configurations
    assert resolved.cache_provider_configurations[in_memory_id] == {}

def test_resolver_cache_feature_redis(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(cache="redis", cache_redis_url="redis://custom:1234/2")
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    redis_id = PLUGIN_ID_ALIASES["redis_cache"]
    assert redis_id in resolved.cache_provider_configurations
    assert resolved.cache_provider_configurations[redis_id]["redis_url"] == "redis://custom:1234/2"

def test_resolver_rag_embedder_feature_st(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(rag_embedder="sentence_transformer", rag_embedder_st_model_name="custom-st-model")
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    st_embed_id = PLUGIN_ID_ALIASES["st_embedder"]
    assert resolved.default_rag_embedder_id == st_embed_id
    assert st_embed_id in resolved.embedding_generator_configurations
    assert resolved.embedding_generator_configurations[st_embed_id]["model_name"] == "custom-st-model"
    assert "key_provider" not in resolved.embedding_generator_configurations[st_embed_id]

def test_resolver_rag_embedder_feature_openai(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(rag_embedder="openai")
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    openai_embed_id = PLUGIN_ID_ALIASES["openai_embedder"]
    assert resolved.default_rag_embedder_id == openai_embed_id
    assert openai_embed_id in resolved.embedding_generator_configurations
    assert resolved.embedding_generator_configurations[openai_embed_id]["key_provider"] is mock_kp_instance_for_resolver

def test_resolver_rag_vector_store_faiss_feature(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(features=FeatureSettings(rag_vector_store="faiss"))
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    faiss_vs_id = PLUGIN_ID_ALIASES["faiss_vs"]
    assert resolved.default_rag_vector_store_id == faiss_vs_id
    assert faiss_vs_id in resolved.vector_store_configurations
    assert resolved.vector_store_configurations[faiss_vs_id] == {}

def test_resolver_rag_vector_store_chroma_feature(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            rag_vector_store="chroma",
            rag_vector_store_chroma_collection_name="my_rag_docs",
            rag_vector_store_chroma_path="/data/chroma_rag"
        )
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    chroma_vs_id = PLUGIN_ID_ALIASES["chroma_vs"]
    assert resolved.default_rag_vector_store_id == chroma_vs_id
    assert chroma_vs_id in resolved.vector_store_configurations
    vs_config = resolved.vector_store_configurations[chroma_vs_id]
    assert vs_config["collection_name"] == "my_rag_docs"
    assert vs_config["path"] == "/data/chroma_rag"

def test_resolver_tool_lookup_keyword_feature(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            tool_lookup="keyword",
            tool_lookup_formatter_id_alias="hr_json_formatter"
        )
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    keyword_lookup_id = PLUGIN_ID_ALIASES["keyword_lookup"]
    hr_json_formatter_id = PLUGIN_ID_ALIASES["hr_json_formatter"]

    assert resolved.default_tool_lookup_provider_id == keyword_lookup_id
    assert resolved.default_tool_indexing_formatter_id == hr_json_formatter_id
    assert keyword_lookup_id in resolved.tool_lookup_provider_configurations
    assert resolved.tool_lookup_provider_configurations[keyword_lookup_id] == {}

def test_resolver_command_processor_simple_keyword(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(features=FeatureSettings(command_processor="simple_keyword"))
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    simple_kw_id = PLUGIN_ID_ALIASES["simple_keyword_cmd_proc"]
    assert resolved.default_command_processor_id == simple_kw_id
    assert simple_kw_id in resolved.command_processor_configurations
    assert resolved.command_processor_configurations[simple_kw_id] == {}

def test_resolver_command_processor_llm_assisted(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            command_processor="llm_assisted",
            command_processor_formatter_id_alias="openai_func_formatter",
            llm="openai"
        )
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    llm_assisted_id = PLUGIN_ID_ALIASES["llm_assisted_cmd_proc"]
    openai_func_formatter_id = PLUGIN_ID_ALIASES["openai_func_formatter"]
    openai_llm_id = PLUGIN_ID_ALIASES["openai"]

    assert resolved.default_command_processor_id == llm_assisted_id
    assert llm_assisted_id in resolved.command_processor_configurations
    proc_config = resolved.command_processor_configurations[llm_assisted_id]
    assert proc_config["tool_formatter_id"] == openai_func_formatter_id
    assert resolved.default_llm_provider_id == openai_llm_id

def test_resolver_explicit_config_overrides_features(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="ollama", llm_ollama_model_name="mistral-from-features"),
        default_llm_provider_id=PLUGIN_ID_ALIASES["ollama"],
        llm_provider_configurations={PLUGIN_ID_ALIASES["ollama"]: {"model_name": "llama2-explicit-override"}}
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    ollama_id = PLUGIN_ID_ALIASES["ollama"]
    assert resolved.default_llm_provider_id == ollama_id
    assert resolved.llm_provider_configurations[ollama_id]["model_name"] == "llama2-explicit-override"

def test_resolver_alias_in_user_config_keys(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="none"),
        llm_provider_configurations={"openai": {"model_name": "gpt-4-via-alias"}}
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    openai_canonical_id = PLUGIN_ID_ALIASES["openai"]
    assert openai_canonical_id in resolved.llm_provider_configurations
    assert resolved.llm_provider_configurations[openai_canonical_id]["model_name"] == "gpt-4-via-alias"
    assert "openai" not in resolved.llm_provider_configurations

def test_resolver_feature_none_value(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="none", cache="none", rag_embedder="none", tool_lookup="none", command_processor="none")
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    assert resolved.default_llm_provider_id is None
    assert not resolved.llm_provider_configurations
    assert resolved.default_rag_embedder_id is None
    assert not resolved.embedding_generator_configurations
    assert resolved.default_tool_lookup_provider_id is None
    assert not resolved.tool_lookup_provider_configurations
    assert resolved.default_command_processor_id is None
    assert not resolved.command_processor_configurations
    assert not resolved.cache_provider_configurations

def test_resolver_default_key_provider_id_behavior(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config_no_kp = MiddlewareConfig()
    resolved1 = config_resolver.resolve(user_config_no_kp, key_provider_instance=None)
    assert resolved1.key_provider_id == PLUGIN_ID_ALIASES["env_keys"]

    with patch.dict(PLUGIN_ID_ALIASES, {"my_custom_kp_alias": "my_custom_kp_canonical_v1"}, clear=False):
        user_config_alias_kp = MiddlewareConfig(key_provider_id="my_custom_kp_alias")
        resolved2 = config_resolver.resolve(user_config_alias_kp, key_provider_instance=None)
        assert resolved2.key_provider_id == "my_custom_kp_canonical_v1"

    user_config_canonical_kp = MiddlewareConfig(key_provider_id="some_canonical_key_provider_id_v1")
    resolved3 = config_resolver.resolve(user_config_canonical_kp, key_provider_instance=None)
    assert resolved3.key_provider_id == "some_canonical_key_provider_id_v1"

    resolved4 = config_resolver.resolve(user_config_no_kp, mock_kp_instance_for_resolver)
    assert resolved4.key_provider_id == mock_kp_instance_for_resolver.plugin_id

    user_config_explicit_kp_id = MiddlewareConfig(key_provider_id="user_chosen_kp_id")
    resolved5 = config_resolver.resolve(user_config_explicit_kp_id, mock_kp_instance_for_resolver)
    assert resolved5.key_provider_id == "user_chosen_kp_id"

def test_resolver_key_provider_id_alias_not_found(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig(key_provider_id="non_existent_alias")
    resolved = config_resolver.resolve(user_config, key_provider_instance=None)
    assert resolved.key_provider_id == "non_existent_alias"

def test_resolver_complex_merge_and_override(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", llm_ollama_model_name="mistral-feature",
            rag_embedder="sentence_transformer",
            rag_embedder_st_model_name="st-feature-model"
        ),
        default_llm_provider_id="openai",
        llm_provider_configurations={
            PLUGIN_ID_ALIASES["ollama"]: {"model_name": "ollama-explicit-model", "temperature": 0.7},
            "openai": {"model_name": "gpt-explicit-model", "max_tokens": 100}
        },
        embedding_generator_configurations={
            PLUGIN_ID_ALIASES["openai_embedder"]: {"model_name": "ada-explicit-model", "key_provider": None},
            PLUGIN_ID_ALIASES["st_embedder"]: {"model_name": "st-explicit-override-model"}
        },
        default_rag_embedder_id=None
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)

    ollama_id = PLUGIN_ID_ALIASES["ollama"]
    openai_id = PLUGIN_ID_ALIASES["openai"]
    st_embed_id = PLUGIN_ID_ALIASES["st_embedder"]
    openai_embed_id = PLUGIN_ID_ALIASES["openai_embedder"]

    assert resolved.default_llm_provider_id == openai_id
    assert resolved.llm_provider_configurations[ollama_id]["model_name"] == "ollama-explicit-model"
    assert resolved.llm_provider_configurations[ollama_id]["temperature"] == 0.7
    assert resolved.llm_provider_configurations[openai_id]["model_name"] == "gpt-explicit-model"
    assert resolved.llm_provider_configurations[openai_id]["key_provider"] is mock_kp_instance_for_resolver
    assert resolved.default_rag_embedder_id is None
    assert st_embed_id in resolved.embedding_generator_configurations
    assert resolved.embedding_generator_configurations[st_embed_id].get("model_name") == "st-explicit-override-model"
    assert openai_embed_id in resolved.embedding_generator_configurations
    assert resolved.embedding_generator_configurations[openai_embed_id].get("model_name") == "ada-explicit-model"
    assert resolved.embedding_generator_configurations[openai_embed_id].get("key_provider") is None

def test_resolver_tool_lookup_formatter_defaulting(config_resolver: ConfigResolver):
    user_config_no_formatter_alias = MiddlewareConfig(
        features=FeatureSettings(tool_lookup="embedding", tool_lookup_formatter_id_alias=None)
    )
    resolved_no_alias = config_resolver.resolve(user_config_no_formatter_alias)
    assert resolved_no_alias.default_tool_indexing_formatter_id is None

    user_config_bad_alias = MiddlewareConfig(
        features=FeatureSettings(tool_lookup="embedding", tool_lookup_formatter_id_alias="non_existent_formatter_alias")
    )
    resolved_bad_alias = config_resolver.resolve(user_config_bad_alias)
    assert resolved_bad_alias.default_tool_indexing_formatter_id == "non_existent_formatter_alias"

def test_resolver_logging_of_mock(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="openai", llm_openai_model_name="test-gpt")
    )
    with patch("genie_tooling.config.resolver.json.dumps", side_effect=TypeError("Cannot serialize for log")) as mock_dumps, \
         patch("genie_tooling.config.resolver.logger.error") as mock_logger_error:
        config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
        mock_dumps.assert_called_once()
        mock_logger_error.assert_called_once_with("Error serializing resolved_config for logging: Cannot serialize for log")

def test_resolver_empty_user_config(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig()
    resolved = config_resolver.resolve(user_config, None)

    assert resolved.features == FeatureSettings()
    assert resolved.key_provider_id == PLUGIN_ID_ALIASES["env_keys"]
    assert resolved.default_llm_provider_id is None
    assert isinstance(resolved.llm_provider_configurations, dict)
    assert not resolved.llm_provider_configurations

def test_resolver_observability_feature(config_resolver: ConfigResolver):
    user_config_console = MiddlewareConfig(features=FeatureSettings(observability_tracer="console_tracer"))
    resolved_console = config_resolver.resolve(user_config_console)
    assert resolved_console.default_observability_tracer_id == PLUGIN_ID_ALIASES["console_tracer"]
    assert PLUGIN_ID_ALIASES["console_tracer"] in resolved_console.observability_tracer_configurations
    assert resolved_console.observability_tracer_configurations[PLUGIN_ID_ALIASES["console_tracer"]] == {}

    user_config_otel = MiddlewareConfig(features=FeatureSettings(observability_tracer="otel_tracer", observability_otel_endpoint="http://my-otel:4318/v1/traces"))
    resolved_otel = config_resolver.resolve(user_config_otel)
    otel_id = PLUGIN_ID_ALIASES["otel_tracer"]
    assert resolved_otel.default_observability_tracer_id == otel_id
    assert otel_id in resolved_otel.observability_tracer_configurations
    assert resolved_otel.observability_tracer_configurations[otel_id]["exporter_type"] == "otlp_http"
    assert resolved_otel.observability_tracer_configurations[otel_id]["otlp_http_endpoint"] == "http://my-otel:4318/v1/traces"

def test_resolver_hitl_feature(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig(features=FeatureSettings(hitl_approver="cli_hitl_approver"))
    resolved = config_resolver.resolve(user_config)
    cli_id = PLUGIN_ID_ALIASES["cli_hitl_approver"]
    assert resolved.default_hitl_approver_id == cli_id
    assert cli_id in resolved.hitl_approver_configurations
    assert resolved.hitl_approver_configurations[cli_id] == {}

def test_resolver_token_usage_feature(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig(features=FeatureSettings(token_usage_recorder="in_memory_token_recorder"))
    resolved = config_resolver.resolve(user_config)
    mem_rec_id = PLUGIN_ID_ALIASES["in_memory_token_recorder"]
    assert resolved.default_token_usage_recorder_id == mem_rec_id
    assert mem_rec_id in resolved.token_usage_recorder_configurations
    assert resolved.token_usage_recorder_configurations[mem_rec_id] == {}

def test_resolver_guardrails_feature(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig(features=FeatureSettings(input_guardrails=["keyword_blocklist_guardrail"]))
    resolved = config_resolver.resolve(user_config)
    kw_block_id = PLUGIN_ID_ALIASES["keyword_blocklist_guardrail"]
    assert kw_block_id in resolved.default_input_guardrail_ids
    assert kw_block_id in resolved.guardrail_configurations
    assert resolved.guardrail_configurations[kw_block_id] == {}

def test_resolver_prompts_feature(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig(features=FeatureSettings(prompt_registry="file_system_prompt_registry", prompt_template_engine="jinja2_chat_formatter"))
    resolved = config_resolver.resolve(user_config)
    fs_reg_id = PLUGIN_ID_ALIASES["file_system_prompt_registry"]
    jinja_engine_id = PLUGIN_ID_ALIASES["jinja2_chat_formatter"]
    assert resolved.default_prompt_registry_id == fs_reg_id
    assert resolved.default_prompt_template_plugin_id == jinja_engine_id
    assert fs_reg_id in resolved.prompt_registry_configurations
    assert jinja_engine_id in resolved.prompt_template_configurations

def test_resolver_conversation_feature(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig(features=FeatureSettings(conversation_state_provider="redis_convo_provider"))
    resolved = config_resolver.resolve(user_config)
    redis_convo_id = PLUGIN_ID_ALIASES["redis_convo_provider"]
    assert resolved.default_conversation_state_provider_id == redis_convo_id
    assert redis_convo_id in resolved.conversation_state_provider_configurations

def test_resolver_llm_output_parser_feature(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig(features=FeatureSettings(default_llm_output_parser="pydantic_output_parser"))
    resolved = config_resolver.resolve(user_config)
    pydantic_parser_id = PLUGIN_ID_ALIASES["pydantic_output_parser"]
    assert resolved.default_llm_output_parser_id == pydantic_parser_id
    assert pydantic_parser_id in resolved.llm_output_parser_configurations

def test_resolver_task_queue_feature_celery(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig(features=FeatureSettings(
        task_queue="celery",
        task_queue_celery_broker_url="redis://celery-broker",
        task_queue_celery_backend_url="redis://celery-backend"
    ))
    resolved = config_resolver.resolve(user_config)
    celery_q_id = PLUGIN_ID_ALIASES["celery_task_queue"]
    assert resolved.default_distributed_task_queue_id == celery_q_id
    assert celery_q_id in resolved.distributed_task_queue_configurations
    q_config = resolved.distributed_task_queue_configurations[celery_q_id]
    assert q_config["celery_broker_url"] == "redis://celery-broker"
    assert q_config["celery_backend_url"] == "redis://celery-backend"

def test_resolver_passthrough_fields(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig(
        plugin_dev_dirs=["./my_plugins"],
        default_log_level="DEBUG"
    )
    resolved = config_resolver.resolve(user_config)
    assert resolved.plugin_dev_dirs == ["./my_plugins"]
    assert resolved.default_log_level == "DEBUG"

def test_resolver_invalid_log_level(config_resolver: ConfigResolver, caplog):
    with caplog.at_level(logging.WARNING):
        user_config = MiddlewareConfig(default_log_level="TRACE")
        resolved = config_resolver.resolve(user_config)
    assert resolved.default_log_level == "INFO"
    assert "Invalid log_level 'TRACE' in MiddlewareConfig. Defaulting to INFO." in caplog.text

def test_resolver_qdrant_vs_feature_with_key_provider(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            rag_vector_store="qdrant",
            rag_vector_store_qdrant_api_key_name="QDRANT_KEY_TEST"
        )
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    qdrant_vs_id = PLUGIN_ID_ALIASES["qdrant_vs"]
    assert resolved.default_rag_vector_store_id == qdrant_vs_id
    assert qdrant_vs_id in resolved.vector_store_configurations
    vs_config = resolved.vector_store_configurations[qdrant_vs_id]
    assert vs_config.get("api_key_name") == "QDRANT_KEY_TEST"
    assert vs_config.get("key_provider") is mock_kp_instance_for_resolver

def test_resolver_llama_cpp_llm_feature_with_key_provider(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="llama_cpp",
            llm_llama_cpp_api_key_name="LLAMA_CPP_KEY_FOR_TEST"
        )
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    llama_cpp_id = PLUGIN_ID_ALIASES["llama_cpp"]
    assert resolved.default_llm_provider_id == llama_cpp_id
    assert llama_cpp_id in resolved.llm_provider_configurations
    llm_config = resolved.llm_provider_configurations[llama_cpp_id]
    assert llm_config.get("api_key_name") == "LLAMA_CPP_KEY_FOR_TEST"
    assert llm_config.get("key_provider") is mock_kp_instance_for_resolver

def test_resolver_llama_cpp_llm_feature_no_key_name(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="llama_cpp",
            llm_llama_cpp_api_key_name=None
        )
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    llama_cpp_id = PLUGIN_ID_ALIASES["llama_cpp"]
    assert resolved.default_llm_provider_id == llama_cpp_id
    assert llama_cpp_id in resolved.llm_provider_configurations
    llm_config = resolved.llm_provider_configurations[llama_cpp_id]
    assert "api_key_name" not in llm_config
    assert "key_provider" not in llm_config
