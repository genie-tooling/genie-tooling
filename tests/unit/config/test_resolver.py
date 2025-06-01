### tests/unit/config/test_resolver.py
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
    # Ollama typically doesn't need a KP, so it shouldn't be injected by default by features
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
    assert lookup_provider_cfg.get("embedder_id") == st_embedder_id
    assert lookup_provider_cfg.get("vector_store_id") == chroma_vs_id

    assert "embedder_config" in lookup_provider_cfg
    assert lookup_provider_cfg["embedder_config"].get("model_name") == FeatureSettings().rag_embedder_st_model_name

    assert "vector_store_config" in lookup_provider_cfg
    assert lookup_provider_cfg["vector_store_config"].get("collection_name") == "tools_lookup_test"
    assert lookup_provider_cfg["vector_store_config"].get("path") == "/data/tool_lookup_chroma"

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
            llm="openai" # Ensure LLM is active for LLM-assisted processor
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
        default_llm_provider_id="openai", # Explicit default (alias)
        llm_provider_configurations={
            PLUGIN_ID_ALIASES["ollama"]: {"model_name": "ollama-explicit-model", "temperature": 0.7},
            "openai": {"model_name": "gpt-explicit-model", "max_tokens": 100} # User provides config for this
        },
        embedding_generator_configurations={
            # Explicit config for openai_embedder
            PLUGIN_ID_ALIASES["openai_embedder"]: {"model_name": "ada-explicit-model", "key_provider": None},
            # User explicitly wants to override st_embedder config from features
            PLUGIN_ID_ALIASES["st_embedder"]: {"model_name": "st-explicit-override-model"}
        },
        default_rag_embedder_id=None # User explicitly sets default RAG embedder to None
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)

    ollama_id = PLUGIN_ID_ALIASES["ollama"]
    openai_id = PLUGIN_ID_ALIASES["openai"]
    st_embed_id = PLUGIN_ID_ALIASES["st_embedder"]
    openai_embed_id = PLUGIN_ID_ALIASES["openai_embedder"]

    # LLM Checks
    assert resolved.default_llm_provider_id == openai_id
    assert resolved.llm_provider_configurations[ollama_id]["model_name"] == "ollama-explicit-model"
    assert resolved.llm_provider_configurations[ollama_id]["temperature"] == 0.7
    assert resolved.llm_provider_configurations[openai_id]["model_name"] == "gpt-explicit-model"
    # Check if key_provider was injected into openai LLM config
    assert resolved.llm_provider_configurations[openai_id]["key_provider"] is mock_kp_instance_for_resolver

    # RAG Embedder Checks
    assert resolved.default_rag_embedder_id is None # User explicitly set to None

    # Check ST embedder config (user explicit override)
    assert st_embed_id in resolved.embedding_generator_configurations
    assert resolved.embedding_generator_configurations[st_embed_id].get("model_name") == "st-explicit-override-model"

    # Check OpenAI embedder config (user explicit)
    assert openai_embed_id in resolved.embedding_generator_configurations
    assert resolved.embedding_generator_configurations[openai_embed_id].get("model_name") == "ada-explicit-model"
    # User explicitly set key_provider to None for openai_embedder, this should be respected over instance.
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
    # If alias is not found, the resolver should now use the alias itself as the ID
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

    try:
        config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    except Exception as e_normal:
        pytest.fail(f"Resolver normal logging failed: {e_normal}")

def test_resolver_empty_user_config(config_resolver: ConfigResolver):
    user_config = MiddlewareConfig()
    resolved = config_resolver.resolve(user_config, None)

    assert resolved.features == FeatureSettings()
    assert resolved.key_provider_id == PLUGIN_ID_ALIASES["env_keys"]
    assert resolved.default_llm_provider_id is None
    assert isinstance(resolved.llm_provider_configurations, dict)
    assert not resolved.llm_provider_configurations
