# src/genie_tooling/genie.py
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

from .config.models import MiddlewareConfig
from .config.resolver import PLUGIN_ID_ALIASES, ConfigResolver
from .core.plugin_manager import PluginManager
from .core.types import (
    Plugin as CorePluginType,
)
from .core.types import RetrievedChunk
from .invocation.invoker import ToolInvoker
from .lookup.service import ToolLookupService
from .rag.manager import RAGManager
from .security.key_provider import KeyProvider
from .tools.abc import Tool as ToolPlugin
from .tools.manager import ToolManager

try:
    from .llm_providers.manager import LLMProviderManager
    from .llm_providers.types import ChatMessage, LLMChatResponse, LLMCompletionResponse
except ImportError:
    LLMProviderManager = type("LLMProviderManager", (), {}) # type: ignore
    ChatMessage = Dict # type: ignore
    LLMChatResponse = Any # type: ignore
    LLMCompletionResponse = Any # type: ignore
    logging.debug("LLMProviderManager or its types not found, using placeholders.")

try:
    from .command_processors.manager import CommandProcessorManager
    from .command_processors.types import CommandProcessorResponse
except ImportError:
    CommandProcessorManager = type("CommandProcessorManager", (), {}) # type: ignore
    CommandProcessorResponse = Any # type: ignore
    logging.debug("CommandProcessorManager or CommandProcessorResponse not found, using placeholders.")

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self, llm_provider_manager: LLMProviderManager, default_provider_id: Optional[str]):
        self._llm_provider_manager = llm_provider_manager
        self._default_provider_id = default_provider_id
        logger.debug(f"LLMInterface initialized with default provider ID: {self._default_provider_id}")

    async def generate(self, prompt: str, provider_id: Optional[str] = None, **kwargs: Any) -> LLMCompletionResponse:
        provider_to_use = provider_id or self._default_provider_id
        if not provider_to_use:
            raise ValueError("No LLM provider ID specified and no default is set for generate.")
        provider = await self._llm_provider_manager.get_llm_provider(provider_to_use)
        if not provider:
            raise RuntimeError(f"LLM Provider '{provider_to_use}' not found or failed to load.")
        return await provider.generate(prompt, **kwargs)

    async def chat(self, messages: List[ChatMessage], provider_id: Optional[str] = None, **kwargs: Any) -> LLMChatResponse:
        provider_to_use = provider_id or self._default_provider_id
        if not provider_to_use:
            raise ValueError("No LLM provider ID specified and no default is set for chat.")
        provider = await self._llm_provider_manager.get_llm_provider(provider_to_use)
        if not provider:
            raise RuntimeError(f"LLM Provider '{provider_to_use}' not found or failed to load.")
        return await provider.chat(messages, **kwargs)

class RAGInterface:
    def __init__(self, rag_manager: RAGManager, config: MiddlewareConfig, key_provider: KeyProvider):
        self._rag_manager = rag_manager
        self._config = config
        self._key_provider = key_provider
        logger.debug("RAGInterface initialized.")

    async def index_directory(
        self, path: str, collection_name: Optional[str] = None,
        loader_id: Optional[str] = None, splitter_id: Optional[str] = None,
        embedder_id: Optional[str] = None, vector_store_id: Optional[str] = None,
        loader_config: Optional[Dict[str, Any]] = None,
        splitter_config: Optional[Dict[str, Any]] = None,
        embedder_config: Optional[Dict[str, Any]] = None,
        vector_store_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        final_loader_id = loader_id or self._config.default_rag_loader_id or "file_system_loader_v1"
        final_splitter_id = splitter_id or self._config.default_rag_splitter_id or "character_recursive_text_splitter_v1"
        final_embedder_id = embedder_id or self._config.default_rag_embedder_id
        final_vector_store_id = vector_store_id or self._config.default_rag_vector_store_id

        def get_base_config(plugin_id: Optional[str], config_map_name: str) -> Dict[str, Any]:
            if plugin_id and hasattr(self._config, config_map_name):
                return getattr(self._config, config_map_name).get(plugin_id, {})
            return {}

        final_loader_config = {**get_base_config(final_loader_id, "document_loader_configurations"), **(loader_config or {}), **kwargs.get("loader_config_override", {})}
        final_splitter_config = {**get_base_config(final_splitter_id, "text_splitter_configurations"), **(splitter_config or {}), **kwargs.get("splitter_config_override", {})}
        final_embedder_config = {**get_base_config(final_embedder_id, "embedding_generator_configurations"), **(embedder_config or {}), **kwargs.get("embedder_config_override", {})}
        final_vector_store_config = {**get_base_config(final_vector_store_id, "vector_store_configurations"), **(vector_store_config or {}), **kwargs.get("vector_store_config_override", {})}

        if "key_provider" not in final_embedder_config and self._key_provider:
            final_embedder_config["key_provider"] = self._key_provider
        if collection_name and "collection_name" not in final_vector_store_config:
            final_vector_store_config["collection_name"] = collection_name

        if not final_embedder_id: raise ValueError("RAG embedder ID not resolved for index_directory.")
        if not final_vector_store_id: raise ValueError("RAG vector store ID not resolved for index_directory.")

        return await self._rag_manager.index_data_source(
            loader_id=final_loader_id, loader_source_uri=path,
            splitter_id=final_splitter_id, embedder_id=final_embedder_id,
            vector_store_id=final_vector_store_id,
            loader_config=final_loader_config, splitter_config=final_splitter_config,
            embedder_config=final_embedder_config, vector_store_config=final_vector_store_config,
        )

    async def index_web_page(
        self, url: str, collection_name: Optional[str] = None,
        loader_id: Optional[str] = None, splitter_id: Optional[str] = None,
        embedder_id: Optional[str] = None, vector_store_id: Optional[str] = None,
        loader_config: Optional[Dict[str, Any]] = None,
        splitter_config: Optional[Dict[str, Any]] = None,
        embedder_config: Optional[Dict[str, Any]] = None,
        vector_store_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        final_loader_id = loader_id or self._config.default_rag_loader_id or "web_page_loader_v1"
        final_splitter_id = splitter_id or self._config.default_rag_splitter_id or "character_recursive_text_splitter_v1"
        final_embedder_id = embedder_id or self._config.default_rag_embedder_id
        final_vector_store_id = vector_store_id or self._config.default_rag_vector_store_id

        def get_base_config(plugin_id: Optional[str], config_map_name: str) -> Dict[str, Any]:
            if plugin_id and hasattr(self._config, config_map_name):
                return getattr(self._config, config_map_name).get(plugin_id, {})
            return {}

        final_loader_config = {**get_base_config(final_loader_id, "document_loader_configurations"), **(loader_config or {}), **kwargs.get("loader_config_override", {})}
        final_splitter_config = {**get_base_config(final_splitter_id, "text_splitter_configurations"), **(splitter_config or {}), **kwargs.get("splitter_config_override", {})}
        final_embedder_config = {**get_base_config(final_embedder_id, "embedding_generator_configurations"), **(embedder_config or {}), **kwargs.get("embedder_config_override", {})}
        final_vector_store_config = {**get_base_config(final_vector_store_id, "vector_store_configurations"), **(vector_store_config or {}), **kwargs.get("vector_store_config_override", {})}

        if "key_provider" not in final_embedder_config and self._key_provider:
            final_embedder_config["key_provider"] = self._key_provider
        if collection_name and "collection_name" not in final_vector_store_config:
            final_vector_store_config["collection_name"] = collection_name

        if not final_embedder_id: raise ValueError("RAG embedder ID not resolved for index_web_page.")
        if not final_vector_store_id: raise ValueError("RAG vector store ID not resolved for index_web_page.")

        return await self._rag_manager.index_data_source(
            loader_id=final_loader_id, loader_source_uri=url,
            splitter_id=final_splitter_id, embedder_id=final_embedder_id,
            vector_store_id=final_vector_store_id,
            loader_config=final_loader_config, splitter_config=final_splitter_config,
            embedder_config=final_embedder_config, vector_store_config=final_vector_store_config,
        )

    async def search(
        self, query: str, collection_name: Optional[str] = None,
        top_k: int = 5, retriever_id: Optional[str] = None,
        retriever_config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List[RetrievedChunk]:
        final_retriever_id = retriever_id or self._config.default_rag_retriever_id or "basic_similarity_retriever_v1"

        base_retriever_cfg = {}
        if hasattr(self._config, "retriever_configurations"):
             base_retriever_cfg = self._config.retriever_configurations.get(final_retriever_id, {})

        final_retriever_config = {**base_retriever_cfg, **(retriever_config or {}), **kwargs}

        if "embedder_config" not in final_retriever_config: final_retriever_config["embedder_config"] = {}
        if "key_provider" not in final_retriever_config["embedder_config"] and self._key_provider:
            final_retriever_config["embedder_config"]["key_provider"] = self._key_provider

        if "vector_store_config" not in final_retriever_config: final_retriever_config["vector_store_config"] = {}
        if collection_name and "collection_name" not in final_retriever_config["vector_store_config"]:
            final_retriever_config["vector_store_config"]["collection_name"] = collection_name

        return await self._rag_manager.retrieve_from_query(
            query_text=query, retriever_id=final_retriever_id,
            retriever_config=final_retriever_config, top_k=top_k
        )

class FunctionToolWrapper(ToolPlugin):
    """
    A wrapper that makes a Python function conform to the ToolPlugin protocol,
    using metadata generated by the @tool decorator.
    """
    _func: Callable
    _metadata: Dict[str, Any]
    _is_async: bool

    @property
    def plugin_id(self) -> str: # ToolPlugin requires plugin_id
        return self._metadata.get("identifier", self._func.__name__)

    @property
    def identifier(self) -> str:
        return self._metadata.get("identifier", self._func.__name__)

    def __init__(self, func: Callable, metadata: Dict[str, Any]):
        if not callable(func):
            raise TypeError("Wrapped object must be callable.")
        self._func = func
        self._metadata = metadata
        self._is_async = inspect.iscoroutinefunction(func)

        # Ensure metadata has an identifier and name, defaulting to function name
        if "identifier" not in self._metadata or not self._metadata["identifier"]:
            self._metadata["identifier"] = self._func.__name__
        if "name" not in self._metadata or not self._metadata["name"]:
            self._metadata["name"] = self._func.__name__.replace("_", " ").title()

    async def get_metadata(self) -> Dict[str, Any]:
        return self._metadata

    async def execute(
        self,
        params: Dict[str, Any],
        key_provider: KeyProvider,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        # KeyProvider and context are not automatically passed to the raw function
        # unless the function is specifically designed to accept them.
        # This wrapper calls the function with only its defined parameters.
        if self._is_async:
            return await self._func(**params)
        else:
            loop = asyncio.get_running_loop()
            # functools.partial helps pass kwargs correctly to the executor
            return await loop.run_in_executor(None, functools.partial(self._func, **params))

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass # Decorated functions typically don't have a separate setup

    async def teardown(self) -> None:
        pass # Decorated functions typically don't have a separate teardown

class Genie:
    def __init__(
        self, plugin_manager: PluginManager, key_provider: KeyProvider,
        config: MiddlewareConfig, tool_manager: ToolManager,
        tool_invoker: ToolInvoker, rag_manager: RAGManager,
        tool_lookup_service: ToolLookupService, llm_provider_manager: LLMProviderManager,
        command_processor_manager: CommandProcessorManager,
        llm_interface: LLMInterface, rag_interface: RAGInterface,
    ):
        self._plugin_manager = plugin_manager
        self._key_provider = key_provider
        self._config = config
        self._tool_manager = tool_manager
        self._tool_invoker = tool_invoker
        self._rag_manager = rag_manager
        self._tool_lookup_service = tool_lookup_service
        self._llm_provider_manager = llm_provider_manager
        self._command_processor_manager = command_processor_manager
        self.llm = llm_interface
        self.rag = rag_interface
        self._config._genie_instance = self # type: ignore
        logger.info("Genie facade initialized with resolved configuration.")

    @classmethod
    async def create(
        cls,
        config: MiddlewareConfig,
        key_provider_instance: Optional[KeyProvider] = None,
    ) -> Genie:
        pm_for_kp_loading = PluginManager(plugin_dev_dirs=config.plugin_dev_dirs)
        await pm_for_kp_loading.discover_plugins()

        actual_key_provider: KeyProvider
        user_kp_id_preference = config.key_provider_id

        if key_provider_instance:
            actual_key_provider = key_provider_instance
            if isinstance(key_provider_instance, CorePluginType):
                kp_instance_plugin_id = getattr(key_provider_instance, "plugin_id", None)
                if kp_instance_plugin_id and kp_instance_plugin_id not in pm_for_kp_loading._plugin_instances: # type: ignore
                     pm_for_kp_loading._plugin_instances[kp_instance_plugin_id] = key_provider_instance # type: ignore
        else:
            kp_id_to_load_alias_or_canonical = user_kp_id_preference or PLUGIN_ID_ALIASES["env_keys"]
            kp_id_canonical_to_load = PLUGIN_ID_ALIASES.get(kp_id_to_load_alias_or_canonical, kp_id_to_load_alias_or_canonical)
            kp_any = await pm_for_kp_loading.get_plugin_instance(kp_id_canonical_to_load)
            if not kp_any or not isinstance(kp_any, KeyProvider):
                raise RuntimeError(f"Failed to load KeyProvider with ID '{kp_id_canonical_to_load}' (resolved from '{user_kp_id_preference}').")
            actual_key_provider = cast(KeyProvider, kp_any)

        logger.info(f"Using KeyProvider: {type(actual_key_provider).__name__} (ID: {getattr(actual_key_provider, 'plugin_id', 'N/A')})")

        resolver = ConfigResolver()
        resolved_config: MiddlewareConfig = resolver.resolve(config, key_provider_instance=actual_key_provider)

        pm = PluginManager(plugin_dev_dirs=resolved_config.plugin_dev_dirs)
        await pm.discover_plugins()

        if isinstance(actual_key_provider, CorePluginType):
            kp_main_pm_plugin_id = getattr(actual_key_provider, "plugin_id", None)
            if kp_main_pm_plugin_id and (kp_main_pm_plugin_id not in pm._plugin_instances or pm._plugin_instances[kp_main_pm_plugin_id] is not actual_key_provider): # type: ignore
                 pm._plugin_instances[kp_main_pm_plugin_id] = actual_key_provider # type: ignore
                 if kp_main_pm_plugin_id not in pm.list_discovered_plugin_classes():
                     pm._discovered_plugin_classes[kp_main_pm_plugin_id] = type(actual_key_provider) # type: ignore

        tool_manager = ToolManager(plugin_manager=pm)
        await tool_manager.initialize_tools(tool_configurations=resolved_config.tool_configurations)

        tool_invoker = ToolInvoker(tool_manager=tool_manager, plugin_manager=pm)
        rag_manager = RAGManager(plugin_manager=pm)

        tool_lookup_service = ToolLookupService(
            tool_manager=tool_manager, plugin_manager=pm,
            default_provider_id=resolved_config.default_tool_lookup_provider_id, # type: ignore
            default_indexing_formatter_id=resolved_config.default_tool_indexing_formatter_id # type: ignore
        )

        llm_provider_manager = LLMProviderManager(pm, actual_key_provider, resolved_config)
        command_processor_manager = CommandProcessorManager(pm, actual_key_provider, resolved_config)

        llm_interface = LLMInterface(llm_provider_manager, resolved_config.default_llm_provider_id)
        rag_interface = RAGInterface(rag_manager, resolved_config, actual_key_provider)

        return cls(
            plugin_manager=pm, key_provider=actual_key_provider, config=resolved_config,
            tool_manager=tool_manager, tool_invoker=tool_invoker, rag_manager=rag_manager,
            tool_lookup_service=tool_lookup_service, llm_provider_manager=llm_provider_manager,
            command_processor_manager=command_processor_manager,
            llm_interface=llm_interface, rag_interface=rag_interface
        )

    async def register_tool_functions(self, functions: List[Callable]) -> None:
        """
        Registers functions decorated with @tool as available tools.
        """
        if not self._tool_manager:
            logger.error("Genie: ToolManager not initialized. Cannot register function-based tools.")
            return

        registered_count = 0
        for func_item in functions:
            # The @tool decorator should attach _tool_metadata_ to the (potentially wrapped) callable
            # and _original_function_ if it created its own wrapper.
            metadata = getattr(func_item, "_tool_metadata_", None)
            original_func_to_call = getattr(func_item, "_original_function_", func_item)

            if metadata and isinstance(metadata, dict) and callable(original_func_to_call):
                tool_wrapper = FunctionToolWrapper(original_func_to_call, metadata)

                if tool_wrapper.identifier in self._tool_manager._tools:
                    logger.warning(f"Genie: Tool with identifier '{tool_wrapper.identifier}' (from function {original_func_to_call.__name__}) already registered. Overwriting.")

                self._tool_manager._tools[tool_wrapper.identifier] = tool_wrapper
                # No specific config for function tools by default, but could be added if needed
                # self._tool_manager._tool_initial_configs[tool_wrapper.plugin_id] = {}

                logger.info(f"Genie: Registered function '{original_func_to_call.__name__}' as tool '{tool_wrapper.identifier}'.")
                registered_count += 1
            else:
                func_name_attr = getattr(func_item, "__name__", str(func_item))
                logger.warning(f"Genie: Function '{func_name_attr}' is not decorated with @tool or metadata is missing/invalid. Skipping registration.")

        if registered_count > 0:
            logger.info(f"Genie: Successfully registered {registered_count} function-based tools.")
            if hasattr(self, "_tool_lookup_service") and self._tool_lookup_service:
                self._tool_lookup_service.invalidate_index()
                logger.info("Genie: Invalidated tool lookup service index due to new tool registrations.")


    async def execute_tool(self, tool_identifier: str, **params: Any) -> Any:
        if not self._tool_invoker: raise RuntimeError("ToolInvoker not initialized.")
        if not self._key_provider: raise RuntimeError("KeyProvider not initialized.")
        return await self._tool_invoker.invoke(
            tool_identifier=tool_identifier, params=params, key_provider=self._key_provider
        )

    async def run_command(
        self, command: str, processor_id: Optional[str] = None,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> Any:
        if not self._command_processor_manager: return {"error": "CommandProcessorManager not initialized."}
        target_processor_id = processor_id or self._config.default_command_processor_id
        if not target_processor_id: return {"error": "No command processor configured."}
        try:
            processor_plugin = await self._command_processor_manager.get_command_processor(target_processor_id, genie_facade=self)
            if not processor_plugin: return {"error": f"CommandProcessor '{target_processor_id}' not found."}
            cmd_proc_response: CommandProcessorResponse = await processor_plugin.process_command(command, conversation_history)
            error_val, thought_val = cmd_proc_response.get("error"), cmd_proc_response.get("llm_thought_process")
            chosen_tool_id, extracted_params = cmd_proc_response.get("chosen_tool_id"), cmd_proc_response.get("extracted_params")
            if error_val: return {"error": error_val, "thought_process": thought_val}
            if chosen_tool_id and extracted_params is not None:
                tool_result = await self.execute_tool(chosen_tool_id, **extracted_params)
                return {"tool_result": tool_result, "thought_process": thought_val}
            return {"message": "No tool selected by command processor.", "thought_process": thought_val}
        except Exception as e: return {"error": f"Unexpected error in run_command: {str(e)}", "raw_exception": e}


    async def close(self) -> None:
        logger.info("Genie: Initiating teardown...")
        if self._plugin_manager: await self._plugin_manager.teardown_all_plugins()
        managers_to_teardown = [self._llm_provider_manager, self._command_processor_manager]
        for m in managers_to_teardown:
            if m and hasattr(m, "teardown") and callable(m.teardown):
                try: await m.teardown()
                except Exception as e_td: logger.error(f"Error tearing down manager {type(m).__name__}: {e_td}", exc_info=True)

        attrs_to_null = ["_plugin_manager", "_key_provider", "_config", "_tool_manager", "_tool_invoker",
                         "_rag_manager", "_tool_lookup_service", "_llm_provider_manager",
                         "_command_processor_manager", "llm", "rag"]
        for attr in attrs_to_null:
            if hasattr(self, attr):
                setattr(self, attr, None)
        logger.info("Genie: Teardown complete.")
