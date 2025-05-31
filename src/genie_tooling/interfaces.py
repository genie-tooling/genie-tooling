### src/genie_tooling/interfaces.py
# src/genie_tooling/interfaces.py
"""
Defines the public interfaces provided by the Genie facade for interacting
with different aspects of the middleware.
"""
import asyncio
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

# Forward references to avoid circular imports at runtime,
# but allow type checking.
if TYPE_CHECKING:
    from .llm_providers.manager import LLMProviderManager
    from .llm_providers.types import (
        ChatMessage, LLMChatChunk, LLMChatResponse,
        LLMCompletionChunk, LLMCompletionResponse, LLMUsageInfo
    )
    from .rag.manager import RAGManager
    from .core.types import RetrievedChunk
    from .observability.manager import InteractionTracingManager
    from .hitl.manager import HITLManager
    from .hitl.types import ApprovalRequest, ApprovalResponse
    from .token_usage.manager import TokenUsageManager
    from .token_usage.types import TokenUsageRecord
    from .guardrails.manager import GuardrailManager
    from .prompts.manager import PromptManager
    from .prompts.types import FormattedPrompt, PromptData, PromptIdentifier
    from .prompts.conversation.manager import ConversationStateManager # CORRECTED: Path to manager
    from .prompts.conversation.types import ConversationState # CORRECTED: Path to type
    from .prompts.llm_output_parsers.manager import LLMOutputParserManager # CORRECTED: Path to manager
    from .prompts.llm_output_parsers.types import ParsedOutput # CORRECTED: Path to type
    from .security.key_provider import KeyProvider
    from .config.models import MiddlewareConfig


logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(
        self,
        llm_provider_manager: "LLMProviderManager",
        default_provider_id: Optional[str],
        output_parser_manager: "LLMOutputParserManager",
        tracing_manager: Optional["InteractionTracingManager"] = None,
        guardrail_manager: Optional["GuardrailManager"] = None,
        token_usage_manager: Optional["TokenUsageManager"] = None
    ):
        self._llm_provider_manager = llm_provider_manager
        self._default_provider_id = default_provider_id
        self._output_parser_manager = output_parser_manager
        self._tracing_manager = tracing_manager
        self._guardrail_manager = guardrail_manager
        self._token_usage_manager = token_usage_manager
        logger.debug(f"LLMInterface initialized with default provider ID: {self._default_provider_id}")

    async def _trace(self, event_name: str, data: Dict, corr_id: Optional[str]):
        if self._tracing_manager:
            await self._tracing_manager.trace_event(event_name, data, "LLMInterface", corr_id)

    async def _record_token_usage(self, provider_id: str, model_name: str, usage_info: Optional["LLMUsageInfo"], call_type: str):
        if self._token_usage_manager and usage_info:
            # Need to import TokenUsageRecord here or ensure it's available
            from .token_usage.types import TokenUsageRecord # Local import for type
            record = TokenUsageRecord(
                provider_id=provider_id, model_name=model_name,
                prompt_tokens=usage_info.get("prompt_tokens"),
                completion_tokens=usage_info.get("completion_tokens"),
                total_tokens=usage_info.get("total_tokens"),
                timestamp=asyncio.get_event_loop().time(), call_type=call_type
            )
            await self._token_usage_manager.record_usage(record)

    async def generate(
        self, prompt: str, provider_id: Optional[str] = None, stream: bool = False, **kwargs: Any
    ) -> Union["LLMCompletionResponse", AsyncIterable["LLMCompletionChunk"]]:
        corr_id = str(uuid.uuid4())
        provider_to_use = provider_id or self._default_provider_id
        if not provider_to_use:
            raise ValueError("No LLM provider ID specified and no default is set for generate.")
        await self._trace("llm.generate.start", {"provider_id": provider_to_use, "prompt_len": len(prompt), "stream": stream, "kwargs": kwargs}, corr_id)
        if self._guardrail_manager:
            input_violation = await self._guardrail_manager.check_input_guardrails(prompt, {"type": "llm_generate_prompt", "provider_id": provider_to_use})
            if input_violation["action"] == "block":
                await self._trace("llm.generate.blocked_by_input_guardrail", {"violation": input_violation}, corr_id)
                raise PermissionError(f"LLM generate blocked by input guardrail: {input_violation.get('reason')}")
        provider = await self._llm_provider_manager.get_llm_provider(provider_to_use)
        if not provider:
            await self._trace("llm.generate.error", {"error": "ProviderNotFound", "provider_id": provider_to_use}, corr_id)
            raise RuntimeError(f"LLM Provider '{provider_to_use}' not found or failed to load.")
        model_name_used = kwargs.get("model", getattr(provider, "_model_name", "unknown"))
        try:
            result_or_stream = await provider.generate(prompt, stream=stream, **kwargs)
            if stream:
                async def wrapped_stream():
                    full_response_text = ""
                    # Need LLMCompletionChunk for type hint
                    from .llm_providers.types import LLMCompletionChunk
                    async for chunk in cast(AsyncIterable[LLMCompletionChunk], result_or_stream):
                        if self._guardrail_manager and chunk.get("text_delta"):
                            output_violation = await self._guardrail_manager.check_output_guardrails(chunk["text_delta"], {"type": "llm_generate_chunk", "provider_id": provider_to_use})
                            if output_violation["action"] == "block":
                                logger.warning(f"LLM generate stream chunk blocked by output guardrail: {output_violation.get('reason')}")
                                yield {"text_delta": f"[STREAM BLOCKED: {output_violation.get('reason')}]", "finish_reason": "blocked_by_guardrail", "raw_chunk": {}} # type: ignore
                                break 
                        full_response_text += chunk.get("text_delta", "")
                        yield chunk
                        if chunk.get("finish_reason") and chunk.get("usage_delta"):
                            await self._record_token_usage(provider_to_use, model_name_used, chunk["usage_delta"], "generate_stream_end")
                    await self._trace("llm.generate.stream_end", {"response_len": len(full_response_text)}, corr_id)
                return wrapped_stream()
            else:
                # Need LLMCompletionResponse for type hint
                from .llm_providers.types import LLMCompletionResponse
                result = cast(LLMCompletionResponse, result_or_stream)
                if self._guardrail_manager:
                    output_violation = await self._guardrail_manager.check_output_guardrails(result["text"], {"type": "llm_generate_response", "provider_id": provider_to_use})
                    if output_violation["action"] == "block":
                        await self._trace("llm.generate.blocked_by_output_guardrail", {"violation": output_violation, "original_text": result["text"]}, corr_id)
                        result["text"] = f"[RESPONSE BLOCKED: {output_violation.get('reason')}]"
                        result["finish_reason"] = "blocked_by_guardrail" # type: ignore
                await self._record_token_usage(provider_to_use, model_name_used, result.get("usage"), "generate")
                await self._trace("llm.generate.success", {"response_len": len(result["text"]), "finish_reason": result.get("finish_reason")}, corr_id)
                return result
        except Exception as e:
            await self._trace("llm.generate.error", {"error": str(e), "type": type(e).__name__}, corr_id)
            raise

    async def chat(
        self, messages: List["ChatMessage"], provider_id: Optional[str] = None, stream: bool = False, **kwargs: Any
    ) -> Union["LLMChatResponse", AsyncIterable["LLMChatChunk"]]:
        corr_id = str(uuid.uuid4())
        provider_to_use = provider_id or self._default_provider_id
        if not provider_to_use:
            raise ValueError("No LLM provider ID specified and no default is set for chat.")
        await self._trace("llm.chat.start", {"provider_id": provider_to_use, "num_messages": len(messages), "stream": stream, "kwargs": kwargs}, corr_id)
        if self._guardrail_manager:
            input_data_for_guardrail = messages[-1] if messages else "" 
            input_violation = await self._guardrail_manager.check_input_guardrails(input_data_for_guardrail, {"type": "llm_chat_messages", "provider_id": provider_to_use})
            if input_violation["action"] == "block":
                await self._trace("llm.chat.blocked_by_input_guardrail", {"violation": input_violation}, corr_id)
                raise PermissionError(f"LLM chat blocked by input guardrail: {input_violation.get('reason')}")
        provider = await self._llm_provider_manager.get_llm_provider(provider_to_use)
        if not provider:
            await self._trace("llm.chat.error", {"error": "ProviderNotFound", "provider_id": provider_to_use}, corr_id)
            raise RuntimeError(f"LLM Provider '{provider_to_use}' not found or failed to load.")
        model_name_used = kwargs.get("model", getattr(provider, "_model_name", "unknown"))
        try:
            result_or_stream = await provider.chat(messages, stream=stream, **kwargs)
            if stream:
                async def wrapped_stream():
                    full_response_content = ""
                    # Need LLMChatChunk for type hint
                    from .llm_providers.types import LLMChatChunk
                    async for chunk in cast(AsyncIterable[LLMChatChunk], result_or_stream):
                        delta_content = chunk.get("message_delta", {}).get("content", "")
                        if self._guardrail_manager and delta_content:
                            output_violation = await self._guardrail_manager.check_output_guardrails(delta_content, {"type": "llm_chat_chunk", "provider_id": provider_to_use})
                            if output_violation["action"] == "block":
                                logger.warning(f"LLM chat stream chunk blocked by output guardrail: {output_violation.get('reason')}")
                                yield {"message_delta": {"role": "assistant", "content": f"[STREAM BLOCKED: {output_violation.get('reason')}]"}, "finish_reason": "blocked_by_guardrail", "raw_chunk": {}} # type: ignore
                                break
                        full_response_content += delta_content or ""
                        yield chunk
                        if chunk.get("finish_reason") and chunk.get("usage_delta"):
                            await self._record_token_usage(provider_to_use, model_name_used, chunk["usage_delta"], "chat_stream_end")
                    await self._trace("llm.chat.stream_end", {"response_len": len(full_response_content)}, corr_id)
                return wrapped_stream()
            else:
                # Need LLMChatResponse for type hint
                from .llm_providers.types import LLMChatResponse
                result = cast(LLMChatResponse, result_or_stream)
                if self._guardrail_manager and result["message"].get("content"):
                    output_violation = await self._guardrail_manager.check_output_guardrails(result["message"]["content"], {"type": "llm_chat_response", "provider_id": provider_to_use})
                    if output_violation["action"] == "block":
                        await self._trace("llm.chat.blocked_by_output_guardrail", {"violation": output_violation, "original_content": result["message"]["content"]}, corr_id)
                        result["message"]["content"] = f"[RESPONSE BLOCKED: {output_violation.get('reason')}]"
                        result["finish_reason"] = "blocked_by_guardrail" # type: ignore
                await self._record_token_usage(provider_to_use, model_name_used, result.get("usage"), "chat")
                await self._trace("llm.chat.success", {"response_content_len": len(result['message'].get('content') or ""), "finish_reason": result.get("finish_reason")}, corr_id)
                return result
        except Exception as e:
            await self._trace("llm.chat.error", {"error": str(e), "type": type(e).__name__}, corr_id)
            raise

    async def parse_output(
        self, 
        response: Union["LLMChatResponse", "LLMCompletionResponse"], 
        parser_id: Optional[str] = None, 
        schema: Optional[Any] = None
    ) -> "ParsedOutput":
        corr_id = str(uuid.uuid4())
        await self._trace("llm.parse_output.start", {"parser_id": parser_id, "has_schema": schema is not None}, corr_id)
        
        text_to_parse: Optional[str] = None
        if "text" in response: # LLMCompletionResponse
            text_to_parse = response["text"] # type: ignore
        elif "message" in response and isinstance(response["message"], dict) and "content" in response["message"]: # LLMChatResponse
            text_to_parse = response["message"]["content"]
        
        if text_to_parse is None:
            await self._trace("llm.parse_output.error", {"error": "No text content found in LLM response"}, corr_id)
            raise ValueError("No text content found in LLM response to parse.")

        try:
            parsed_data = await self._output_parser_manager.parse(text_to_parse, parser_id, schema)
            await self._trace("llm.parse_output.success", {"parsed_type": type(parsed_data).__name__}, corr_id)
            return parsed_data
        except Exception as e:
            await self._trace("llm.parse_output.error", {"error": str(e), "type": type(e).__name__}, corr_id)
            raise

class RAGInterface:
    def __init__(self, rag_manager: "RAGManager", config: "MiddlewareConfig", key_provider: "KeyProvider", tracing_manager: Optional["InteractionTracingManager"] = None):
        self._rag_manager = rag_manager
        self._config = config
        self._key_provider = key_provider
        self._tracing_manager = tracing_manager
        logger.debug("RAGInterface initialized.")

    async def _trace(self, event_name: str, data: Dict, corr_id: Optional[str]):
        if self._tracing_manager:
            await self._tracing_manager.trace_event(event_name, data, "RAGInterface", corr_id)

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
        corr_id = str(uuid.uuid4())
        await self._trace("rag.index_directory.start", {"path": path, "collection_name": collection_name}, corr_id)
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
        result = await self._rag_manager.index_data_source(
            loader_id=final_loader_id, loader_source_uri=path,
            splitter_id=final_splitter_id, embedder_id=final_embedder_id,
            vector_store_id=final_vector_store_id,
            loader_config=final_loader_config, splitter_config=final_splitter_config,
            embedder_config=final_embedder_config, vector_store_config=final_vector_store_config,
        )
        await self._trace("rag.index_directory.end", {"result_status": result.get("status"), "added_count": result.get("added_count")}, corr_id)
        return result

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
        corr_id = str(uuid.uuid4())
        await self._trace("rag.index_web_page.start", {"url": url, "collection_name": collection_name}, corr_id)
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
        result = await self._rag_manager.index_data_source(
            loader_id=final_loader_id, loader_source_uri=url,
            splitter_id=final_splitter_id, embedder_id=final_embedder_id,
            vector_store_id=final_vector_store_id,
            loader_config=final_loader_config, splitter_config=final_splitter_config,
            embedder_config=final_embedder_config, vector_store_config=final_vector_store_config,
        )
        await self._trace("rag.index_web_page.end", {"result_status": result.get("status"), "added_count": result.get("added_count")}, corr_id)
        return result

    async def search(
        self, query: str, collection_name: Optional[str] = None,
        top_k: int = 5, retriever_id: Optional[str] = None,
        retriever_config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> List["RetrievedChunk"]:
        corr_id = str(uuid.uuid4())
        await self._trace("rag.search.start", {"query_len": len(query), "collection_name": collection_name, "top_k": top_k}, corr_id)
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
        results = await self._rag_manager.retrieve_from_query(
            query_text=query, retriever_id=final_retriever_id,
            retriever_config=final_retriever_config, top_k=top_k
        )
        await self._trace("rag.search.end", {"num_results": len(results)}, corr_id)
        return results

class ObservabilityInterface:
    def __init__(self, tracing_manager: "InteractionTracingManager"): self._tracing_manager = tracing_manager
    async def trace_event(self, event_name: str, data: Dict[str, Any], component: Optional[str] = None, correlation_id: Optional[str] = None) -> None:
        if self._tracing_manager: await self._tracing_manager.trace_event(event_name, data, component, correlation_id)

class HITLInterface:
    def __init__(self, hitl_manager: "HITLManager"): self._hitl_manager = hitl_manager
    async def request_approval(self, request: "ApprovalRequest", approver_id: Optional[str] = None) -> "ApprovalResponse":
        if self._hitl_manager: return await self._hitl_manager.request_approval(request, approver_id)
        logger.error("HITLManager not available in HITLInterface."); 
        # Need ApprovalResponse for type hint
        from .hitl.types import ApprovalResponse
        return ApprovalResponse(request_id=request.get("request_id", str(uuid.uuid4())), status="error", reason="HITL system unavailable.")

class UsageTrackingInterface:
    def __init__(self, token_usage_manager: "TokenUsageManager"): self._token_usage_manager = token_usage_manager
    async def record_usage(self, record: "TokenUsageRecord") -> None:
        if self._token_usage_manager: await self._token_usage_manager.record_usage(record)
    async def get_summary(self, recorder_id: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._token_usage_manager: return await self._token_usage_manager.get_summary(recorder_id, filter_criteria)
        return {"error": "TokenUsageManager unavailable."}

class PromptInterface:
    def __init__(self, prompt_manager: "PromptManager"): self._prompt_manager = prompt_manager
    async def get_prompt_template_content(self, name: str, version: Optional[str] = None, registry_id: Optional[str] = None) -> Optional[str]:
        return await self._prompt_manager.get_raw_template(name, version, registry_id)
    async def render_prompt(self, name: str, data: "PromptData", version: Optional[str] = None, registry_id: Optional[str] = None, template_engine_id: Optional[str] = None) -> Optional["FormattedPrompt"]:
        return await self._prompt_manager.render_prompt(name, data, version, registry_id, template_engine_id)
    async def render_chat_prompt(self, name: str, data: "PromptData", version: Optional[str] = None, registry_id: Optional[str] = None, template_engine_id: Optional[str] = None) -> Optional[List["ChatMessage"]]:
        return await self._prompt_manager.render_chat_prompt(name, data, version, registry_id, template_engine_id)
    async def list_templates(self, registry_id: Optional[str] = None) -> List["PromptIdentifier"]: # type: ignore
        return await self._prompt_manager.list_available_templates(registry_id)

class ConversationInterface:
    def __init__(self, conversation_manager: Optional["ConversationStateManager"]): # Allow Optional manager
        self._conversation_manager = conversation_manager
        if not self._conversation_manager:
            logger.warning("ConversationInterface initialized without a ConversationStateManager. Operations will be no-ops or return defaults.")
    async def load_state(self, session_id: str, provider_id: Optional[str] = None) -> Optional["ConversationState"]:
        if not self._conversation_manager:
            logger.error("ConversationStateManager not available in ConversationInterface for load_state.")
            return None
        return await self._conversation_manager.load_state(session_id, provider_id)
    async def save_state(self, state: "ConversationState", provider_id: Optional[str] = None) -> None:
        if not self._conversation_manager:
            logger.error("ConversationStateManager not available in ConversationInterface for save_state.")
            return
        await self._conversation_manager.save_state(state, provider_id)
    async def add_message(self, session_id: str, message: "ChatMessage", provider_id: Optional[str] = None) -> None:
        if not self._conversation_manager:
            logger.error("ConversationStateManager not available in ConversationInterface for add_message.")
            return
        await self._conversation_manager.add_message(session_id, message, provider_id)
    async def delete_state(self, session_id: str, provider_id: Optional[str] = None) -> bool:
        if not self._conversation_manager:
            logger.error("ConversationStateManager not available in ConversationInterface for delete_state.")
            return False
        return await self._conversation_manager.delete_state(session_id, provider_id)