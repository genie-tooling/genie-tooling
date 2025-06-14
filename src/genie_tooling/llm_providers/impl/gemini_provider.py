### src/genie_tooling/llm_providers/impl/gemini_provider.py
import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, Union

from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatChunk,
    LLMChatChunkDeltaMessage,
    LLMChatResponse,
    LLMCompletionChunk,
    LLMCompletionResponse,
    LLMUsageInfo,
    ToolCall,
    ToolCallFunction,
)
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.token_usage.manager import TokenUsageManager

logger = logging.getLogger(__name__)

try:
    # --- Adhering to original user request to use the 'google.genai' library structure ---
    from google import genai
    from google.auth.exceptions import DefaultCredentialsError
    from google.genai import types as genai_types

    # --- MODIFICATION START: Import BaseModel for type checking ---
    from pydantic import BaseModel
    # --- MODIFICATION END ---
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    genai = None # type: ignore
    genai_types = None # type: ignore
    DefaultCredentialsError = None # type: ignore
    BaseModel = None # type: ignore
    GEMINI_SDK_AVAILABLE = False
    logger.warning(
        "GeminiLLMProviderPlugin: 'google-genai' or 'pydantic' library not installed. "
        "This plugin will not be functional. Please install it."
    )


class GeminiLLMProviderPlugin(LLMProviderPlugin):
    plugin_id: str = "gemini_llm_provider_v1"
    description: str = "LLM provider for Google Gemini models using the google-genai SDK."

    _client: Optional[genai.Client] = None
    _model_name: str
    _api_key_name: str = "GOOGLE_API_KEY"
    _key_provider: Optional[KeyProvider] = None
    _token_usage_manager: Optional[TokenUsageManager] = None

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        await super().setup(config)
        if not GEMINI_SDK_AVAILABLE or not genai:
            logger.error(f"{self.plugin_id}: 'google-genai' library is not available. Cannot proceed.")
            return

        cfg = config or {}
        self._key_provider = cfg.get("key_provider")
        if not self._key_provider or not isinstance(self._key_provider, KeyProvider):
            logger.error(f"{self.plugin_id}: KeyProvider not found in config or is invalid. Cannot fetch API key.")
            return

        self._api_key_name = cfg.get("api_key_name", self._api_key_name)
        self._model_name = cfg.get("model_name", "gemini-1.5-flash-latest")
        self._token_usage_manager = cfg.get("token_usage_manager")

        api_key = await self._key_provider.get_key(self._api_key_name)
        try:
            if api_key:
                self._client = genai.Client(api_key=api_key)
                logger.info(f"{self.plugin_id}: Initialized Gemini client with API key for model '{self._model_name}'.")
            else:
                logger.warning(
                    f"{self.plugin_id}: API key '{self._api_key_name}' not found. "
                    "Attempting to initialize Gemini client with Application Default Credentials (ADC)."
                )
                self._client = genai.Client()
                logger.info(f"{self.plugin_id}: Initialized Gemini client with ADC for model '{self._model_name}'.")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize Gemini client: {e}", exc_info=True)
            self._client = None

    async def _record_usage_if_manager(self, usage_metadata: Optional[genai_types.UsageMetadata], call_type: str):
        if self._token_usage_manager and usage_metadata:
            from genie_tooling.token_usage.types import TokenUsageRecord
            record = TokenUsageRecord(
                provider_id=self.plugin_id,
                model_name=self._model_name,
                prompt_tokens=usage_metadata.prompt_token_count,
                completion_tokens=usage_metadata.candidates_token_count,
                total_tokens=usage_metadata.total_token_count,
                timestamp=asyncio.get_event_loop().time(),
                call_type=call_type
            )
            await self._token_usage_manager.record_usage(record)

    def _serialize_gemini_parts(self, parts: List[genai_types.Part]) -> List[Dict[str, Any]]:
        serialized_parts = []
        for part in parts:
            part_dict = {}
            if part.text:
                part_dict["text"] = part.text
            if hasattr(part, "function_call") and part.function_call:
                part_dict["function_call"] = {
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args) if part.function_call.args else {}
                }
            if hasattr(part, "function_response") and part.function_response:
                part_dict["function_response"] = {
                    "name": part.function_response.name,
                    "response": dict(part.function_response.response) if part.function_response.response else {}
                }
            if part_dict:
                serialized_parts.append(part_dict)
        return serialized_parts

    def _minimal_serialize_gemini_response(self, response_or_chunk: genai_types.GenerateContentResponse) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        try:
            data["text_content_via_property"] = response_or_chunk.text
        except Exception:
            data["text_content_via_property"] = None
        if hasattr(response_or_chunk, "candidates") and response_or_chunk.candidates:
            data["finish_reason_from_candidate"] = response_or_chunk.candidates[0].finish_reason.name if response_or_chunk.candidates[0].finish_reason else None
            if response_or_chunk.candidates[0].content and response_or_chunk.candidates[0].content.parts:
                data["candidate_content_parts"] = self._serialize_gemini_parts(response_or_chunk.candidates[0].content.parts)
        if hasattr(response_or_chunk, "function_calls") and response_or_chunk.function_calls:
            data["function_calls_direct"] = [{"name": fc.name, "args": dict(fc.args) if fc.args else {}} for fc in response_or_chunk.function_calls]
        if hasattr(response_or_chunk, "usage_metadata") and response_or_chunk.usage_metadata:
            data["usage_metadata_for_raw"] = {"prompt_token_count": response_or_chunk.usage_metadata.prompt_token_count, "candidates_token_count": response_or_chunk.usage_metadata.candidates_token_count, "total_token_count": response_or_chunk.usage_metadata.total_token_count}
        return data

    def _convert_chat_messages_to_gemini(self, messages: List[ChatMessage]) -> Tuple[List[genai_types.Content], Optional[genai_types.Content]]:
        gemini_contents: List[genai_types.Content] = []
        system_instruction_content: Optional[genai_types.Content] = None

        for i, msg in enumerate(messages):
            role = msg["role"]
            content_parts: List[Union[str, genai_types.Part]] = []

            if role == "system":
                if msg.get("content"):
                    system_instruction_content = genai_types.Content(parts=[genai_types.Part.from_text(text=msg["content"])])
                continue

            if role == "tool":
                gemini_role = "function"
                tool_name = msg.get("name")
                if not tool_name:
                    logger.warning(f"{self.plugin_id}: Tool message missing 'name' (function name). Skipping tool response part.")
                    continue

                tool_response_content = msg.get("content")
                response_data_for_gemini: Dict[str, Any]
                if isinstance(tool_response_content, str):
                    try:
                        parsed_content = json.loads(tool_response_content)
                        if isinstance(parsed_content, dict):
                            response_data_for_gemini = parsed_content
                        else:
                            response_data_for_gemini = {"content": parsed_content}
                    except json.JSONDecodeError:
                        response_data_for_gemini = {"content": tool_response_content}
                elif isinstance(tool_response_content, dict):
                    response_data_for_gemini = tool_response_content
                elif tool_response_content is None:
                    response_data_for_gemini = {"content": "None"}
                else:
                    response_data_for_gemini = {"content": str(tool_response_content)}

                content_parts.append(genai_types.Part.from_function_response(name=tool_name, response=response_data_for_gemini))
                gemini_contents.append(genai_types.Content(role=gemini_role, parts=content_parts))
                continue

            gemini_role = "user" if role == "user" else "model"

            if msg.get("content"):
                content_parts.append(genai_types.Part.from_text(text=msg["content"]))

            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                    content_parts.append(genai_types.Part.from_function_call(name=tc["function"]["name"], args=args))

            if content_parts:
                 gemini_contents.append(genai_types.Content(role=gemini_role, parts=content_parts))

        return gemini_contents, system_instruction_content

    def _parse_gemini_tool_calls(self, function_calls: List[genai_types.FunctionCall]) -> List[ToolCall]:
        genie_tool_calls: List[ToolCall] = []
        for fc in function_calls:
            tool_call_id = f"gemini_tool_call_{uuid.uuid4().hex[:8]}"
            genie_tool_calls.append(ToolCall(
                id=tool_call_id,
                type="function",
                function=ToolCallFunction(
                    name=fc.name,
                    arguments=json.dumps(dict(fc.args)) if fc.args is not None else "{}"
                )
            ))
        return genie_tool_calls

    def _flatten_pydantic_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema, dict):
            return schema

        schema_copy = json.loads(json.dumps(schema))
        defs = schema_copy.pop("$defs", None)

        if not defs:
            return schema_copy

        def _resolve_refs_recursive(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node and isinstance(node["$ref"], str):
                    ref_path = node["$ref"]
                    parts = ref_path.split("/")
                    if len(parts) == 3 and parts[0] == "#" and parts[1] == "$defs":
                        def_name = parts[2]
                        if def_name in defs:
                            return _resolve_refs_recursive(defs[def_name])
                        else:
                            logger.warning(f"Reference '{ref_path}' not found in $defs. Leaving as is.")
                            return node
                    else:
                        return node
                else:
                    return {k: _resolve_refs_recursive(v) for k, v in node.items()}
            elif isinstance(node, list):
                return [_resolve_refs_recursive(item) for item in node]
            else:
                return node

        return _resolve_refs_recursive(schema_copy)

    def _remove_additional_properties(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively removes the 'additionalProperties' key from a schema dictionary,
        as it is not supported by the Gemini API.
        """
        if not isinstance(schema, dict):
            return schema

        if "additionalProperties" in schema:
            del schema["additionalProperties"]

        for key, value in schema.items():
            if isinstance(value, dict):
                schema[key] = self._remove_additional_properties(value)
            elif isinstance(value, list):
                schema[key] = [
                    self._remove_additional_properties(item) if isinstance(item, dict) else item
                    for item in value
                ]
        return schema

    async def generate(
        self, prompt: str, stream: bool = False, **kwargs: Any
    ) -> Union[LLMCompletionResponse, AsyncIterable[LLMCompletionChunk]]:
        if not self._client or not self._client.aio:
            raise RuntimeError(f"{self.plugin_id}: Client or async client (aio) not initialized.")

        model_name_to_use = f"models/{kwargs.pop('model', self._model_name)}"
        contents = [genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=prompt)])]

        gen_config_kwargs: Dict[str, Any] = {}
        if "temperature" in kwargs:
            gen_config_kwargs["temperature"] = kwargs.pop("temperature")
        if "top_p" in kwargs:
            gen_config_kwargs["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            gen_config_kwargs["top_k"] = kwargs.pop("top_k")
        if "max_tokens" in kwargs:
            gen_config_kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        if "stop_sequences" in kwargs:
            gen_config_kwargs["stop_sequences"] = kwargs.pop("stop_sequences")

        if "output_schema" in kwargs and kwargs["output_schema"] is not None:
            output_schema = kwargs.pop("output_schema")
            gen_config_kwargs["response_mime_type"] = "application/json"
            if BaseModel and isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
                json_schema = output_schema.model_json_schema()
                flattened_schema = self._flatten_pydantic_schema(json_schema)
                gen_config_kwargs["response_schema"] = self._remove_additional_properties(flattened_schema)
                logger.debug(f"{self.plugin_id}: Converted, flattened, and sanitized Pydantic model '{output_schema.__name__}' for Gemini.")
            else:
                gen_config_kwargs["response_schema"] = self._remove_additional_properties(output_schema)
            logger.debug(f"{self.plugin_id}: Configuring Gemini generate for JSON output with provided schema.")

        generation_config_obj = genai_types.GenerationConfig(**gen_config_kwargs) if gen_config_kwargs else None

        request_kwargs: Dict[str, Any] = {"model": model_name_to_use, "contents": contents}
        if generation_config_obj:
            request_kwargs["config"] = generation_config_obj.model_dump(by_alias=True, exclude_none=True)
        if kwargs:
            logger.debug(f"{self.plugin_id}: Unused kwargs for generate: {kwargs}")

        try:
            if stream:
                async def stream_generator() -> AsyncIterable[LLMCompletionChunk]:
                    response_stream = await self._client.aio.models.generate_content_stream(**request_kwargs)
                    final_usage_info = None

                    async for chunk in response_stream:
                        text_delta = chunk.text
                        current_chunk: LLMCompletionChunk = {"text_delta": text_delta, "raw_chunk": chunk.to_dict()}

                        if chunk.usage_metadata:
                            final_usage_info = {
                                "prompt_tokens": chunk.usage_metadata.prompt_token_count,
                                "completion_tokens": chunk.usage_metadata.candidates_token_count,
                                "total_tokens": chunk.usage_metadata.total_token_count,
                            }
                            current_chunk["usage_delta"] = final_usage_info

                        yield current_chunk

                    aggregated_response = await response_stream.aggregate_response()
                    finish_reason = "unknown"
                    if aggregated_response.candidates and aggregated_response.candidates[0].finish_reason:
                        finish_reason = aggregated_response.candidates[0].finish_reason.name.lower()

                    final_chunk: LLMCompletionChunk = {"finish_reason": finish_reason, "raw_chunk": {}}
                    if aggregated_response.usage_metadata:
                        usage_meta = aggregated_response.usage_metadata
                        final_usage_info = {
                            "prompt_tokens": usage_meta.prompt_token_count,
                            "completion_tokens": usage_meta.candidates_token_count,
                            "total_tokens": usage_meta.total_token_count,
                        }
                        final_chunk["usage_delta"] = final_usage_info
                        await self._record_usage_if_manager(usage_meta, "generate_stream_end")

                    if final_chunk.get("finish_reason") or final_chunk.get("usage_delta"):
                        yield final_chunk

                return stream_generator()
            else:
                response: genai_types.GenerateContentResponse = await self._client.aio.models.generate_content(**request_kwargs)
                text_content = response.text
                finish_reason = "unknown"
                if response.candidates:
                    finish_reason = genai_types.FinishReason(response.candidates[0].finish_reason).name.lower()

                usage_info: Optional[LLMUsageInfo] = None
                if response.usage_metadata:
                    usage_info = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.total_token_count,
                    }
                    await self._record_usage_if_manager(response.usage_metadata, "generate")

                return LLMCompletionResponse(
                    text=text_content,
                    finish_reason=finish_reason,
                    usage=usage_info,
                    raw_response=self._minimal_serialize_gemini_response(response)
                )
        except Exception as e:
            logger.error(f"{self.plugin_id}: Gemini API call failed: {e}", exc_info=True)
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    async def chat(
        self, messages: List[ChatMessage], stream: bool = False, **kwargs: Any
    ) -> Union[LLMChatResponse, AsyncIterable[LLMChatChunk]]:
        if not self._client or not self._client.aio:
            raise RuntimeError(f"{self.plugin_id}: Client or async client (aio) not initialized.")

        model_name_to_use = f"models/{kwargs.pop('model', self._model_name)}"
        gemini_contents, system_instruction = self._convert_chat_messages_to_gemini(messages)

        gen_config_kwargs: Dict[str, Any] = {}
        if "temperature" in kwargs:
            gen_config_kwargs["temperature"] = kwargs.pop("temperature")
        if "top_p" in kwargs:
            gen_config_kwargs["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            gen_config_kwargs["top_k"] = kwargs.pop("top_k")
        if "max_tokens" in kwargs:
            gen_config_kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        if "stop_sequences" in kwargs:
            gen_config_kwargs["stop_sequences"] = kwargs.pop("stop_sequences")

        if "output_schema" in kwargs and kwargs["output_schema"] is not None:
            output_schema = kwargs.pop("output_schema")
            gen_config_kwargs["response_mime_type"] = "application/json"
            if BaseModel and isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
                json_schema = output_schema.model_json_schema()
                flattened_schema = self._flatten_pydantic_schema(json_schema)
                gen_config_kwargs["response_schema"] = self._remove_additional_properties(flattened_schema)
                logger.debug(f"{self.plugin_id}: Converted, flattened, and sanitized Pydantic model '{output_schema.__name__}' for Gemini.")
            else:
                gen_config_kwargs["response_schema"] = self._remove_additional_properties(output_schema)
            logger.debug(f"{self.plugin_id}: Configuring Gemini chat for JSON output with provided schema.")

        generation_config_obj = genai_types.GenerationConfig(**gen_config_kwargs) if gen_config_kwargs else None

        request_kwargs: Dict[str, Any] = {"model": model_name_to_use, "contents": gemini_contents}
        if generation_config_obj:
            request_kwargs["config"] = generation_config_obj.model_dump(by_alias=True, exclude_none=True)
        if system_instruction:
            request_kwargs["system_instruction"] = system_instruction

        if "tools" in kwargs:
            request_kwargs["tools"] = kwargs.pop("tools")
        if "tool_choice" in kwargs:
             request_kwargs["tool_config"] = {"function_calling_config": {"mode": kwargs.pop("tool_choice")}}

        if kwargs:
            logger.debug(f"{self.plugin_id}: Unused kwargs for chat: {kwargs}")

        try:
            if stream:
                async def stream_chat_generator() -> AsyncIterable[LLMChatChunk]:
                    response_stream = await self._client.aio.models.generate_content_stream(**request_kwargs)

                    async for chunk in response_stream:
                        delta_msg_content = chunk.text
                        delta_tool_calls: Optional[List[ToolCall]] = None
                        if chunk.function_calls:
                             delta_tool_calls = self._parse_gemini_tool_calls(chunk.function_calls)

                        delta_message = LLMChatChunkDeltaMessage(role="assistant")
                        if delta_msg_content:
                            delta_message["content"] = delta_msg_content
                        if delta_tool_calls:
                            delta_message["tool_calls"] = delta_tool_calls

                        if delta_message:
                            yield LLMChatChunk(message_delta=delta_message, raw_chunk=chunk.to_dict())

                    aggregated_response = await response_stream.aggregate_response()
                    finish_reason = "unknown"
                    if aggregated_response.candidates and aggregated_response.candidates[0].finish_reason:
                        finish_reason = aggregated_response.candidates[0].finish_reason.name.lower()

                    final_chunk = LLMChatChunk(finish_reason=finish_reason, raw_chunk={})

                    if aggregated_response.usage_metadata:
                        usage_meta = aggregated_response.usage_metadata
                        final_usage_info = {
                            "prompt_tokens": usage_meta.prompt_token_count,
                            "completion_tokens": usage_meta.candidates_token_count,
                            "total_tokens": usage_meta.total_token_count,
                        }
                        final_chunk["usage_delta"] = final_usage_info
                        await self._record_usage_if_manager(usage_meta, "chat_stream_end")

                    if final_chunk.get("finish_reason") or final_chunk.get("usage_delta"):
                        yield final_chunk

                return stream_chat_generator()
            else:
                response: genai_types.GenerateContentResponse = await self._client.aio.models.generate_content(**request_kwargs)

                if not response.candidates and hasattr(response, "prompt_feedback") and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name
                    return LLMChatResponse(
                        message={"role": "assistant", "content": f"[Chat blocked: {reason}]"},
                        finish_reason=f"blocked: {reason.lower()}",
                        usage=None,
                        raw_response=self._minimal_serialize_gemini_response(response)
                    )

                assistant_response_content = response.text
                assistant_tool_calls: Optional[List[ToolCall]] = None
                if response.function_calls:
                    assistant_tool_calls = self._parse_gemini_tool_calls(response.function_calls)

                finish_reason = "unknown"
                if response.candidates:
                    finish_reason = genai_types.FinishReason(response.candidates[0].finish_reason).name.lower()

                usage_info: Optional[LLMUsageInfo] = None
                if response.usage_metadata:
                    usage_info = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.total_token_count,
                    }
                    await self._record_usage_if_manager(response.usage_metadata, "chat")

                final_assistant_message = ChatMessage(role="assistant")
                if assistant_response_content is not None:
                    final_assistant_message["content"] = assistant_response_content
                if assistant_tool_calls:
                    final_assistant_message["tool_calls"] = assistant_tool_calls

                return LLMChatResponse(
                    message=final_assistant_message,
                    finish_reason=finish_reason,
                    usage=usage_info,
                    raw_response=self._minimal_serialize_gemini_response(response)
                )
        except Exception as e:
            logger.error(f"{self.plugin_id}: Gemini API call failed: {e}", exc_info=True)
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    async def get_model_info(self) -> Dict[str, Any]:
        if not self._client or not self._client.aio:
            return {"error": "Gemini client or async client (aio) not initialized"}

        try:
            model_info_resp = await self._client.aio.models.get(name=f"models/{self._model_name}")
            return {
                "provider": "Google Gemini",
                "model_name_configured": self._model_name,
                "model_name_api": model_info_resp.name,
                "display_name": model_info_resp.display_name,
                "version": model_info_resp.version,
                "input_token_limit": model_info_resp.input_token_limit,
                "output_token_limit": model_info_resp.output_token_limit,
                "supported_generation_methods": model_info_resp.supported_generation_methods,
                "temperature_default": model_info_resp.temperature,
                "top_p_default": model_info_resp.top_p,
                "top_k_default": model_info_resp.top_k,
            }
        except Exception as e:
            logger.warning(f"{self.plugin_id}: Could not retrieve detailed model info for 'models/{self._model_name}': {e}")
            return {
                "provider": "Google Gemini",
                "model_name_configured": self._model_name,
                "error_retrieving_details": str(e)
            }

    async def teardown(self) -> None:
        self._client = None
        self._key_provider = None
        self._token_usage_manager = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
        await super().teardown()
