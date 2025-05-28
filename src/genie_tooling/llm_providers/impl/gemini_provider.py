# src/genie_tooling/llm_providers/impl/gemini_provider.py
import asyncio
import logging
from typing import Any, Dict, List, Optional, cast, Literal, Tuple
import json


from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatResponse,
    LLMCompletionResponse,
    LLMUsageInfo,
    ToolCall as GenieToolCall, # Renamed to avoid conflict
    ToolCallFunction as GenieToolCallFunction,
)
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

# Optional import for Google Gemini
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, ContentDict, Tool as GeminiSDKTool, FunctionDeclaration
    from google.generativeai.types.generation_types import GenerateContentResponse, Candidate
except ImportError:
    genai = None # type: ignore
    GenerationConfig = None # type: ignore
    ContentDict = None # type: ignore
    GeminiSDKTool = None # type: ignore
    FunctionDeclaration = None #type: ignore
    GenerateContentResponse = None # type: ignore
    Candidate = None # type: ignore
    logger.warning("GeminiLLMProviderPlugin: 'google-generativeai' library not installed. This plugin will not be functional.")


# Mapping from our generic roles to Gemini's roles
GEMINI_ROLE_MAP = {
    "user": "user",
    "assistant": "model", # Gemini uses "model" for assistant messages
    "system": "user", # System prompts often go as the first user message or in specific instruction fields
    "tool": "tool", # Gemini uses "tool" role for tool responses.
}

# Mapping from Gemini's finish reasons to our generic ones
GEMINI_FINISH_REASON_MAP = {
    1: "stop",       # FINISH_REASON_STOP
    2: "length",     # FINISH_REASON_MAX_TOKENS
    3: "safety",     # FINISH_REASON_SAFETY
    4: "recitation", # FINISH_REASON_RECITATION
    0: "unknown",    # FINISH_REASON_UNSPECIFIED
    5: "other",      # FINISH_REASON_OTHER
    # Tool calling specific reasons might need custom handling if Gemini defines them
    # For example, if a candidate has function_calls, finish_reason might be "tool_calls" implicitly
}


class GeminiLLMProviderPlugin(LLMProviderPlugin):
    plugin_id: str = "gemini_llm_provider_v1"
    description: str = "LLM provider for Google's Gemini models (e.g., gemini-pro, gemini-1.5-flash)."

    _model_client: Optional[Any] = None # genai.GenerativeModel
    _model_name: str
    _api_key_name: str = "GOOGLE_API_KEY"

    async def setup(self, config: Optional[Dict[str, Any]], key_provider: KeyProvider) -> None:
        await super().setup(config, key_provider)
        if not genai:
            logger.error(f"{self.plugin_id}: 'google-generativeai' library is not installed. Cannot proceed.")
            return

        cfg = config or {}
        self._api_key_name = cfg.get("api_key_name", self._api_key_name)
        self._model_name = cfg.get("model_name", "gemini-1.5-flash-latest") # A good default

        api_key = await key_provider.get_key(self._api_key_name)
        if not api_key:
            logger.error(f"{self.plugin_id}: API key '{self._api_key_name}' not found via KeyProvider.")
            return

        try:
            genai.configure(api_key=api_key)
            
            # System instruction handling (if provided in config)
            system_instruction_text = cfg.get("system_instruction")
            system_instruction_content = None
            if system_instruction_text:
                system_instruction_content = ContentDict(role="system", parts=[{"text": system_instruction_text}]) # type: ignore
            
            self._model_client = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=system_instruction_content # type: ignore
            )
            logger.info(f"{self.plugin_id}: Initialized. Model: {self._model_name}. System Instruction set: {bool(system_instruction_text)}")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize Gemini client for model '{self._model_name}': {e}", exc_info=True)
            self._model_client = None

    def _convert_messages_to_gemini(self, messages: List[ChatMessage]) -> List[ContentDict]:
        gemini_messages: List[ContentDict] = []
        for msg in messages:
            role = GEMINI_ROLE_MAP.get(msg["role"])
            if not role:
                logger.warning(f"{self.plugin_id}: Unsupported role '{msg['role']}' for Gemini, skipping message.")
                continue
            
            parts = []
            if msg.get("content"):
                parts.append({"text": msg["content"]})

            # Handle tool calls (outgoing from assistant)
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]: # type: ignore
                    parts.append({
                        "function_call": {
                            "name": tc["function"]["name"],
                            "args": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                        }
                    })
            
            # Handle tool responses (incoming to model)
            if msg["role"] == "tool" and msg.get("tool_call_id") and msg.get("content"):
                 parts.append({
                    "function_response": {
                        "name": msg.get("name", "unknown_tool_function"), # Tool name should be here
                        "response": {"content": msg.get("content")} # Gemini expects response to be a dict
                    }
                })
                # Gemini tool role messages expect tool_call_id at a higher level, not directly in ContentDict usually.
                # This might need adjustment depending on how chat history with tool calls is constructed.
                # For now, ContentDict only takes role and parts. The tool_call_id mapping is tricky here.
                # Typically, for history, one might not need to re-specify the tool_call_id for model consumption.
                # It's more for the assistant's response.

            if parts: # Only add message if it has parts
                 gemini_messages.append(ContentDict(role=role, parts=parts)) # type: ignore
        return gemini_messages

    def _parse_gemini_candidate(self, candidate: Candidate) -> Tuple[ChatMessage, Optional[str]]:
        """Parses a Gemini Candidate object into our ChatMessage and finish_reason."""
        role: Literal["assistant"] = "assistant" # Gemini candidate responses are from the model
        content: Optional[str] = None
        tool_calls: Optional[List[GenieToolCall]] = None

        if candidate.content and candidate.content.parts:
            # Aggregate text parts
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
            if text_parts:
                content = " ".join(text_parts)
            
            # Check for function calls
            function_calls_from_gemini = [part.function_call for part in candidate.content.parts if hasattr(part, 'function_call')]
            if function_calls_from_gemini:
                tool_calls = []
                for fc in function_calls_from_gemini:
                    # Gemini function call `args` are already dicts. `arguments` in our spec is a JSON string.
                    tool_calls.append({
                        "id": f"call_{fc.name}_{hash(str(fc.args))}", # Create a stable ID
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(fc.args or {}) # Convert dict to JSON string
                        }
                    })
        
        chat_message: ChatMessage = {"role": role, "content": content}
        if tool_calls:
            chat_message["tool_calls"] = tool_calls
        
        finish_reason_enum = getattr(candidate, 'finish_reason', 0) # Access safely
        finish_reason_str = GEMINI_FINISH_REASON_MAP.get(finish_reason_enum, "unknown")
        if tool_calls and finish_reason_str == "stop": # If there are tool calls, override "stop"
            finish_reason_str = "tool_calls"
            
        return chat_message, finish_reason_str

    async def _execute_gemini_request(self, gemini_formatted_messages: List[ContentDict], **kwargs: Any) -> GenerateContentResponse:
        if not self._model_client:
            raise RuntimeError(f"{self.plugin_id}: Model client not initialized.")

        # Prepare GenerationConfig
        gen_config_params = {
            key: value for key, value in kwargs.items() 
            if key in ["temperature", "top_p", "top_k", "max_output_tokens", "stop_sequences", "candidate_count"]
        }
        generation_config = GenerationConfig(**gen_config_params) if gen_config_params else None # type: ignore

        # Prepare Tools (if any specified in kwargs, for function calling)
        # This part is a simplified V1. Real tool integration needs schema mapping.
        tools_arg = kwargs.get("tools") # Expects list of GeminiSDKTool or FunctionDeclaration
        safety_settings = kwargs.get("safety_settings") # List of SafetySetting

        try:
            # Use run_in_executor for the blocking SDK call
            loop = asyncio.get_running_loop()
            response: GenerateContentResponse = await loop.run_in_executor(
                None,
                self._model_client.generate_content, # type: ignore
                gemini_formatted_messages,
                generation_config=generation_config,
                tools=tools_arg,
                safety_settings=safety_settings,
                # stream=False # Default for generate_content
            )
            return response
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error during Gemini API call: {e}", exc_info=True)
            # Attempt to parse Google specific API errors if possible
            # For now, re-raise as a generic error.
            raise RuntimeError(f"Gemini API call failed: {str(e)}") from e


    async def generate(self, prompt: str, **kwargs: Any) -> LLMCompletionResponse:
        if not self._model_client:
            raise RuntimeError(f"{self.plugin_id}: Model client not initialized.")

        gemini_messages = self._convert_messages_to_gemini([{"role": "user", "content": prompt}])
        logger.debug(f"{self.plugin_id}: Sending generate request to Gemini. Model: {self._model_name}, Prompt: '{prompt[:50]}...'")
        
        api_response = await self._execute_gemini_request(gemini_messages, **kwargs)

        text_output = ""
        finish_reason_str: Optional[str] = "unknown"

        if api_response.candidates:
            # For simple generation, usually one candidate
            candidate = api_response.candidates[0]
            chat_msg, finish_reason_str = self._parse_gemini_candidate(candidate)
            text_output = chat_msg.get("content") or ""
        else: # Prompt might have been blocked
            logger.warning(f"{self.plugin_id}: Gemini response has no candidates. Prompt may have been blocked. Response: {api_response.prompt_feedback}")
            # Determine finish_reason from prompt_feedback if possible
            if api_response.prompt_feedback and api_response.prompt_feedback.block_reason: # type: ignore
                 finish_reason_str = f"blocked: {api_response.prompt_feedback.block_reason.name}" # type: ignore

        usage_info: Optional[LLMUsageInfo] = None
        if hasattr(api_response, "usage_metadata") and api_response.usage_metadata:
            um = api_response.usage_metadata
            usage_info = {
                "prompt_tokens": um.prompt_token_count,
                "completion_tokens": um.candidates_token_count, # Sum of tokens in all candidates
                "total_tokens": um.total_token_count,
            }
        
        return {
            "text": text_output,
            "finish_reason": finish_reason_str,
            "usage": usage_info,
            "raw_response": api_response, # Or convert to dict: candidate.to_dict()
        }

    async def chat(self, messages: List[ChatMessage], **kwargs: Any) -> LLMChatResponse:
        if not self._model_client:
            raise RuntimeError(f"{self.plugin_id}: Model client not initialized.")

        gemini_messages = self._convert_messages_to_gemini(messages)
        logger.debug(f"{self.plugin_id}: Sending chat request to Gemini. Model: {self._model_name}, Messages: {len(messages)}")

        api_response = await self._execute_gemini_request(gemini_messages, **kwargs)
        
        assistant_chat_message: ChatMessage = {"role": "assistant", "content": ""} # Default empty
        finish_reason_str: Optional[str] = "unknown"

        if api_response.candidates:
            # For chat, typically one primary candidate is used.
            # Multi-candidate responses would require more complex handling.
            candidate = api_response.candidates[0]
            assistant_chat_message, finish_reason_str = self._parse_gemini_candidate(candidate)
        else: # Prompt might have been blocked
            logger.warning(f"{self.plugin_id}: Gemini chat response has no candidates. Prompt may have been blocked. Response: {api_response.prompt_feedback}")
            if api_response.prompt_feedback and api_response.prompt_feedback.block_reason: # type: ignore
                 finish_reason_str = f"blocked: {api_response.prompt_feedback.block_reason.name}" # type: ignore
                 assistant_chat_message["content"] = f"[Chat blocked due to: {api_response.prompt_feedback.block_reason.name}]" # type: ignore

        usage_info: Optional[LLMUsageInfo] = None
        if hasattr(api_response, "usage_metadata") and api_response.usage_metadata:
            um = api_response.usage_metadata
            usage_info = {
                "prompt_tokens": um.prompt_token_count,
                "completion_tokens": um.candidates_token_count,
                "total_tokens": um.total_token_count,
            }

        return {
            "message": assistant_chat_message,
            "finish_reason": finish_reason_str,
            "usage": usage_info,
            "raw_response": api_response,
        }

    async def get_model_info(self) -> Dict[str, Any]:
        if not genai or not self._model_client:
            return {"error": "Gemini client not initialized or library not available."}
        
        info: Dict[str, Any] = {"provider": "Google Gemini", "configured_model_name": self._model_name}
        try:
            loop = asyncio.get_running_loop()
            model_info_sdk = await loop.run_in_executor(None, genai.get_model, f"models/{self._model_name}") # type: ignore
            
            # Extract relevant details
            info["display_name"] = getattr(model_info_sdk, "display_name", None)
            info["version"] = getattr(model_info_sdk, "version", None)
            info["input_token_limit"] = getattr(model_info_sdk, "input_token_limit", None)
            info["output_token_limit"] = getattr(model_info_sdk, "output_token_limit", None)
            info["supported_generation_methods"] = getattr(model_info_sdk, "supported_generation_methods", None)
            
        except Exception as e:
            logger.warning(f"{self.plugin_id}: Could not fetch detailed model info for '{self._model_name}' from Gemini: {e}")
            info["model_info_error"] = str(e)
        return info

    async def teardown(self) -> None:
        self._model_client = None # Client itself doesn't have an explicit close usually
        logger.info(f"{self.plugin_id}: Teardown complete.")
        await super().teardown()