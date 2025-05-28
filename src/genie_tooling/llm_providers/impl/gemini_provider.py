# src/genie_tooling/llm_providers/impl/gemini_provider.py
import asyncio
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatResponse,
    LLMCompletionResponse,
    LLMUsageInfo,
    ToolCall as GenieToolCall, 
    ToolCallFunction as GenieToolCallFunction,
)
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

# Correct library: google-generativeai (installed via pip)
# Correct import: import google.generativeai as genai
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, ContentDict
    from google.generativeai.types import Tool as GeminiSDKTool # For future tool use integration
    from google.generativeai.types import FunctionDeclaration # For future tool use integration
    from google.generativeai.types.generation_types import GenerateContentResponse, Candidate
except ImportError:
    genai = None 
    GenerationConfig = None 
    ContentDict = None 
    GeminiSDKTool = None 
    FunctionDeclaration = None
    GenerateContentResponse = None 
    Candidate = None 
    logger.warning(
        "GeminiLLMProviderPlugin: 'google-generativeai' library not installed. "
        "This plugin will not be functional. Please install it: pip install google-generativeai"
    )

# Mapping from our generic roles to Gemini's roles
GEMINI_ROLE_MAP = {
    "user": "user",
    "assistant": "model", 
    "system": "user", # System prompts for Gemini are often handled differently (see setup)
    "tool": "tool",
}

GEMINI_FINISH_REASON_MAP = {
    1: "stop",       # FINISH_REASON_STOP
    2: "length",     # FINISH_REASON_MAX_TOKENS
    3: "safety",     # FINISH_REASON_SAFETY
    4: "recitation", # FINISH_REASON_RECITATION
    0: "unknown",    # FINISH_REASON_UNSPECIFIED
    5: "other",      # FINISH_REASON_OTHER
    # 6: "tool_code", # FINISH_REASON_TOOL_CODE - This indicates function calling is expected.
}

class GeminiLLMProviderPlugin(LLMProviderPlugin):
    plugin_id: str = "gemini_llm_provider_v1"
    description: str = "LLM provider for Google's Gemini models using the google-generativeai library."

    _model_client: Optional[genai.GenerativeModel] = None # Correct type
    _model_name: str
    _api_key_name: str = "GOOGLE_API_KEY" # Default name for KeyProvider

    async def setup(self, config: Optional[Dict[str, Any]], key_provider: KeyProvider) -> None:
        await super().setup(config, key_provider)
        if not genai:
            logger.error(f"{self.plugin_id}: 'google-generativeai' library is not available. Cannot proceed.")
            return

        cfg = config or {}
        self._api_key_name = cfg.get("api_key_name", self._api_key_name)
        self._model_name = cfg.get("model_name", "gemini-1.5-flash-latest") 

        api_key = await key_provider.get_key(self._api_key_name)
        if not api_key:
            logger.error(f"{self.plugin_id}: API key '{self._api_key_name}' not found via KeyProvider.")
            return

        try:
            # Configure the library with the API key
            genai.configure(api_key=api_key)
            
            system_instruction_text = cfg.get("system_instruction")
            
            # Safety settings from config
            safety_settings_config = cfg.get("safety_settings") # e.g., [{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}]
            
            self._model_client = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=system_instruction_text, # Handles system prompt
                safety_settings=safety_settings_config # Pass safety settings
            )
            logger.info(f"{self.plugin_id}: Initialized Gemini client for model '{self._model_name}'. System Instruction set: {bool(system_instruction_text)}. Safety Settings configured: {bool(safety_settings_config)}")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize Gemini client for model '{self._model_name}': {e}", exc_info=True)
            self._model_client = None

    def _convert_messages_to_gemini(self, messages: List[ChatMessage]) -> List[ContentDict]:
        gemini_messages: List[ContentDict] = []
        for msg in messages:
            role = GEMINI_ROLE_MAP.get(msg["role"])
            if not role and msg["role"] == "system": # Handle system message if not directly mapped
                # System messages are handled by GenerativeModel's system_instruction now,
                # but if one appears in history, it might need to be a user message or handled specially.
                # For now, let's make it a user message if it's not the first message and system_instruction is not used.
                # This part can be refined based on specific multi-turn system prompt strategies.
                logger.warning(f"{self.plugin_id}: Converting 'system' role in history to 'user' for Gemini. Prefer system_instruction in setup.")
                role = "user" 
            
            if not role:
                logger.warning(f"{self.plugin_id}: Unsupported role '{msg['role']}' for Gemini, skipping message.")
                continue
            
            parts: list = [] # Parts can be text or function_call/function_response
            if msg.get("content"):
                parts.append({"text": msg["content"]}) # type: ignore

            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]: # type: ignore
                    try:
                        # Gemini expects 'args' as a dict, not a JSON string.
                        arguments = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                        parts.append({ # type: ignore
                            "function_call": { 
                                "name": tc["function"]["name"],
                                "args": arguments
                            }
                        })
                    except json.JSONDecodeError:
                        logger.error(f"{self.plugin_id}: Could not parse tool_call arguments for {tc['function']['name']}. Skipping tool call part.")
            
            if msg["role"] == "tool" and msg.get("tool_call_id") and msg.get("content"):
                # tool_call_id maps to the 'name' of the function in function_response for Gemini
                tool_name = msg.get("name", msg.get("tool_call_id")) # Prefer 'name' if present
                try:
                    # Gemini function_response 'response' expects a dict, often {"content": actual_response_str}
                    # or could be structured if the tool output schema is defined.
                    tool_response_content = msg.get("content")
                    response_part_content = {"content": tool_response_content} # Simple wrapping
                    # If tool_response_content is already a dict from tool execution, use it directly
                    # if isinstance(tool_response_content, dict):
                    #    response_part_content = tool_response_content

                    parts.append({ # type: ignore
                        "function_response": {
                            "name": tool_name, 
                            "response": response_part_content 
                        }
                    })
                except Exception as e:
                    logger.error(f"{self.plugin_id}: Error processing tool response for {tool_name}: {e}")


            if parts:
                 gemini_messages.append(ContentDict(role=role, parts=parts)) # type: ignore
        return gemini_messages

    def _parse_gemini_candidate(self, candidate: Candidate) -> Tuple[ChatMessage, Optional[str]]:
        role: Literal["assistant"] = "assistant"
        content: Optional[str] = None
        tool_calls: Optional[List[GenieToolCall]] = None

        if candidate.content and candidate.content.parts:
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text is not None]
            if text_parts:
                content = " ".join(text_parts)
            
            function_calls_from_gemini = [part.function_call for part in candidate.content.parts if hasattr(part, 'function_call')]
            if function_calls_from_gemini:
                tool_calls = []
                for i, fc in enumerate(function_calls_from_gemini):
                    tool_call_id = f"call_{fc.name}_{i}_{hash(str(fc.args))}" # More unique ID
                    tool_calls.append({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(fc.args or {}) 
                        }
                    })
        
        chat_message: ChatMessage = {"role": role}
        if content is not None: # Only add content if it's not None
            chat_message["content"] = content
        if tool_calls:
            chat_message["tool_calls"] = tool_calls
        
        finish_reason_enum = getattr(candidate, 'finish_reason', 0)
        finish_reason_str = GEMINI_FINISH_REASON_MAP.get(finish_reason_enum, "unknown")

        # Check for implicit tool calls finish reason
        if function_calls_from_gemini and finish_reason_enum != 3 and finish_reason_enum != 4: # Not safety/recitation
            # If the model stopped to call functions, this is the reason.
             is_native_tool_call_reason = (hasattr(candidate, 'finish_reason') and candidate.finish_reason.name == 'TOOL_CODE') # FINISH_REASON_TOOL_CODE = 6 (not in map yet)
             if is_native_tool_call_reason or function_calls_from_gemini: # If there are function calls, it's tool_calls
                finish_reason_str = "tool_calls"

        return chat_message, finish_reason_str

    async def _execute_gemini_request(self, gemini_formatted_messages: List[ContentDict], generation_config_args: Dict[str,Any], tools_arg: Optional[List[Any]], safety_settings_arg: Optional[List[Any]], stream:bool) -> GenerateContentResponse:
        if not self._model_client:
            raise RuntimeError(f"{self.plugin_id}: Model client not initialized.")

        generation_config = GenerationConfig(**generation_config_args) if generation_config_args else None # type: ignore

        try:
            loop = asyncio.get_running_loop()
            response: GenerateContentResponse = await loop.run_in_executor(
                None,
                self._model_client.generate_content,
                gemini_formatted_messages,
                generation_config=generation_config,
                tools=tools_arg, # type: ignore
                safety_settings=safety_settings_arg, # type: ignore
                stream=stream
            )
            return response
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error during Gemini API call: {e}", exc_info=True)
            raise RuntimeError(f"Gemini API call failed: {str(e)}") from e

    async def generate(self, prompt: str, **kwargs: Any) -> LLMCompletionResponse:
        if not self._model_client:
            raise RuntimeError(f"{self.plugin_id}: Model client not initialized.")

        gemini_messages = self._convert_messages_to_gemini([{"role": "user", "content": prompt}])
        logger.debug(f"{self.plugin_id}: Sending generate request to Gemini. Model: {self._model_name}, Prompt: '{prompt[:50]}...'")
        
        # Extract relevant kwargs for GenerationConfig
        gen_config_args = {k: v for k, v in kwargs.items() if k in ["temperature", "top_p", "top_k", "max_output_tokens", "stop_sequences", "candidate_count"]}
        
        api_response = await self._execute_gemini_request(gemini_messages, gen_config_args, None, kwargs.get("safety_settings"), stream=False)

        text_output = ""
        finish_reason_str: Optional[str] = "unknown"

        if api_response.candidates:
            candidate = api_response.candidates[0]
            chat_msg, temp_finish_reason = self._parse_gemini_candidate(candidate)
            text_output = chat_msg.get("content") or ""
            finish_reason_str = temp_finish_reason # Use finish reason from parsed candidate
        elif api_response.prompt_feedback and api_response.prompt_feedback.block_reason: # type: ignore
            logger.warning(f"{self.plugin_id}: Gemini prompt blocked. Reason: {api_response.prompt_feedback.block_reason.name}") # type: ignore
            finish_reason_str = f"blocked: {api_response.prompt_feedback.block_reason.name}" # type: ignore
            text_output = f"[Content generation blocked due to: {api_response.prompt_feedback.block_reason.name}]" # type: ignore

        usage_info: Optional[LLMUsageInfo] = None
        if hasattr(api_response, "usage_metadata") and api_response.usage_metadata:
            um = api_response.usage_metadata
            usage_info = {
                "prompt_tokens": um.prompt_token_count,
                "completion_tokens": um.candidates_token_count,
                "total_tokens": um.total_token_count,
            }
        
        return {
            "text": text_output,
            "finish_reason": finish_reason_str,
            "usage": usage_info,
            "raw_response": api_response.to_dict() if hasattr(api_response, "to_dict") else str(api_response)
        }

    async def chat(self, messages: List[ChatMessage], **kwargs: Any) -> LLMChatResponse:
        if not self._model_client:
            raise RuntimeError(f"{self.plugin_id}: Model client not initialized.")

        gemini_messages = self._convert_messages_to_gemini(messages)
        logger.debug(f"{self.plugin_id}: Sending chat request to Gemini. Model: {self._model_name}, Messages: {len(messages)}")

        gen_config_args = {k: v for k, v in kwargs.items() if k in ["temperature", "top_p", "top_k", "max_output_tokens", "stop_sequences", "candidate_count"]}
        tools_arg = kwargs.get("tools") # For function calling
        safety_settings_arg = kwargs.get("safety_settings")

        api_response = await self._execute_gemini_request(gemini_messages, gen_config_args, tools_arg, safety_settings_arg, stream=False)
        
        assistant_chat_message: ChatMessage = {"role": "assistant", "content": ""}
        finish_reason_str: Optional[str] = "unknown"

        if api_response.candidates:
            candidate = api_response.candidates[0]
            assistant_chat_message, temp_finish_reason = self._parse_gemini_candidate(candidate)
            finish_reason_str = temp_finish_reason
        elif api_response.prompt_feedback and api_response.prompt_feedback.block_reason: # type: ignore
            logger.warning(f"{self.plugin_id}: Gemini chat prompt blocked. Reason: {api_response.prompt_feedback.block_reason.name}") # type: ignore
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
            "raw_response": api_response.to_dict() if hasattr(api_response, "to_dict") else str(api_response)
        }

    async def get_model_info(self) -> Dict[str, Any]:
        if not genai or not self._model_client:
            return {"error": "Gemini client not initialized or library not available."}
        
        info: Dict[str, Any] = {"provider": "Google Gemini", "configured_model_name": self._model_name}
        try:
            loop = asyncio.get_running_loop()
            # The genai.get_model() function requires a string like "models/gemini-1.5-flash-latest"
            model_info_sdk = await loop.run_in_executor(None, genai.get_model, f"models/{self._model_name}") # type: ignore
            
            if model_info_sdk:
                info["display_name"] = getattr(model_info_sdk, "display_name", None)
                info["version"] = getattr(model_info_sdk, "version", None)
                info["input_token_limit"] = getattr(model_info_sdk, "input_token_limit", None)
                info["output_token_limit"] = getattr(model_info_sdk, "output_token_limit", None)
                info["supported_generation_methods"] = getattr(model_info_sdk, "supported_generation_methods", None)
            else:
                info["error"] = "Could not retrieve model info from SDK."
            
        except Exception as e:
            logger.warning(f"{self.plugin_id}: Could not fetch detailed model info for '{self._model_name}' from Gemini: {e}")
            info["model_info_error"] = str(e)
        return info

    async def teardown(self) -> None:
        self._model_client = None 
        logger.info(f"{self.plugin_id}: Teardown complete (client reference released).")
        await super().teardown()