# tests/integration/test_rewoo_complex_financial_news.py
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from genie_tooling import tool
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField


# FIX: Renamed models to not start with "Test" to avoid PytestCollectionWarning
class ReWOOIntegrationStepModel(PydanticBaseModel):
    thought: str
    tool_id: str
    # FIX: The implementation expects a JSON string, not a dict.
    params: str
    output_variable_name: Optional[str] = None


class ReWOOIntegrationPlanModel(PydanticBaseModel):
    plan: List[ReWOOIntegrationStepModel]
    overall_reasoning: Optional[str] = None


from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import ChatMessage, LLMChatResponse

# --- Configuration ---
LLAMA_CPP_INTERNAL_MODEL_PATH_FOR_REWOO = (
    "/home/kal/code/models/Qwen3-8B.Q4_K_M.gguf"
)


# --- Mock Tools (Unchanged) ---
@tool
async def get_stock_ticker(
    company_name: str, context: Dict[str, Any]
) -> Dict[str, Optional[str]]:
    tickers = {"microsoft": "MSFT", "apple": "AAPL", "google": "GOOGL"}
    ticker = tickers.get(company_name.lower())
    if ticker:
        return {"ticker_symbol": ticker}
    return {"ticker_symbol": None, "error": f"Ticker not found for {company_name}"}


@tool
async def get_stock_price(
    ticker_symbol: str, context: Dict[str, Any]
) -> Dict[str, Optional[float]]:
    prices = {"MSFT": 350.75, "AAPL": 180.50, "GOOGL": 150.25}
    price = prices.get(ticker_symbol.upper())
    if price is not None:
        return {"current_price": price}
    return {"current_price": None, "error": f"Price not found for {ticker_symbol}"}


@tool
async def get_latest_news(
    company_name_or_ticker: str, context: Dict[str, Any], num_headlines: int = 1
) -> Dict[str, List[str]]:
    news_db = {
        "microsoft": [
            "Microsoft launches new AI chip series.",
            "Microsoft quarterly earnings exceed expectations.",
        ],
        "MSFT": ["MSFT stock up after AI chip announcement."],
        "apple": ["Apple reveals Vision Pro 2 details."],
        "AAPL": ["AAPL expected to announce new iPhone soon."],
    }
    key_to_check = company_name_or_ticker.lower()
    headlines = news_db.get(key_to_check, [])
    if not headlines and company_name_or_ticker.upper() in news_db:
        headlines = news_db.get(company_name_or_ticker.upper(), [])
    return {"headlines": headlines[:num_headlines]}


async def run_complex_rewoo_test():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("genie_tooling.command_processors.impl.rewoo_processor").setLevel(
        logging.DEBUG
    )
    logging.getLogger("genie_tooling").setLevel(logging.DEBUG)

    print("--- Genie Tooling: Complex ReWOO Multi-Step Financial News Test ---")

    model_path = Path(LLAMA_CPP_INTERNAL_MODEL_PATH_FOR_REWOO)
    if (
        not model_path.exists()
        or LLAMA_CPP_INTERNAL_MODEL_PATH_FOR_REWOO == "/path/to/your/model.gguf"
    ):
        print(
            "\nWARNING: LLM Model path not configured or file does not exist for ReWOO test base LLM."
        )
        llm_feature_setting = "none"
        llm_internal_model_path = None
    else:
        llm_feature_setting = "llama_cpp_internal"
        llm_internal_model_path = str(model_path.resolve())

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm=llm_feature_setting,  # type: ignore
            llm_llama_cpp_internal_model_path=llm_internal_model_path,
            command_processor="rewoo",
            prompt_template_engine="jinja2_chat_formatter",
            default_llm_output_parser="pydantic_output_parser",
            observability_tracer="console_tracer",
            logging_adapter="pyvider_log_adapter",
        ),
        auto_enable_registered_tools=False,
        tool_configurations={
            "get_stock_ticker": {},
            "get_stock_price": {},
            "get_latest_news": {},
        },
        command_processor_configurations={"rewoo_command_processor_v1": {"max_plan_retries": 0}},
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie for complex ReWOO test...")
        genie = await Genie.create(config=app_config)
        await genie.register_tool_functions(
            [get_stock_ticker, get_stock_price, get_latest_news]
        )
        print("Genie initialized and financial/news tools registered.")

        # FIX: Use renamed model and ensure params are JSON strings
        mock_plan_data = ReWOOIntegrationPlanModel(
            plan=[
                ReWOOIntegrationStepModel(
                    thought="First, I need to find the stock ticker for Microsoft.",
                    tool_id="get_stock_ticker",
                    params=json.dumps({"company_name": "Microsoft"}),
                    output_variable_name="ticker_info",
                ),
                ReWOOIntegrationStepModel(
                    thought="Now that I have the ticker, I can get the current stock price.",
                    tool_id="get_stock_price",
                    params=json.dumps(
                        {"ticker_symbol": "{{outputs.ticker_info.ticker_symbol}}"}
                    ),
                ),
                ReWOOIntegrationStepModel(
                    thought="Finally, I need to get the latest news headline for Microsoft.",
                    tool_id="get_latest_news",
                    params=json.dumps(
                        {"company_name_or_ticker": "Microsoft", "num_headlines": 1}
                    ),
                ),
            ]
        )

        expected_final_answer_str = "The stock ticker for Microsoft is MSFT, its current price is $350.75, and the latest news is: Microsoft launches new AI chip series."
        mock_solver_llm_response_content = LLMChatResponse(
            message={"role": "assistant", "content": expected_final_answer_str}
        )

        original_genie_llm_chat = genie.llm.chat
        original_genie_llm_parse_output = genie.llm.parse_output

        planner_parse_call_count = 0
        solver_chat_call_count = 0

        async def mock_genie_llm_parse_output_for_rewoo_test(
            response: Any, schema: Type[PydanticBaseModel], **kwargs
        ):
            nonlocal planner_parse_call_count
            if schema and hasattr(schema, "__name__") and "DynamicReWOOPlan" in schema.__name__:
                planner_parse_call_count += 1
                print(
                    "MOCK PARSE_OUTPUT: Intercepted call for ReWOO Plan. Returning mock plan instance based on DynamicReWOOPlan schema."
                )
                return schema(**mock_plan_data.model_dump())
            return await original_genie_llm_parse_output(
                response=response, schema=schema, **kwargs
            )

        async def mock_genie_llm_chat_for_rewoo_test(
            messages: List[ChatMessage], provider_id: Optional[str] = None, **kwargs
        ):
            nonlocal solver_chat_call_count
            prompt_content_for_solver_check = messages[-1]["content"]
            if (
                isinstance(prompt_content_for_solver_check, str)
                and "Original Goal:" in prompt_content_for_solver_check
                and "The following evidence was gathered"
                in prompt_content_for_solver_check
            ):
                solver_chat_call_count += 1
                print(
                    "MOCK CHAT: Intercepted call for ReWOO Solver. Returning mock solver response."
                )
                return mock_solver_llm_response_content
            return await original_genie_llm_chat(
                messages=messages, provider_id=provider_id, **kwargs
            )

        genie.llm.parse_output = mock_genie_llm_parse_output_for_rewoo_test
        genie.llm.chat = mock_genie_llm_chat_for_rewoo_test

        goal = "What is the stock price for Microsoft and their latest news headline?"
        print(f"\nSending command to ReWOO agent: '{goal}'")
        command_result = await genie.run_command(goal)

        print("\n--- ReWOO Agent Final Output ---")
        if command_result.get("error"):
            print(f"Error: {command_result['error']}")
            assert False, f"ReWOO command failed: {command_result['error']}"

        assert command_result.get("final_answer") == expected_final_answer_str
        print(f"Final Answer: {command_result['final_answer']}")

        assert planner_parse_call_count >= 1
        assert solver_chat_call_count == 1

        assert "llm_thought_process" in command_result
        if command_result.get("llm_thought_process"):
            try:
                thought_data = json.loads(command_result["llm_thought_process"])
                print(
                    "\n--- ReWOO Agent Thought Process (Plan & Evidence from result) ---"
                )
                print(json.dumps(thought_data, indent=2))

                assert "plan" in thought_data
                assert "evidence" in thought_data

                assert len(thought_data["plan"]["plan"]) == 3
                assert thought_data["plan"]["plan"][0]["tool_id"] == "get_stock_ticker"
                assert thought_data["plan"]["plan"][1]["tool_id"] == "get_stock_price"
                assert thought_data["plan"]["plan"][2]["tool_id"] == "get_latest_news"

                assert len(thought_data["evidence"]) == 3
                assert thought_data["evidence"][0]["result"] == {"ticker_symbol": "MSFT"}
                assert thought_data["evidence"][1]["result"] == {"current_price": 350.75}
                assert thought_data["evidence"][2]["result"] == {
                    "headlines": ["Microsoft launches new AI chip series."]
                }
            except json.JSONDecodeError:
                assert False, "llm_thought_process was not a valid JSON string"
        else:
            assert False, "llm_thought_process field is missing from command_result"

        assert genie.execute_tool.call_count == 3
        genie.execute_tool.assert_any_call(
            "get_stock_ticker", company_name="Microsoft"
        )
        genie.execute_tool.assert_any_call("get_stock_price", ticker_symbol="MSFT")
        genie.execute_tool.assert_any_call(
            "get_latest_news", company_name_or_ticker="Microsoft", num_headlines=1
        )

    except Exception as e:
        print(f"\nAn unexpected error occurred in the test: {e}")
        logging.exception("Complex ReWOO test error details:")
        raise
    finally:
        if genie:
            genie.llm.parse_output = original_genie_llm_parse_output
            genie.llm.chat = original_genie_llm_chat
            await genie.close()
            print("\nGenie facade torn down.")


if __name__ == "__main__":
    asyncio.run(run_complex_rewoo_test())