### tests/unit/agents/test_deep_research_agent.py
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Type
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
from genie_tooling.agents.deep_research_agent import (
    ARXIV_EXTRACTOR_PROMPT_ID,
    CONTROLLER_PROMPT_ID,
    FINAL_SYNTHESIZER_PROMPT_ID,
    ITERATIVE_SYNTHESIZER_PROMPT_ID,
    PLANNER_PROMPT_ID,
    RAG_EXTRACTOR_PROMPT_ID,
    TOOL_SELECTOR_PROMPT_ID,
    WEB_EXTRACTOR_PROMPT_ID,
    ControllerOutputModel,
    DeepResearchAgent,
    ExtractionOutputModel,
    PlannerOutputModel,
    ProcessedSnippet,
    ResearchTaskModel,
    ToolSelectionOutputModel,
)
from genie_tooling.agents.types import AgentOutput
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.interfaces import PromptInterface
from genie_tooling.llm_providers.types import ChatMessage, LLMChatResponse
from genie_tooling.lookup.types import RankedToolResult
from pydantic import BaseModel

logger = logging.getLogger(__name__)
AGENT_LOGGER_NAME = "genie_tooling.agents.deep_research_agent"


@pytest.fixture
def mock_genie_config_for_dra() -> MiddlewareConfig:
    """Provides a basic MiddlewareConfig for agent tests."""
    return MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="test_model_for_agents",
            prompt_registry="file_system_prompt_registry",
        ),
        tool_configurations={
            "google_search_tool_v1": {},
            "arxiv_search_tool": {},
            # RAG is handled internally by the agent, not as a typical "tool"
            # that needs to be listed here for the agent's direct execution.
        },
    )


@pytest.fixture
def mock_genie_for_dra(
    mocker, mock_genie_config_for_dra: MiddlewareConfig
) -> MagicMock:
    """Mocks the Genie facade for DeepResearchAgent tests."""
    genie_mock = MagicMock(name="MockGenieFacadeForDRA")
    genie_mock._config = mock_genie_config_for_dra

    # LLM Interface
    genie_mock.llm = AsyncMock(name="MockLLMInterfaceOnGenieDRA")
    genie_mock.llm.chat = AsyncMock(
        return_value=LLMChatResponse(
            message=ChatMessage(role="assistant", content="LLM default response"),
            raw_response={},
            usage={"total_tokens": 10},
        )
    )
    genie_mock.llm.parse_output = AsyncMock()

    # Prompts Interface
    prompts_mock = MagicMock(
        spec=PromptInterface, name="MockPromptsInterfaceOnGenieDRA"
    )
    prompts_mock.render_chat_prompt = AsyncMock(
        return_value=[{"role": "user", "content": "Rendered prompt"}]
    )
    genie_mock.prompts = prompts_mock

    # Observability Interface
    genie_mock.observability = AsyncMock(name="MockObservabilityInterfaceOnGenieDRA")
    genie_mock.observability.trace_event = AsyncMock()

    # Tool Manager & Lookup Service
    genie_mock._tool_manager = AsyncMock(name="MockToolManagerOnGenieDRA")
    genie_mock._tool_manager.get_formatted_tool_definition = AsyncMock(
        return_value="Formatted Tool String"
    )
    genie_mock._tool_lookup_service = AsyncMock(
        name="MockToolLookupServiceOnGenieDRA"
    )
    genie_mock._tool_lookup_service.find_tools = AsyncMock(return_value=[])

    # RAG and Execute Tool
    genie_mock.rag = AsyncMock(name="MockRAGInterfaceOnGenieDRA")
    genie_mock.rag.search = AsyncMock(return_value=[])
    genie_mock.execute_tool = AsyncMock(
        return_value={"result": "Tool executed successfully"}
    )

    return genie_mock


@pytest.fixture
def deep_research_agent(mock_genie_for_dra: MagicMock) -> DeepResearchAgent:
    """Provides a DeepResearchAgent instance with mocked Genie."""
    # Add a placeholder tool_id for rag_search in the agent's default config
    # This helps the _select_tool_for_sub_query method.
    agent = DeepResearchAgent(genie=mock_genie_for_dra)
    agent.tool_configs["rag_search"]["tool_id"] = "internal_rag_search_tool"
    agent.tool_configs["rag_search"]["description_for_llm"] = "Searches internal knowledge base for relevant documents."
    return agent


# REMOVED @pytest.mark.asyncio from synchronous tests
class TestDeepResearchAgentInit:
    def test_dra_init_defaults(self, mock_genie_for_dra: MagicMock):
        agent = DeepResearchAgent(genie=mock_genie_for_dra)
        assert agent.max_total_gathering_cycles == 15
        assert agent.max_plan_refinement_cycles == 2
        assert agent.max_sub_questions_per_plan == 7
        assert agent.llm_call_retries == 1
        assert agent.tool_configs["web_search"]["tool_id"] == "google_search_tool_v1"
        assert agent.tool_configs["arxiv_search"]["tool_id"] == "arxiv_search_tool"
        assert (
            agent.tool_configs["rag_search"]["collection_name"]
            == "main_knowledge_base"
        )
        assert (
            agent.tool_definition_formatter_id == "compact_text_formatter_plugin_v1"
        )

    def test_dra_init_custom_config(self, mock_genie_for_dra: MagicMock):
        custom_config = {
            "max_total_gathering_cycles": 5,
            "max_plan_refinement_cycles": 1,
            "max_sub_questions_per_plan": 3,
            "llm_call_retries": 2,
            "web_search_tool_config": {"tool_id": "my_web_search", "num_results": 2},
            "arxiv_search_tool_config": {
                "tool_id": "my_arxiv",
                "max_results": 1,
            },
            "rag_search_tool_config": {
                "collection_name": "custom_rag",
                "top_k": 1,
            },
            "tool_definition_formatter_id": "custom_formatter",
            "planner_llm_provider_id": "planner_llm",
        }
        agent = DeepResearchAgent(genie=mock_genie_for_dra, agent_config=custom_config)
        assert agent.max_total_gathering_cycles == 5
        assert agent.max_plan_refinement_cycles == 1
        assert agent.max_sub_questions_per_plan == 3
        assert agent.llm_call_retries == 2
        assert agent.tool_configs["web_search"]["tool_id"] == "my_web_search"
        assert agent.tool_configs["arxiv_search"]["tool_id"] == "my_arxiv"
        assert agent.tool_configs["rag_search"]["collection_name"] == "custom_rag"
        assert agent.tool_definition_formatter_id == "custom_formatter"
        assert agent.planner_llm_provider_id == "planner_llm"


@pytest.mark.asyncio
class TestDeepResearchAgentInternalMethods:
    async def test_call_llm_with_retry_success_first_try(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra: MagicMock
    ):
        class OutputSchema(BaseModel):
            data: str

        expected_parsed_output = OutputSchema(data="parsed_data")
        mock_genie_for_dra.llm.chat.return_value = LLMChatResponse(
            message=ChatMessage(role="assistant", content='{"data": "parsed_data"}'),
            raw_response={},
            usage={"total_tokens": 5},
        )
        mock_genie_for_dra.llm.parse_output.return_value = expected_parsed_output

        result = await deep_research_agent._call_llm_with_retry(
            messages=[{"role": "user", "content": "test"}], output_schema=OutputSchema
        )

        assert result == expected_parsed_output
        mock_genie_for_dra.llm.chat.assert_awaited_once()
        mock_genie_for_dra.llm.parse_output.assert_awaited_once()

    async def test_call_llm_with_retry_fails_then_succeeds(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra: MagicMock
    ):
        deep_research_agent.llm_call_retries = 1 # Ensure it retries once
        mock_genie_for_dra.llm.chat.side_effect = [
            RuntimeError("LLM API failed first time"),
            LLMChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content='{"sub_query": "q1", "suggested_tool_types": ["web_search"]}',
                ),
                raw_response={},
                usage={"total_tokens": 10},
            ),
        ]
        mock_genie_for_dra.llm.parse_output.return_value = ResearchTaskModel(
            sub_query="q1", suggested_tool_types=["web_search"]
        )

        with patch("asyncio.sleep", AsyncMock()):  # Speed up test
            result = await deep_research_agent._call_llm_with_retry(
                messages=[{"role": "user", "content": "test"}],
                output_schema=ResearchTaskModel,
            )
        assert isinstance(result, ResearchTaskModel)
        assert result.sub_query == "q1"
        assert mock_genie_for_dra.llm.chat.call_count == 2

    async def test_generate_initial_plan_success(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra: MagicMock
    ):
        expected_plan_output = PlannerOutputModel(
            research_tasks=[
                ResearchTaskModel(
                    sub_query="Define RAG", suggested_tool_types=["rag_search"]
                )
            ],
            estimated_depth="Quick Overview",
            overall_reasoning="To understand RAG.",
        )
        mock_genie_for_dra.llm.parse_output.return_value = expected_plan_output

        plan, depth = await deep_research_agent._generate_initial_plan("What is RAG?")

        assert plan == expected_plan_output.research_tasks
        assert depth == expected_plan_output.estimated_depth
        mock_genie_for_dra.prompts.render_chat_prompt.assert_awaited_with(
            name=PLANNER_PROMPT_ID,
            data={
                "goal": "What is RAG?",
                "planner_schema_dict_for_template": PlannerOutputModel.model_json_schema(),
            },
        )

    async def test_select_tool_for_sub_query_success_rag(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra: MagicMock
    ):
        # RAG is a special case, tool_lookup_service might not be called,
        # or it might return a conceptual RAG tool.
        # The agent's tool_configs should have a placeholder for RAG.
        rag_tool_id_placeholder = deep_research_agent.tool_configs["rag_search"]["tool_id"]
        rag_tool_description = deep_research_agent.tool_configs["rag_search"]["description_for_llm"]

        mock_genie_for_dra._tool_lookup_service.find_tools.return_value = [
            RankedToolResult(tool_identifier=rag_tool_id_placeholder, score=0.9)
        ]
        mock_genie_for_dra._tool_manager.get_formatted_tool_definition.return_value = (
            f"Formatted: {rag_tool_description}"
        )
        expected_selection = ToolSelectionOutputModel(
            selected_tool_id=rag_tool_id_placeholder, reasoning="RAG is best for definitions"
        )
        mock_genie_for_dra.llm.parse_output.return_value = expected_selection

        result = await deep_research_agent._select_tool_for_sub_query(
            "Define RAG", suggested_tool_types=["rag_search"]
        )

        assert result is not None
        tool_type, tool_id = result
        assert tool_type == "rag_search"
        assert tool_id == rag_tool_id_placeholder
        mock_genie_for_dra.prompts.render_chat_prompt.assert_awaited_with(
            name=TOOL_SELECTOR_PROMPT_ID, data=ANY
        )

    async def test_execute_tool_and_process_rag_search_success(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra: MagicMock
    ):
        mock_genie_for_dra.rag.search.return_value = [
            MagicMock(
                content="RAG result content",
                metadata={"source_uri": "doc1.txt"},
                id="rag_chunk1",
            )
        ]
        expected_extraction = ExtractionOutputModel(
            relevant_summary="Extracted RAG data", source_identifier="doc1.txt"
        )
        mock_genie_for_dra.llm.parse_output.return_value = expected_extraction

        snippet = await deep_research_agent._execute_tool_and_process(
            tool_type="rag_search",
            tool_id="internal_rag_search_tool", # This ID is used conceptually for RAG
            sub_query="what is RAG from my docs",
        )

        assert snippet is not None
        assert snippet["extracted_info"] == "Extracted RAG data"
        assert snippet["source_identifier"] == "doc1.txt"
        mock_genie_for_dra.rag.search.assert_awaited_once()
        mock_genie_for_dra.prompts.render_chat_prompt.assert_awaited_with(
            name=RAG_EXTRACTOR_PROMPT_ID, data=ANY
        )

    async def test_decide_next_action_proceed_to_synthesis(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra: MagicMock
    ):
        expected_decision = ControllerOutputModel(
            decision="proceed_to_synthesis", reasoning="Enough info"
        )
        mock_genie_for_dra.llm.parse_output.return_value = expected_decision

        decision, _ = await deep_research_agent._decide_next_action(
            "goal",
            [ResearchTaskModel(sub_query="Q1", suggested_tool_types=["web_search"], status="completed")],
            [{"source_type":"web", "source_identifier":"s1", "sub_query":"Q1", "extracted_info":"info1", "key_points":[]}],
            "Quick Overview",
            {"gathering": 1, "refinement": 0},
        )

        assert decision == "proceed_to_synthesis"
        mock_genie_for_dra.prompts.render_chat_prompt.assert_awaited_with(
            name=CONTROLLER_PROMPT_ID, data=ANY
        )

    async def test_synthesize_report_final(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra: MagicMock
    ):
        mock_genie_for_dra.llm.chat.return_value = LLMChatResponse(
            message=ChatMessage(role="assistant", content="Final report content."),
            raw_response={},
            usage={"total_tokens": 5},
        )
        snippets: List[ProcessedSnippet] = [
            {
                "source_type": "web",
                "source_identifier": "url1",
                "sub_query": "q1",
                "extracted_info": "info1",
                "key_points": ["kp1"],
            }
        ]
        report = await deep_research_agent._synthesize_report(
            "goal", snippets, is_final=True
        )

        assert report == "Final report content."
        mock_genie_for_dra.prompts.render_chat_prompt.assert_awaited_with(
            name=FINAL_SYNTHESIZER_PROMPT_ID, data=ANY
        )


@pytest.mark.asyncio
class TestDeepResearchAgentRun:
    async def test_run_complete_flow_success(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra: MagicMock
    ):
        # --- Configure Agent's Tool IDs ---
        rag_tool_placeholder_id = "internal_rag_search_tool"
        web_search_tool_id = "google_search_tool_v1"
        deep_research_agent.tool_configs["rag_search"]["tool_id"] = rag_tool_placeholder_id
        deep_research_agent.tool_configs["web_search"]["tool_id"] = web_search_tool_id


        # --- Mock Planner Output ---
        plan_tasks = [
            ResearchTaskModel(
                sub_query="Define RAG", suggested_tool_types=["rag_search"]
            ),
            ResearchTaskModel(
                sub_query="Latest RAG advancements",
                suggested_tool_types=["web_search"], # Simplified to one type for easier mocking
            ),
        ]
        planner_output = PlannerOutputModel(
            research_tasks=plan_tasks,
            estimated_depth="Standard Investigation",
            overall_reasoning="Comprehensive understanding of RAG.",
        )

        # --- Mock Tool Selector Outputs ---
        tool_selection_rag = ToolSelectionOutputModel(
            selected_tool_id=rag_tool_placeholder_id, reasoning="RAG for definition"
        )
        tool_selection_web = ToolSelectionOutputModel(
            selected_tool_id=web_search_tool_id, reasoning="Web for recent news"
        )

        # --- Mock Tool Formatted Definitions ---
        def get_formatted_def_side_effect(tool_id, formatter_id):
            if tool_id == rag_tool_placeholder_id:
                return "Formatted RAG Tool Definition"
            if tool_id == web_search_tool_id:
                return "Formatted Web Search Tool Definition"
            return "Unknown tool formatted def"
        mock_genie_for_dra._tool_manager.get_formatted_tool_definition.side_effect = get_formatted_def_side_effect


        # --- Mock RAG Search & Web Execute Tool ---
        mock_genie_for_dra.rag.search.return_value = [
            MagicMock(
                content="RAG is Retrieval Augmented Generation.",
                metadata={"source_uri": "doc_rag.txt"},
                id="rag1",
            )
        ]
        mock_genie_for_dra.execute_tool.return_value = {
            "results": [
                {"link": "http://news.com/rag_adv", "snippet": "New RAG paper out!"}
            ]
        }

        # --- Mock Extractor Outputs ---
        extraction_rag = ExtractionOutputModel(
            relevant_summary="RAG means Retrieval Augmented Generation.",
            source_identifier="doc_rag.txt",
        )
        extraction_web = ExtractionOutputModel(
            relevant_summary="A new paper on RAG was published.",
            source_identifier="http://news.com/rag_adv",
        )

        # --- Mock Controller Output ---
        controller_decision_synth = ControllerOutputModel(
            decision="proceed_to_synthesis", reasoning="Gathered enough info."
        )

        # --- Mock Final Synthesizer Output ---
        final_report_text = "Final comprehensive report on RAG."
        final_synth_response = LLMChatResponse(
            message=ChatMessage(role="assistant", content=final_report_text),
            raw_response={},
            usage={"total_tokens": 100},
        )

        # Configure side effects for genie.llm.parse_output
        # This needs to be stateful or more specific based on the call context
        # For simplicity, we'll use a list of return values.
        parse_output_call_sequence = [
            planner_output,         # 1. For _generate_initial_plan
            tool_selection_rag,     # 2. For _select_tool_for_sub_query (Task 1: RAG)
            extraction_rag,         # 3. For _execute_tool_and_process (Task 1: RAG)
            tool_selection_web,     # 4. For _select_tool_for_sub_query (Task 2: Web)
            extraction_web,         # 5. For _execute_tool_and_process (Task 2: Web)
            controller_decision_synth # 6. For _decide_next_action
        ]
        mock_genie_for_dra.llm.parse_output.side_effect = parse_output_call_sequence
        mock_genie_for_dra.llm.chat.return_value = final_synth_response # For final synthesis

        # --- Run the agent ---
        result = await deep_research_agent.run(goal="Deep dive into RAG")

        # --- Assertions ---
        assert result["status"] == "success"
        assert result["output"] == final_report_text
        assert len(result["history"]) >= 7 # Adjusted expectation
        assert result["plan"] is not None
        assert len(result["plan"]) == 2
        assert result["plan"][0]["status"] == "completed"
        assert result["plan"][1]["status"] == "completed"

        mock_genie_for_dra.rag.search.assert_awaited_once()
        mock_genie_for_dra.execute_tool.assert_awaited_once()


    async def test_run_initial_plan_fails(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra: MagicMock
    ):
        mock_genie_for_dra.llm.parse_output.return_value = None # Planner fails

        result = await deep_research_agent.run(goal="Test plan failure")

        assert result["status"] == "error"
        assert "Failed to generate initial research plan." in result["output"]
        mock_genie_for_dra.execute_tool.assert_not_called()
        mock_genie_for_dra.rag.search.assert_not_called()