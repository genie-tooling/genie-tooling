###src/genie_tooling/agents/math_proof_assistant_agent.py###
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from genie_tooling.agents.base_agent import BaseAgent
from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

# --- Pydantic Models (remain the same) ---
class IntentResponse(BaseModel):
    intent: Literal["research_concept", "test_hypothesis", "state_insight", "continue_proof", "unsupported"] = Field(
        description="The classified intent of the user's input."
    )
    query: Optional[str] = Field(None, description="A refined, self-contained query for the 'research_concept' intent.")

class DecompositionPlan(BaseModel):
    sub_goals: List[str] = Field(description="A list of logical sub-goals or research questions to tackle the main goal.")

class HypothesisTestPlan(BaseModel):
    thought: str = Field(description="Reasoning for choosing the tool and parameters.")
    tool_id: str = Field(description="The ID of the tool to use ('symbolic_math_tool' or 'generic_code_execution_tool').")
    params: Dict[str, Any] = Field(description="A valid JSON object of parameters for the chosen tool.")

# --- The Agent Class (with corrected run method) ---
class MathProofAssistantAgent(BaseAgent):
    def __init__(self, genie: Genie, agent_config: Optional[Dict[str, Any]] = None):
        super().__init__(genie, agent_config)
        self.state: str = "AWAITING_GOAL"
        self.main_goal: str = ""
        self.project_name: str = ""
        self.session_id: str = ""
        self.sub_goals: List[str] = []
        self.current_sub_goal_index: int = 0
        self.research_processor_id = self.agent_config.get("research_processor_id", "deep_research_agent_v1")

    async def _call_llm_for_json(self, prompt: str, schema: Any, model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            response = await self.genie.llm.chat([{"role": "user", "content": prompt}], output_schema=schema, provider_id=model_id)
            parsed_output = await self.genie.llm.parse_output(response, schema=schema, parser_id="pydantic_output_parser_v1")
            if isinstance(parsed_output, BaseModel):
                return parsed_output.model_dump()
            elif isinstance(parsed_output, dict):
                return parsed_output
            logger.warning(f"LLM parser returned unexpected type: {type(parsed_output)}")
            return None
        except Exception as e:
            logger.error(f"LLM call for structured JSON failed: {e}", exc_info=True)
            return None

    async def determine_intent(self, user_input: str) -> IntentResponse:
        # --- FIX: Properly await and handle potential None ---
        conversation_state = await self.genie.conversation.load_state(self.session_id)
        history_str = json.dumps(conversation_state['history'][-5:], indent=2) if conversation_state else "[]"
        
        prompt = f"""
        Given the conversation history and the latest user input, classify the user's primary intent.
        - If the user is asking a question that requires external knowledge (e.g., "what is X?", "explain Y"), classify as 'research_concept' and formulate a clear, self-contained query.
        - If the user wants to perform a calculation or check an equation, classify as 'test_hypothesis'.
        - If the user is stating a realization, idea, or connection, classify as 'state_insight'.
        - If the user wants to proceed with the main plan, classify as 'continue_proof'.
        - Otherwise, classify as 'unsupported'.

        Conversation History (last 5 turns):
        {history_str}

        Latest User Input: "{user_input}"

        Respond with a single JSON object matching the required schema.
        """
        response_dict = await self._call_llm_for_json(prompt, IntentResponse)
        if response_dict:
            return IntentResponse(**response_dict)
        return IntentResponse(intent="unsupported")

    async def run(self, initial_goal: str) -> None:
        """Main conversational loop for the agent."""
        self.main_goal = initial_goal
        self.project_name = "".join(filter(str.isalnum, self.main_goal.lower().replace(" ", "_")))[:50]
        self.session_id = f"math_proof_{self.project_name}"
        
        print(f"\n🤖 Welcome! I'm your Math Proof Assistant. Let's tackle: '{self.main_goal}'")
        print(f"🧠 Project memory will be stored under collection: '{self.project_name}'\n")
        
        self.state = "DECOMPOSING_GOAL"
        decomp_prompt = f"Decompose the complex goal '{self.main_goal}' into a short, logical list of sub-goals or research questions to guide our exploration. Your response MUST be a JSON object that strictly adheres to the provided schema."
        decomp_result = await self._call_llm_for_json(decomp_prompt, DecompositionPlan)
        self.sub_goals = decomp_result.get("sub_goals", []) if decomp_result else []

        if not self.sub_goals:
            print("🤖 I couldn't break down the initial goal. Could you please rephrase or be more specific?")
            self.state = "FINISHED"
        else:
            self.state = "WORKING_ON_PROOF"
            print("🤖 I've broken the goal into these steps:")
            for i, goal in enumerate(self.sub_goals):
                print(f"   {i+1}. {goal}")
            print(f"\n🤖 Let's start with step 1: '{self.sub_goals[0]}'")

        while self.state != "FINISHED":
            user_input = await asyncio.to_thread(input, f"\n[{self.state}] > ")
            await self.genie.conversation.add_message(self.session_id, {"role": "user", "content": user_input})
            
            if user_input.lower() in ["quit", "exit", "stop"]:
                self.state = "FINISHED"
                continue

            intent_response = await self.determine_intent(user_input)
            
            if intent_response.intent == "research_concept":
                query = intent_response.query or user_input
                await self._handle_research(query)
            elif intent_response.intent == "test_hypothesis":
                await self._handle_hypothesis(user_input)
            elif intent_response.intent == "state_insight":
                await self._handle_insight(user_input)
            elif intent_response.intent == "continue_proof":
                await self._handle_continue_proof()
            else:
                # --- THIS IS THE FIX ---
                # 1. Await the conversation state and store it.
                conversation_state = await self.genie.conversation.load_state(self.session_id)
                # 2. Check if the state and history exist before using them.
                if conversation_state and conversation_state.get('history'):
                    # 3. Pass the actual history list to the chat method.
                    response = await self.genie.llm.chat(conversation_state['history'])
                    print(f"🤖 {response['message']['content']}")
                    await self.genie.conversation.add_message(self.session_id, response['message'])
                else:
                    print("🤖 I'm sorry, I seem to have lost the context of our conversation. Could you repeat that?")

        print("\n🤖 Session finished. All our work has been saved. Goodbye!")

    async def _handle_research(self, query: str):
        print(f"🤖 Conducting deep research on: '{query}'. This may take a moment...")
        research_result = await self.genie.run_command(command=query, processor_id=self.research_processor_id)
        
        final_answer = research_result.get('final_answer', 'Research did not produce a final answer.')
        print(f"\n🤖 Research Summary:\n{final_answer}")
        
        try:
            await self.genie.rag.index_text( # type: ignore
                text=f"Research on '{query}':\n{final_answer}",
                collection_name=self.project_name,
                metadata={"type": "research_summary", "query": query, "timestamp": time.time()}
            )
            print("\n💾 I've saved this summary to our project memory.")
        except Exception as e:
            logger.error(f"Failed to save research summary to RAG memory: {e}")
        self.state = "WORKING_ON_PROOF"

    async def _handle_hypothesis(self, hypothesis: str):
        print(f"🤖 Let's test that hypothesis. I'll formulate a tool call...")
        tool_defs_list = [
            await self.genie._tool_manager.get_formatted_tool_definition("symbolic_math_tool", "compact_text_formatter_plugin_v1"),
            await self.genie._tool_manager.get_formatted_tool_definition("generic_code_execution_tool", "compact_text_formatter_plugin_v1")
        ]
        tools_str = "\n".join(filter(None, tool_defs_list))
        
        prompt = f"""
        User hypothesis: "{hypothesis}"
        Available tools for testing:
        ---
        {tools_str}
        ---
        Formulate a plan to test this hypothesis with a SINGLE tool call. Your response MUST be a JSON object adhering to the schema.
        """
        plan_dict = await self._call_llm_for_json(prompt, HypothesisTestPlan)
        if not plan_dict:
            print("🤖 I couldn't figure out how to test that hypothesis with my available tools.")
            return

        try:
            plan = HypothesisTestPlan(**plan_dict)
            print(f"🤖 My plan: {plan.thought}")
            print(f"🤖 Executing tool '{plan.tool_id}' with params: {json.dumps(plan.params)}")
            
            result = await self.genie.execute_tool(plan.tool_id, **plan.params)
            print(f"🤖 Result:\n{json.dumps(result, indent=2)}")

            await self.genie.rag.index_text( # type: ignore
                text=f"Hypothesis test: '{hypothesis}'\nTool: {plan.tool_id}\nParams: {plan.params}\nResult: {json.dumps(result)}",
                collection_name=self.project_name,
                metadata={"type": "hypothesis_test", "timestamp": time.time()}
            )
            print("\n💾 I've saved the result of this test to our project memory.")
        except Exception as e:
            print(f"🔥 An error occurred while testing the hypothesis: {e}")
        self.state = "WORKING_ON_PROOF"

    async def _handle_insight(self, insight: str):
        print("🤖 That's a great insight! I'm saving it to our project memory.")
        await self.genie.rag.index_text( # type: ignore
            text=f"User insight: {insight}",
            collection_name=self.project_name,
            metadata={"type": "user_hypothesis", "timestamp": time.time()}
        )
        self.state = "WORKING_ON_PROOF"

    async def _handle_continue_proof(self):
        self.current_sub_goal_index += 1
        if self.current_sub_goal_index < len(self.sub_goals):
            next_goal = self.sub_goals[self.current_sub_goal_index]
            print(f"🤖 Alright, let's move on. The next step ({self.current_sub_goal_index + 1}/{len(self.sub_goals)}) is: '{next_goal}'")
        else:
            print("🤖 We've completed all the initial sub-goals! We can now work on synthesizing a final proof or explore new avenues.")
            await self._synthesize_final_report()
            self.state = "FINISHED"

    async def _synthesize_final_report(self):
        print("🤖 Synthesizing a final report based on our conversation and findings...")
        all_memories = await self.genie.rag.search(query=self.main_goal, collection_name=self.project_name, top_k=50)
        
        if not all_memories:
            print("🤖 There's nothing in our project memory to synthesize a report from.")
            return

        memory_context = "\n\n---\n\n".join(
            f"Source (Type: {mem.metadata.get('type', 'N/A')}, Time: {time.ctime(mem.metadata.get('timestamp', 0))})\n{mem.content}"
            for mem in all_memories
        )

        synthesis_prompt = f"""
        Original Goal: {self.main_goal}
        
        Our Decomposed Plan Was:
        {json.dumps(self.sub_goals, indent=2)}

        Collected Research & Insights:
        ---
        {memory_context}
        ---

        Based on all the information gathered, synthesize a comprehensive report or proof sketch that addresses the original goal. Structure your answer logically.
        """
        response = await self.genie.llm.chat([{"role": "user", "content": synthesis_prompt}])
        print("\n\n--- FINAL SYNTHESIZED REPORT ---")
        print(response['message']['content'])
        print("--- END OF REPORT ---")
