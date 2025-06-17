# src/genie_tooling/agents/types.py
"""Type definitions specific to Agentic Loops."""
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField


class PlannedStep(TypedDict):
    """Represents a single step in a generated plan."""
    step_number: int
    tool_id: str
    params: Dict[str, Any]
    reasoning: Optional[str]
    output_variable_name: Optional[str]


class ReActObservation(TypedDict):
    """Represents one cycle of Thought-Action-Observation in ReAct."""
    thought: str
    action: str
    observation: str


class AgentOutput(TypedDict):
    """Standardized output structure from an agent's run method."""
    status: Literal["success", "error", "max_iterations_reached", "user_stopped"]
    output: Any
    history: Optional[List[Any]]
    plan: Optional[List[Any]]


# --- Types for PlanAndExecuteAgent ---
class PlanStepModelPydantic(PydanticBaseModel):
    """Pydantic model for a single step in a plan for PlanAndExecuteAgent."""
    step_number: int = PydanticField(description="Sequential number of the step.")
    tool_id: str = PydanticField(description="The ID of the tool to use for this step.")
    params: Any = PydanticField(description="A valid JSON object of parameters for the tool, which may contain placeholders.")
    reasoning: Optional[str] = PydanticField(None, description="Reasoning for this step.")
    output_variable_name: Optional[str] = PydanticField(None, description="If this step's output should be stored for later use, provide a variable name here (e.g., 'search_results').")


class PlanModelPydantic(PydanticBaseModel):
    """Pydantic model for the overall plan for PlanAndExecuteAgent."""
    plan: List[PlanStepModelPydantic] = PydanticField(description="The list of steps to execute.")
    overall_reasoning: Optional[str] = PydanticField(None, description="Overall reasoning for the plan.")


# --- Types for DeepResearchAgent ---
class ExecutionEvidence(TypedDict, total=False):
    """Structures the evidence gathered from a single tool execution step."""
    step_number: int
    sub_question: str
    action: Dict[str, Any]  # A dump of the TacticalPlanStep model
    outcome: Any
    error: Optional[str]
    quality: Literal["high", "low", "unevaluated"]
    source_url: Optional[str]


class InitialResearchPlan(PydanticBaseModel):
    """Pydantic model for the initial high-level research plan."""
    sub_questions: List[str] = PydanticField(description="A list of sub-questions to research to answer the user's main goal.")


class TacticalPlanStep(PydanticBaseModel):
    """Pydantic model for a single step in a tactical plan."""
    thought: str = PydanticField(description="The reasoning for why this specific tool call is necessary.")
    tool_id: str = PydanticField(description="The identifier of the tool to execute.")
    params: Dict[str, Any] = PydanticField(description="A valid JSON object of parameters for the tool, which may contain placeholders.")
    output_variable_name: Optional[str] = PydanticField(None, description="If this step's output should be stored for later use, provide a variable name here (e.g., 'search_results').", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class TacticalPlan(PydanticBaseModel):
    """Pydantic model for a tactical plan to answer a sub-question."""
    plan: List[TacticalPlanStep] = PydanticField(description="The sequence of tool calls to execute to answer a sub-question.")
