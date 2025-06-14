# src/genie_tooling/agents/types.py
"""Type definitions specific to Agentic Loops."""
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel as PydanticBaseModel  # For Pydantic models
from pydantic import Field as PydanticField  # For Pydantic models


class PlannedStep(TypedDict):
    """Represents a single step in a generated plan."""
    step_number: int
    tool_id: str
    params: Dict[str, Any]
    reasoning: Optional[str] # LLM's reasoning for this step
    output_variable_name: Optional[str] # Name to store this step's output under

class ReActObservation(TypedDict):
    """Represents one cycle of Thought-Action-Observation in ReAct."""
    thought: str
    action: str # e.g., "ToolName[params_json]" or "Answer" or "Error"
    observation: str # Result of action or error message

class AgentOutput(TypedDict):
    """Standardized output structure from an agent's run method."""
    status: Literal["success", "error", "max_iterations_reached", "user_stopped"]
    output: Any # The final result or error message
    history: Optional[List[Any]] # e.g., ReAct scratchpad, or list of executed plan steps with their results
    plan: Optional[List[PlannedStep]] # The plan that was executed (for PlanAndExecute)
    # Add other common fields like 'cost', 'tokens_used' if agents track this directly


# Pydantic models for PlanAndExecuteAgent's internal plan structure
class PlanStepModelPydantic(PydanticBaseModel):
    """Pydantic model for a single step in a plan."""
    step_number: int = PydanticField(description="Sequential number of the step.")
    tool_id: str = PydanticField(description="The ID of the tool to use for this step.")
    # FIX: ReWOO instructs the LLM to return a JSON string, so the model should expect a string.
    params: str = PydanticField(default="{}", description="A JSON-encoded STRING containing a dictionary of parameters for the tool.")
    reasoning: Optional[str] = PydanticField(None, description="Reasoning for this step.")
    output_variable_name: Optional[str] = PydanticField(None, description="If this step's output should be stored for later use, provide a variable name here (e.g., 'search_results').")

class PlanModelPydantic(PydanticBaseModel):
    """Pydantic model for the overall plan."""
    plan: List[PlanStepModelPydantic] = PydanticField(description="The list of steps to execute.")
    overall_reasoning: Optional[str] = PydanticField(None, description="Overall reasoning for the plan.")
