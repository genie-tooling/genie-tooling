###src/genie_tooling/agents/__init__.py###
# src/genie_tooling/agents/__init__.py
"""
Agentic loop implementations (ReAct, Plan-and-Execute, etc.)
that leverage the Genie facade for their operations.
"""
from .base_agent import BaseAgent
from .deep_research_agent import DeepResearchAgent
from .math_proof_assistant_agent import MathProofAssistantAgent  # ADDED
from .plan_and_execute_agent import PlanAndExecuteAgent
from .react_agent import ReActAgent
from .types import (
    AgentOutput,
    ExecutionEvidence,
    InitialResearchPlan,
    PlannedStep,
    ReActObservation,
    TacticalPlan,
)

__all__ = [
    "BaseAgent",
    "ReActAgent",
    "PlanAndExecuteAgent",
    "DeepResearchAgent",
    "MathProofAssistantAgent",
    "AgentOutput",
    "PlannedStep",
    "ReActObservation",
    "ExecutionEvidence",
    "InitialResearchPlan",
    "TacticalPlan",
]
