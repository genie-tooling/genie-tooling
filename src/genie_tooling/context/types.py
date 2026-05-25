from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict


# A TypedDict representation of the Rule Object from the paper
class RuleObject(TypedDict):
    rule_id: str
    predicate: str
    priority: int
    conditions: List[Tuple[str, str, Any]]
    actions: List[Tuple[Literal["C_D", "C_F"], str, str, Any]]
    description: Optional[str]  # For semantic matching


# A TypedDict for the output of the inference plugin
class InferredContext(TypedDict, total=False):
    AudienceProfile: Dict[str, Any]
    DiscourseTopic: Dict[str, float]
    # Other inferred properties can be added here
