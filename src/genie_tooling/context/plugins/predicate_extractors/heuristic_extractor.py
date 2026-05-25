import logging
from typing import TYPE_CHECKING

from genie_tooling.context.protocols import PredicateExtractorPlugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


class HeuristicPredicateExtractorPlugin(PredicateExtractorPlugin):
    """A simple, fast predicate extractor based on keywords."""

    plugin_id: str = "heuristic_predicate_extractor_v1"
    description: str = "Extracts predicates using simple keyword matching."

    async def extract(self, query: str, genie: "Genie") -> str:
        query_lower = query.lower()
        words = query_lower.split()
        for word in words:
            if word in [
                "is", "are", "what", "who", "where", "when", "can", "do",
                "does", "compare", "evaluate", "summarize", "find", "lookup", "calculate",
            ]:
                return f"predicate_{word}"
        return "predicate_generic_inquiry"
