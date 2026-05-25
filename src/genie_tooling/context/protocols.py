from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Protocol, Tuple, runtime_checkable

from genie_tooling.core.types import Plugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie


@runtime_checkable
class ContextSourcePlugin(Plugin, Protocol):
    """
    Protocol for a plugin that provides application-specific user/world profile data.
    """

    async def get_profile(self, session_id: str, genie: "Genie") -> Dict[str, Any]:
        """
        Fetches the User/World Profile ('U' in λ-CQS).

        Args:
            session_id: The session identifier for the current interaction.
            genie: The main Genie facade instance, for accessing other components if needed.

        Returns:
            A dictionary of key-value pairs representing the profile.
        """
        ...


@runtime_checkable
class ContextInferencePlugin(Plugin, Protocol):
    """
    Protocol for a plugin that infers high-level properties from raw context.
    """

    async def infer_context_properties(
        self, raw_context: Dict[str, Any], genie: "Genie"
    ) -> Dict[str, Any]:
        """
        Takes raw context (history and profile) and infers structured properties.

        Args:
            raw_context: A dictionary containing 'history' and 'profile'.
            genie: The main Genie facade instance.

        Returns:
            A structured dictionary of inferred properties (e.g., AudienceProfile, DiscourseTopic).
        """
        ...


@runtime_checkable
class RuleEnginePlugin(Plugin, Protocol):
    """
    Protocol for a plugin that evaluates rules against context to generate constraints.

    Both `load_rules` and `evaluate` accept the Genie facade so engines that need
    framework subsystems (LLMs, RAG, vector stores) can use them. Engines that
    don't need it should accept and ignore the argument.
    """

    async def load_rules(self, genie: "Genie") -> bool:
        """
        Loads and prepares rules from the configured backend.
        This is called once during setup.

        Args:
            genie: The main Genie facade instance, available for engines that
                need to index rules into a vector store, call an LLM, etc.
        """
        ...

    async def evaluate(
        self,
        inferred_context: Dict[str, Any],
        query_predicate: str,
        genie: "Genie",
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Finds and ranks relevant rules based on the context and query.

        Args:
            inferred_context: The structured dictionary of inferred properties.
            query_predicate: A string representing the logical intent of the query.
            genie: The main Genie facade instance, available for engines that
                need to call subsystems like RAG. Deterministic engines should
                accept and ignore it.

        Returns:
            A list of (Rule Object, score) tuples, sorted by relevance score descending.
        """
        ...

@runtime_checkable
class PredicateExtractorPlugin(Plugin, Protocol):
    """Protocol for a plugin that extracts a logical predicate from a query."""
    async def extract(self, query: str, genie: "Genie") -> str:
        """Extracts a predicate string from the user's query."""
        ...

@runtime_checkable
class DerivationStrategyPlugin(Plugin, Protocol):
    """Protocol for a plugin that executes a derivation strategy."""
    async def derive(
        self, query: str, constraints: Dict[str, Any], genie: "Genie"
    ) -> Dict[str, Any]:
        """
        Takes a query and derivation constraints (C_D) to produce raw data.
        Returns a dictionary, typically with 'status' and 'result' or 'error'.
        """
        ...

@runtime_checkable
class FormulationStrategyPlugin(Plugin, Protocol):
    """Protocol for a plugin that executes a formulation strategy."""
    async def formulate(
        self, query: str, raw_data: Any, constraints: Dict[str, Any], genie: "Genie"
    ) -> str:
        """
        Takes raw data and formulation constraints (C_F) to produce a final response.
        Returns the final user-facing string.
        """
        ...
