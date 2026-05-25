from typing import Any, Dict, Optional

from .audit import DecisionRecord
from .manager import ContextManager


class ContextInterface:
    """
    The public, developer-facing interface for the CQS-Engine.

    This interface is attached to the `genie` instance as `genie.context`,
    providing a simple entry point to the context-scoping pipeline.
    """

    def __init__(self, manager: ContextManager):
        self._manager = manager

    async def resolve_and_formulate(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_identity: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Processes a query through the full context-scoping pipeline.

        Args:
            query: The natural language query or stimulus.
            session_id: The session identifier to retrieve conversation history
                        and user profile information.
            user_identity: Optional dict of caller-supplied identity (SSO uid,
                        role, tenant id, etc.) attached verbatim to the
                        DecisionRecord for audit. The framework doesn't
                        interpret it.

        Returns:
            The final, articulated response from the engine.
        """
        return await self._manager.resolve_and_formulate(
            query=query, session_id=session_id, user_identity=user_identity
        )

    @property
    def last_decision(self) -> Optional[DecisionRecord]:
        """The DecisionRecord assembled by the most recent
        resolve_and_formulate call.

        Convenience accessor — production audit consumers should subscribe
        to the ``audit.decision_record`` trace event instead so they don't
        miss records when calls overlap.
        """
        return self._manager.last_decision

    async def reload_rules(self) -> bool:
        """Reload the rule set from disk without restarting the process.

        Governance/compliance teams update YAML rules in version control;
        this lets the engine pick those up on demand (e.g. via an admin
        endpoint or a file-watcher plugin).

        Returns True if the swap succeeded, False if reload failed and
        the previous rules remain active. A reload failure most commonly
        comes from the A2 schema validator rejecting a freshly-edited
        rule — fix the file and call reload again.
        """
        engine = self._manager._rule_engine
        if engine is None or not hasattr(engine, "reload_rules"):
            return False
        return await engine.reload_rules(genie=self._manager._genie)
