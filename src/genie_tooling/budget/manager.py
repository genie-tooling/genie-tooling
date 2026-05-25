"""BudgetManager: wraps a BudgetEnforcerPlugin and exposes a clean API.

This is the thin orchestrator the rest of the framework talks to. It
loads the configured enforcer plugin lazily and provides the no-op
fall-through when budget enforcement isn't configured.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from genie_tooling.core.plugin_manager import PluginManager

from .abc import BudgetEnforcerPlugin
from .types import BudgetSnapshot, BudgetSpec

logger = logging.getLogger(__name__)


class BudgetManager:
    def __init__(
        self,
        plugin_manager: PluginManager,
        default_enforcer_id: Optional[str] = None,
        enforcer_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._plugin_manager = plugin_manager
        self._enforcer_id = default_enforcer_id
        self._configs = enforcer_configurations or {}
        self._enforcer: Optional[BudgetEnforcerPlugin] = None
        self._initialized = False

    @property
    def is_active(self) -> bool:
        return bool(self._enforcer_id and self._enforcer_id.lower() != "none")

    async def _ensure_loaded(self) -> Optional[BudgetEnforcerPlugin]:
        if self._initialized:
            return self._enforcer
        if not self.is_active:
            self._initialized = True
            return None
        config = self._configs.get(self._enforcer_id, {})
        try:
            inst = await self._plugin_manager.get_plugin_instance(self._enforcer_id, config=config)
        except Exception as e:
            logger.error(f"BudgetManager: failed to load enforcer '{self._enforcer_id}': {e}", exc_info=True)
            inst = None
        if inst and isinstance(inst, BudgetEnforcerPlugin):
            self._enforcer = inst
        elif inst:
            logger.warning(f"BudgetManager: plugin '{self._enforcer_id}' is not a BudgetEnforcerPlugin.")
        self._initialized = True
        return self._enforcer

    async def set_budget(self, scope: str, spec: BudgetSpec) -> None:
        enforcer = await self._ensure_loaded()
        if enforcer:
            await enforcer.set_budget(scope, spec)

    async def check_and_charge_llm_call(
        self,
        scope: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        provider_id: str,
        cost_usd: float = 0.0,
    ) -> None:
        if not scope:
            return
        enforcer = await self._ensure_loaded()
        if enforcer:
            await enforcer.check_and_charge_llm_call(scope, prompt_tokens, completion_tokens, provider_id, cost_usd)

    async def check_and_charge_tool_call(self, scope: Optional[str]) -> None:
        if not scope:
            return
        enforcer = await self._ensure_loaded()
        if enforcer:
            await enforcer.check_and_charge_tool_call(scope)

    async def get_usage(self, scope: str) -> Optional[BudgetSnapshot]:
        enforcer = await self._ensure_loaded()
        if not enforcer:
            return None
        return await enforcer.get_usage(scope)

    async def clear(self, scope: str) -> None:
        enforcer = await self._ensure_loaded()
        if enforcer:
            await enforcer.clear(scope)

    async def teardown(self) -> None:
        if self._enforcer:
            try:
                await self._enforcer.teardown()
            except Exception as e:
                logger.error(f"BudgetManager teardown error: {e}", exc_info=True)
        self._enforcer = None
        self._initialized = False
