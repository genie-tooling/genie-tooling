"""HITLManager: Orchestrates HumanApprovalRequestPlugins.

Supports two configurations:
  * Single default approver: ``default_approver_id="webhook_approval_v1"``
  * Approver **chain** (Phase 6A.5b): ``default_approver_chain=["claude_code_permissions_v1", "webhook_approval_v1"]``
    — each request walks the chain; a response of ``status="ask_human"``
    delegates to the next approver. First non-``ask_human`` response wins.
    A chain is the recommended setup for production: a deterministic
    permissions plugin in front, with a human approver as the fallback.
"""
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager

from .abc import HumanApprovalRequestPlugin
from .ledger.abc import HITLLedgerPlugin
from .ledger.types import LedgerEntry
from .types import ApprovalRequest, ApprovalResponse

logger = logging.getLogger(__name__)

class HITLManager:
    def __init__(
        self,
        plugin_manager: PluginManager,
        default_approver_id: Optional[str] = None,
        approver_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
        default_approver_chain: Optional[List[str]] = None,
        ledger_id: Optional[str] = None,
        ledger_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._plugin_manager = plugin_manager
        self._default_approver_id = default_approver_id
        self._approver_configurations = approver_configurations or {}
        self._default_approver_instance: Optional[HumanApprovalRequestPlugin] = None
        self._initialized_default = False
        # Phase 6A.7: optional durable approval ledger.
        self._ledger_id = ledger_id
        self._ledger_configurations = ledger_configurations or {}
        self._ledger: Optional[HITLLedgerPlugin] = None
        self._ledger_initialized = False
        # Phase 6A.5b: explicit chain takes precedence; if not provided, derive
        # a single-element chain from default_approver_id for back-compat.
        if default_approver_chain:
            self._default_chain_ids: List[str] = list(default_approver_chain)
        elif default_approver_id and default_approver_id.lower() != "none":
            self._default_chain_ids = [default_approver_id]
        else:
            self._default_chain_ids = []
        self._chain_instance_cache: Dict[str, HumanApprovalRequestPlugin] = {}
        logger.info(
            f"HITLManager initialized. chain={self._default_chain_ids or 'none'} "
            f"(default_approver_id={default_approver_id!r})"
        )

    @property
    def is_active(self) -> bool:
        """Returns True if at least one approver is configured."""
        if self._default_chain_ids:
            return True
        return bool(self._default_approver_id and self._default_approver_id.lower() != "none")

    async def _get_default_approver(self) -> Optional[HumanApprovalRequestPlugin]:
        if not self._initialized_default:
            if self.is_active:
                config = self._approver_configurations.get(self._default_approver_id, {}) # type: ignore
                try:
                    instance_any = await self._plugin_manager.get_plugin_instance(self._default_approver_id, config=config) # type: ignore
                    if instance_any and isinstance(instance_any, HumanApprovalRequestPlugin):
                        self._default_approver_instance = cast(HumanApprovalRequestPlugin, instance_any)
                        logger.info(f"Default HITL approver '{self._default_approver_id}' loaded.")
                    elif instance_any:
                        logger.warning(f"Default HITL approver '{self._default_approver_id}' loaded but is not a valid HumanApprovalRequestPlugin.")
                        self._default_approver_instance = None
                    else:
                        logger.warning(f"Default HITL approver '{self._default_approver_id}' not found or failed to load.")
                        self._default_approver_instance = None
                except Exception as e:
                    logger.error(f"Error loading default HITL approver '{self._default_approver_id}': {e}", exc_info=True)
                    self._default_approver_instance = None
            else:
                logger.info("No default HITL approver configured.")
            self._initialized_default = True
        return self._default_approver_instance

    async def _get_ledger(self) -> Optional[HITLLedgerPlugin]:
        if self._ledger_initialized:
            return self._ledger
        self._ledger_initialized = True
        if not self._ledger_id:
            return None
        try:
            config = self._ledger_configurations.get(self._ledger_id, {})
            inst = await self._plugin_manager.get_plugin_instance(self._ledger_id, config=config)
            if inst and isinstance(inst, HITLLedgerPlugin):
                self._ledger = cast(HITLLedgerPlugin, inst)
                logger.info(f"HITL ledger '{self._ledger_id}' loaded.")
            elif inst:
                logger.warning(f"HITL ledger '{self._ledger_id}' loaded but is not a valid HITLLedgerPlugin.")
        except Exception as e:
            logger.error(f"Error loading HITL ledger '{self._ledger_id}': {e}", exc_info=True)
        return self._ledger

    async def _record_to_ledger(
        self, request: ApprovalRequest, response: ApprovalResponse, requested_at: float
    ) -> None:
        ledger = await self._get_ledger()
        if not ledger:
            return
        data = request.get("data_to_approve") or {}
        ctx = request.get("context") or {}
        entry = LedgerEntry(
            request_id=response.get("request_id") or request.get("request_id", ""),
            decision_id=ctx.get("decision_id") if isinstance(ctx, dict) else None,
            correlation_id=ctx.get("correlation_id") if isinstance(ctx, dict) else None,
            tool_id=data.get("tool_id"),
            params=data.get("params"),
            tool_metadata=data.get("tool_metadata"),
            status=response.get("status", "unknown"),
            approver_id=response.get("approver_id"),
            reason=response.get("reason"),
            requested_at=requested_at,
            decided_at=response.get("timestamp") or time.time(),
            user_identity=ctx.get("user_identity") if isinstance(ctx, dict) else None,
            attribution_tags=ctx.get("attribution_tags") if isinstance(ctx, dict) else None,
        )
        try:
            await ledger.record(entry)
        except Exception as e:
            logger.error(f"Failed to record HITL decision to ledger: {e}", exc_info=True)

    async def _load_chain_approver(self, approver_id: str) -> Optional[HumanApprovalRequestPlugin]:
        cached = self._chain_instance_cache.get(approver_id)
        if cached is not None:
            return cached
        config = self._approver_configurations.get(approver_id, {})
        try:
            instance_any = await self._plugin_manager.get_plugin_instance(approver_id, config=config)
        except Exception as e:
            logger.error(f"Error loading HITL approver '{approver_id}': {e}", exc_info=True)
            return None
        if instance_any and isinstance(instance_any, HumanApprovalRequestPlugin):
            self._chain_instance_cache[approver_id] = instance_any
            return cast(HumanApprovalRequestPlugin, instance_any)
        if instance_any:
            logger.warning(f"HITL approver '{approver_id}' loaded but is not a valid HumanApprovalRequestPlugin.")
        else:
            logger.warning(f"HITL approver '{approver_id}' not found or failed to load.")
        return None

    async def request_approval(self, request: ApprovalRequest, approver_id: Optional[str] = None) -> ApprovalResponse:
        requested_at = time.time()
        request_id_from_input = request.get("request_id")
        if request_id_from_input is None:
            logger.warning("Input ApprovalRequest dictionary is missing 'request_id'. Generating a new UUID for it.")
            request_id_from_input = str(uuid.uuid4())
        response_for_ledger: Optional[ApprovalResponse] = None
        try:
            response_for_ledger = await self._request_approval_inner(request, approver_id, request_id_from_input)
            return response_for_ledger
        finally:
            if response_for_ledger is not None:
                await self._record_to_ledger(request, response_for_ledger, requested_at)

    async def _request_approval_inner(
        self,
        request: ApprovalRequest,
        approver_id: Optional[str],
        request_id_from_input: str,
    ) -> ApprovalResponse:

        # Explicit approver_id bypasses the chain
        if approver_id:
            if approver_id.lower() == "none":
                return ApprovalResponse(request_id=request_id_from_input, status="denied", reason="No HITL approver configured.")
            approver = await self._load_chain_approver(approver_id)
            if not approver:
                logger.warning(f"Specified HITL approver '{approver_id}' not found or failed to load.")
                return ApprovalResponse(request_id=request_id_from_input, status="denied", reason=f"HITL approver '{approver_id}' unavailable.")
            try:
                resp = await approver.request_approval(request)
                # ask_human from an explicitly-targeted approver has no fallback;
                # treat it as denial-with-explanation rather than silently dropping.
                if resp.get("status") == "ask_human":
                    resp["status"] = "denied"
                    resp["reason"] = f"{resp.get('reason') or 'policy requested human review'} (no fallback approver configured)"
                return resp
            except Exception as e:
                logger.error(f"Error during HITL approval request with '{approver.plugin_id}': {e}", exc_info=True)
                return ApprovalResponse(request_id=request_id_from_input, status="error", reason=f"Error in approval process: {e!s}")

        # Default chain
        if not self._default_chain_ids:
            logger.warning("HITL approval requested, but no approver ID specified and no default configured.")
            return ApprovalResponse(request_id=request_id_from_input, status="denied", reason="No HITL approver configured.")

        is_single_approver = len(self._default_chain_ids) == 1
        last_response: Optional[ApprovalResponse] = None
        for idx, aid in enumerate(self._default_chain_ids):
            approver = await self._load_chain_approver(aid)
            if not approver:
                if is_single_approver:
                    # Legacy: a single default approver that fails to load is
                    # an unavailable-approver denial (preserves original error shape).
                    logger.error(f"HITL approver '{aid}' could not be loaded or is not configured.")
                    return ApprovalResponse(
                        request_id=request_id_from_input,
                        status="denied",
                        reason=f"HITL approver '{aid}' unavailable.",
                    )
                # Multi-link chain: skip the broken link and try the next.
                continue
            try:
                resp = await approver.request_approval(request)
            except Exception as e:
                if is_single_approver:
                    # Legacy: an approver raising an exception is reported as
                    # a status="error" response, not a chain skip. Preserve the
                    # original "Error during HITL approval request with '<id>'"
                    # log shape for backward compat with downstream log parsers.
                    logger.error(
                        f"Error during HITL approval request with '{approver.plugin_id}': {e}",
                        exc_info=True,
                    )
                    return ApprovalResponse(
                        request_id=request_id_from_input,
                        status="error",
                        reason=f"Error in approval process: {e!s}",
                    )
                logger.error(
                    f"Error during HITL approval with chain step '{aid}': {e}",
                    exc_info=True,
                )
                continue
            last_response = resp
            if resp.get("status") == "ask_human":
                # Delegate to the next link.
                continue
            # First decisive response wins.
            return resp

        # Chain exhausted. If the final link still said ask_human (e.g. webhook
        # in the chain returned ask_human which would be unusual), surface a
        # denial with explanation.
        if last_response is not None:
            if last_response.get("status") == "ask_human":
                last_response["status"] = "denied"
                last_response["reason"] = (
                    f"{last_response.get('reason') or 'policy requested human review'} "
                    "(chain exhausted without a decisive approver)"
                )
            return last_response

        return ApprovalResponse(
            request_id=request_id_from_input,
            status="denied",
            reason="HITL chain exhausted without a usable approver.",
        )

    async def teardown(self) -> None:
        logger.info("HITLManager tearing down...")
        if self._ledger:
            try:
                await self._ledger.teardown()
            except Exception as e:
                logger.error(f"Error tearing down HITL ledger: {e}", exc_info=True)
            self._ledger = None
            self._ledger_initialized = False
        if self._default_approver_instance:
            try:
                await self._default_approver_instance.teardown()
            except Exception as e:
                logger.error(f"Error tearing down default HITL approver '{self._default_approver_instance.plugin_id}': {e}", exc_info=True)
        for aid, inst in list(self._chain_instance_cache.items()):
            try:
                await inst.teardown()
            except Exception as e:
                logger.error(f"Error tearing down chain HITL approver '{aid}': {e}", exc_info=True)
        self._chain_instance_cache.clear()
        self._default_approver_instance = None
        self._initialized_default = False
        logger.info("HITLManager teardown complete.")
