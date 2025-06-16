"""HITLManager: Orchestrates HumanApprovalRequestPlugins."""
import logging
import uuid
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager

from .abc import HumanApprovalRequestPlugin
from .types import ApprovalRequest, ApprovalResponse

logger = logging.getLogger(__name__)

class HITLManager:
    def __init__(self, plugin_manager: PluginManager, default_approver_id: Optional[str] = None, approver_configurations: Optional[Dict[str, Dict[str, Any]]] = None):
        self._plugin_manager = plugin_manager
        self._default_approver_id = default_approver_id
        self._approver_configurations = approver_configurations or {}
        self._default_approver_instance: Optional[HumanApprovalRequestPlugin] = None
        self._initialized_default = False
        logger.info("HITLManager initialized.")

    @property
    def is_active(self) -> bool:
        """Returns True if a default approver is configured (and not 'none')."""
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

    async def request_approval(self, request: ApprovalRequest, approver_id: Optional[str] = None) -> ApprovalResponse:
        target_approver: Optional[HumanApprovalRequestPlugin] = None
        target_approver_id = approver_id or self._default_approver_id

        request_id_from_input = request.get("request_id")
        if request_id_from_input is None:
            logger.warning("Input ApprovalRequest dictionary is missing 'request_id'. Generating a new UUID for it.")
            request_id_from_input = str(uuid.uuid4())

        if not target_approver_id or target_approver_id.lower() == "none":
            logger.warning("HITL approval requested, but no approver ID specified and no default configured.")
            return ApprovalResponse(request_id=request_id_from_input, status="denied", reason="No HITL approver configured.")

        if approver_id and approver_id == self._default_approver_id:
            target_approver = await self._get_default_approver()
        elif approver_id:
            config = self._approver_configurations.get(approver_id, {})
            try:
                instance_any = await self._plugin_manager.get_plugin_instance(approver_id, config=config)
                if instance_any and isinstance(instance_any, HumanApprovalRequestPlugin):
                    target_approver = cast(HumanApprovalRequestPlugin, instance_any)
                elif instance_any:
                    logger.warning(f"Specified HITL approver '{approver_id}' loaded but is not a valid HumanApprovalRequestPlugin.")
                    target_approver = None
                else:
                    logger.warning(f"Specified HITL approver '{approver_id}' not found or failed to load.")
                    target_approver = None
            except Exception as e:
                logger.error(f"Error loading specified HITL approver '{approver_id}': {e}", exc_info=True)
                target_approver = None
        else:
            target_approver = await self._get_default_approver()

        if not target_approver:
            logger.error(f"HITL approver '{target_approver_id}' could not be loaded or is not configured.")
            return ApprovalResponse(request_id=request_id_from_input, status="denied", reason=f"HITL approver '{target_approver_id}' unavailable.")

        try:
            return await target_approver.request_approval(request)
        except Exception as e:
            logger.error(f"Error during HITL approval request with '{target_approver.plugin_id}': {e}", exc_info=True)
            return ApprovalResponse(request_id=request_id_from_input, status="error", reason=f"Error in approval process: {e!s}")

    async def teardown(self) -> None:
        logger.info("HITLManager tearing down...")
        if self._default_approver_instance:
            try:
                await self._default_approver_instance.teardown()
            except Exception as e:
                logger.error(f"Error tearing down default HITL approver '{self._default_approver_instance.plugin_id}': {e}", exc_info=True)
        self._default_approver_instance = None
        self._initialized_default = False
        logger.info("HITLManager teardown complete.")
