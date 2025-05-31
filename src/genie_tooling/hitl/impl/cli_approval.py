"""CliApprovalPlugin: Requests human approval via CLI input."""
import asyncio
import logging
import time
import uuid
from typing import Any, Dict, Optional

from genie_tooling.hitl.abc import HumanApprovalRequestPlugin
from genie_tooling.hitl.types import ApprovalRequest, ApprovalResponse

logger = logging.getLogger(__name__)

class CliApprovalPlugin(HumanApprovalRequestPlugin):
    plugin_id: str = "cli_approval_plugin_v1"
    description: str = "Requests human approval by prompting on the command line."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        logger.info(f"{self.plugin_id}: Initialized. Will prompt on CLI for approvals.")

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        req_id = request.get("request_id", str(uuid.uuid4()))
        prompt_message = request.get("prompt", "No prompt provided.")
        data_to_approve_str = str(request.get("data_to_approve", {}))[:500] # Truncate long data
        
        print("\n--- HUMAN APPROVAL REQUIRED ---")
        print(f"Request ID: {req_id}")
        print(f"Prompt: {prompt_message}")
        print(f"Data/Action: {data_to_approve_str}")
        if request.get("context"):
            print(f"Context: {str(request['context'])[:200]}")
        
        timeout = request.get("timeout_seconds")
        
        try:
            if timeout is not None and timeout <=0: timeout = None # Treat non-positive as no timeout

            loop = asyncio.get_running_loop()
            
            if timeout:
                user_input_task = loop.run_in_executor(None, input, "Approve? (yes/no/y/n): ")
                try:
                    user_input = await asyncio.wait_for(user_input_task, timeout=timeout)
                except asyncio.TimeoutError:
                    print("\nApproval request timed out.")
                    return ApprovalResponse(request_id=req_id, status="timeout", reason="User did not respond in time.", timestamp=time.time())
            else:
                user_input = await loop.run_in_executor(None, input, "Approve? (yes/no/y/n): ")

            user_input_lower = user_input.strip().lower()

            if user_input_lower in ["yes", "y"]:
                reason = await loop.run_in_executor(None, input, "Optional reason/comment for approval: ")
                return ApprovalResponse(request_id=req_id, status="approved", approver_id="cli_user", reason=reason.strip() or None, timestamp=time.time())
            else:
                reason = await loop.run_in_executor(None, input, "Reason for denial (required if not approved): ")
                while not reason.strip():
                    print("Denial reason cannot be empty.")
                    reason = await loop.run_in_executor(None, input, "Reason for denial: ")
                return ApprovalResponse(request_id=req_id, status="denied", approver_id="cli_user", reason=reason.strip(), timestamp=time.time())

        except Exception as e:
            logger.error(f"{self.plugin_id}: Error during CLI approval prompt: {e}", exc_info=True)
            return ApprovalResponse(request_id=req_id, status="error", reason=f"CLI prompt error: {str(e)}", timestamp=time.time())

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Teardown complete.")
