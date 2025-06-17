import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
import aiofiles.os as aios  # For async os operations like listdir

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

class SandboxedFileSystemTool(Tool):
    identifier: str = "sandboxed_fs_tool_v1"
    plugin_id: str = "sandboxed_fs_tool_v1"

    _sandbox_root: Optional[Path] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        sandbox_base_path_str = cfg.get("sandbox_base_path")
        if not sandbox_base_path_str:
            logger.error(f"{self.identifier}: 'sandbox_base_path' not configured. Tool will be disabled.")
            self._sandbox_root = None # Ensure it's None
            return

        try:
            # Resolve the path. strict=False allows resolving even if it doesn't exist yet.
            resolved_path = Path(sandbox_base_path_str).resolve(strict=False)

            # Synchronous directory creation is acceptable in setup for path initialization.
            # If the path already exists and is a directory, mkdir with exist_ok=True is fine.
            # If it exists and is a file, it will raise an error.
            if resolved_path.exists() and not resolved_path.is_dir():
                logger.error(f"{self.identifier}: Configured 'sandbox_base_path' ({resolved_path}) exists but is not a directory. Tool disabled.")
                self._sandbox_root = None
                return

            # Create the directory if it doesn't exist.
            # pathlib.Path.mkdir is synchronous.
            resolved_path.mkdir(parents=True, exist_ok=True)

            self._sandbox_root = resolved_path # Assign only after successful creation/validation
            logger.info(f"{self.identifier}: Sandbox root initialized at {self._sandbox_root}")

        except Exception as e:
            logger.error(f"{self.identifier}: Error setting up sandbox directory '{sandbox_base_path_str}': {e}", exc_info=True)
            self._sandbox_root = None

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Sandboxed File Operations",
            "description_human": "Performs basic file operations (read, write, list) within a securely sandboxed directory. Path traversal outside the sandbox is prevented. Requires 'sandbox_base_path' configuration during setup.",
            "description_llm": "FileOps: Read/write files, list directories in a safe sandbox. Args: operation (str: 'read_file', 'write_file', 'list_directory'), path (str, relative to sandbox), content (str, for write_file, opt), overwrite (bool, for write_file, opt, default False).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["read_file", "write_file", "list_directory"]},
                    "path": {"type": "string", "description": "Relative path within the sandbox (e.g., 'notes/meeting.txt')."},
                    "content": {"type": "string", "description": "Content for 'write_file'."},
                    "overwrite": {"type": "boolean", "default": False, "description": "If true, overwrite existing file."}
                },
                "required": ["operation", "path"],
                "if": {"properties": {"operation": {"const": "write_file"}}},
                "then": {"required": ["content"]}
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": ["string", "null"]},
                    "content": {"type": ["string", "null"]},
                    "file_list": {"type": ["array", "null"], "items": {"type": "string"}}
                }, "required": ["success"]
            },
            "key_requirements": [], "tags": ["file", "filesystem", "local", "storage", "sandboxed"],
            "version": "1.0.0", "cacheable": False
        }

    def _resolve_secure_path(self, relative_path_str: str) -> Path:
        if not self._sandbox_root:
            raise PermissionError("Sandbox not initialized.")
        if Path(relative_path_str).is_absolute():
            raise ValueError("Relative path cannot be absolute.")

        # Normalize the relative path to prevent issues like "subdir/../file"
        # effectively becoming "file" before joining with sandbox_root.
        # os.path.normpath is good for this, but Path objects also handle some normalization.
        # A common way to handle this with pathlib is to resolve the joined path.
        prospective_path = (self._sandbox_root / relative_path_str).resolve()

        # Check if the resolved path is truly within the sandbox root.
        # This is the crucial jailbreak check.
        if self._sandbox_root not in prospective_path.parents and prospective_path != self._sandbox_root:
            # More robust check:
            # common = os.path.commonpath([str(self._sandbox_root.resolve()), str(prospective_path.resolve())])
            # if common != str(self._sandbox_root.resolve()):
            # A simpler check for most cases:
            if not str(prospective_path).startswith(str(self._sandbox_root.resolve()) + Path.home().drive + Path.home().root): # Handles drive letters for windows
                 if not str(prospective_path).startswith(str(self._sandbox_root.resolve())): # For non-windows
                    raise PermissionError(f"Path traversal attempt detected: '{relative_path_str}' resolves to '{prospective_path}' which is outside sandbox '{self._sandbox_root}'.")
        return prospective_path

    async def execute(
        self, params: Dict[str, Any], key_provider: KeyProvider, context: Dict[str, Any]
    ) -> Dict[str, Union[bool, str, List[str], None]]:
        if not self._sandbox_root:
            return {"success": False, "message": "Sandbox not initialized. Check configuration."}

        operation = params["operation"]
        relative_path = params["path"]

        try:
            target_path = self._resolve_secure_path(relative_path)

            if operation == "read_file":
                if not await aios.path.isfile(target_path): # Use aiofiles.os.path for async check
                    return {"success": False, "message": f"File not found: {relative_path}"}
                async with aiofiles.open(target_path, mode="r", encoding="utf-8") as f:
                    content = await f.read()
                return {"success": True, "content": content, "message": "File read successfully."}

            elif operation == "write_file":
                content_to_write = params.get("content", "")
                overwrite = params.get("overwrite", False)

                if await aios.path.exists(target_path) and not overwrite and await aios.path.isfile(target_path):
                    return {"success": False, "message": f"File exists and overwrite is false: {relative_path}"}

                parent_dir = target_path.parent
                # Ensure parent_dir is within sandbox before creating
                if self._sandbox_root not in parent_dir.parents and parent_dir != self._sandbox_root:
                     return {"success": False, "message": "Cannot create parent directory outside sandbox."}

                # Use synchronous mkdir for parent directory creation within execute,
                # or ensure aios.mkdir is used correctly if it's preferred.
                # For simplicity and robustness in setup-like operations:
                if not await aios.path.isdir(parent_dir):
                    parent_dir.mkdir(parents=True, exist_ok=True) # Pathlib's sync mkdir

                async with aiofiles.open(target_path, mode="w", encoding="utf-8") as f:
                    await f.write(content_to_write)
                return {"success": True, "message": f"File written successfully to {relative_path}."}

            elif operation == "list_directory":
                if not await aios.path.isdir(target_path): # Use aiofiles.os.path for async check
                    return {"success": False, "message": f"Directory not found: {relative_path}"}
                items = await aios.listdir(target_path) # Use aiofiles.os.listdir
                return {"success": True, "file_list": items, "message": "Directory listed successfully."}

            else:
                return {"success": False, "message": f"Unknown operation: {operation}"}

        except PermissionError as e_perm:
            logger.warning(f"{self.identifier}: Permission error for path '{relative_path}': {e_perm}")
            return {"success": False, "message": str(e_perm)}
        except ValueError as e_val:
             logger.warning(f"{self.identifier}: Value error for path '{relative_path}': {e_val}")
             return {"success": False, "message": str(e_val)}
        except FileNotFoundError:
            return {"success": False, "message": f"Path not found: {relative_path}"}
        except Exception as e:
            logger.error(f"{self.identifier}: Error during file operation '{operation}' on '{relative_path}': {e}", exc_info=True)
            return {"success": False, "message": f"An unexpected error occurred: {e!s}"}

    async def teardown(self) -> None:
        logger.debug(f"{self.identifier}: Teardown complete (no specific resources to release).")
