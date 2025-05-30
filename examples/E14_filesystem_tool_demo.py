# examples/E14_filesystem_tool_demo.py
"""
Example: SandboxedFileSystemTool Demo
--------------------------------------
This example demonstrates configuring and using the SandboxedFileSystemTool
via the Genie facade.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Run from the root of the project:
   `poetry run python examples/E14_filesystem_tool_demo.py`

The demo will:
- Create a sandbox directory ('./my_agent_sandbox_demo').
- Initialize Genie with the SandboxedFileSystemTool configured to use this sandbox.
- Perform write, read, and list operations using the tool.
- Clean up the sandbox directory.
"""
import asyncio
import os
import shutil
from pathlib import Path

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.genie import Genie
from genie_tooling.security.key_provider import KeyProvider


# 1. Basic KeyProvider (not strictly needed by FS tool, but Genie requires one)
class DemoFSKeyProvider(KeyProvider, CorePluginType):
    plugin_id = "demo_fs_key_provider_v1"
    async def get_key(self, key_name: str) -> str | None: return os.environ.get(key_name)
    async def setup(self, config=None): pass
    async def teardown(self): pass

async def run_fs_tool_demo():
    print("--- SandboxedFileSystemTool Demo ---")

    sandbox_dir_name = "my_agent_sandbox_demo"
    sandbox_path = Path(__file__).parent / sandbox_dir_name

    # Ensure clean state for the demo
    if sandbox_path.exists():
        shutil.rmtree(sandbox_path)
    sandbox_path.mkdir(parents=True, exist_ok=True)
    print(f"Sandbox directory prepared at: {sandbox_path.resolve()}")

    # 2. Configure MiddlewareConfig
    app_config = MiddlewareConfig(
        features=FeatureSettings(
            # No specific features needed for direct tool execution demo,
            # but an LLM might be needed if using run_command with LLM-assisted processor.
            # For simplicity, we'll use direct execute_tool.
            llm="none",
            command_processor="none"
        ),
        tool_configurations={
            "sandboxed_fs_tool_v1": {"sandbox_base_path": str(sandbox_path.resolve())}
        }
    )

    # 3. Instantiate KeyProvider and Genie
    key_provider = DemoFSKeyProvider()
    genie: Genie | None = None

    try:
        genie = await Genie.create(config=app_config, key_provider_instance=key_provider)
        print("Genie facade initialized with SandboxedFileSystemTool.")

        # 4. Demonstrate Tool Operations
        file_to_write = "test_doc.txt"
        content_to_write = "Hello from the sandboxed world of Genie!"
        subdir_file = "notes/important.md"
        subdir_content = "# Project Ideas\n- Build more cool agents!"

        # Write a file
        print(f"\nAttempting to write '{file_to_write}'...")
        write_result = await genie.execute_tool(
            "sandboxed_fs_tool_v1",
            operation="write_file",
            path=file_to_write,
            content=content_to_write
        )
        print(f"Write result: {write_result}")
        assert write_result.get("success") is True

        # Write a file in a subdirectory (should be created)
        print(f"\nAttempting to write '{subdir_file}'...")
        write_subdir_result = await genie.execute_tool(
            "sandboxed_fs_tool_v1",
            operation="write_file",
            path=subdir_file,
            content=subdir_content,
            overwrite=True # Allow overwrite for demo simplicity
        )
        print(f"Write to subdir result: {write_subdir_result}")
        assert write_subdir_result.get("success") is True


        # Read the file
        print(f"\nAttempting to read '{file_to_write}'...")
        read_result = await genie.execute_tool(
            "sandboxed_fs_tool_v1",
            operation="read_file",
            path=file_to_write
        )
        print(f"Read result: {read_result}")
        assert read_result.get("success") is True
        assert read_result.get("content") == content_to_write

        # List directory (root of sandbox)
        print("\nAttempting to list sandbox root directory '.' ...")
        list_result_root = await genie.execute_tool(
            "sandboxed_fs_tool_v1",
            operation="list_directory",
            path="." # Relative to sandbox root
        )
        print(f"List directory (root) result: {list_result_root}")
        assert list_result_root.get("success") is True
        assert file_to_write in list_result_root.get("file_list", [])
        assert "notes" in list_result_root.get("file_list", [])


        # List subdirectory
        print("\nAttempting to list 'notes' subdirectory...")
        list_result_subdir = await genie.execute_tool(
            "sandboxed_fs_tool_v1",
            operation="list_directory",
            path="notes"
        )
        print(f"List directory ('notes') result: {list_result_subdir}")
        assert list_result_subdir.get("success") is True
        assert Path(subdir_file).name in list_result_subdir.get("file_list", [])


        # Attempt path traversal (should fail)
        print("\nAttempting path traversal (should fail)...")
        traversal_result = await genie.execute_tool(
            "sandboxed_fs_tool_v1",
            operation="read_file",
            path="../outside_file.txt"
        )
        print(f"Path traversal result: {traversal_result}")
        assert traversal_result.get("success") is False
        assert "Path traversal attempt detected" in traversal_result.get("message", "")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if genie:
            await genie.close()
            print("\nGenie facade torn down.")

        # Cleanup sandbox directory
        if sandbox_path.exists():
            shutil.rmtree(sandbox_path)
            print(f"Sandbox directory '{sandbox_path.resolve()}' cleaned up.")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    # For more detailed Genie logs:
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(run_fs_tool_demo())
