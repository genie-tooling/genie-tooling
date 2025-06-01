"""Unit tests for the SandboxedFileSystemTool."""
from pathlib import Path
from typing import AsyncGenerator, Optional

import aiofiles
import pytest

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.impl.sandboxed_fs_tool import SandboxedFileSystemTool


class MockFSKeyProvider(KeyProvider): # Not used by tool, but required by execute signature
    async def get_key(self, key_name: str) -> Optional[str]: return None
    async def setup(self,c=None): pass
    async def teardown(self): pass

@pytest.fixture
async def fs_tool_with_sandbox(tmp_path: Path) -> AsyncGenerator[SandboxedFileSystemTool, None]:
    sandbox_root = tmp_path / "test_sandbox"
    tool = SandboxedFileSystemTool()
    await tool.setup(config={"sandbox_base_path": str(sandbox_root)})
    yield tool

@pytest.fixture
def mock_fs_key_provider() -> MockFSKeyProvider:
    return MockFSKeyProvider()

@pytest.mark.asyncio
async def test_fs_get_metadata(fs_tool_with_sandbox: AsyncGenerator[SandboxedFileSystemTool, None]):
    tool = await anext(fs_tool_with_sandbox)
    metadata = await tool.get_metadata()
    assert metadata["identifier"] == "sandboxed_fs_tool_v1"
    assert metadata["name"] == "Sandboxed File Operations"
    assert "operation" in metadata["input_schema"]["properties"]
    assert metadata["input_schema"]["properties"]["operation"]["enum"] == ["read_file", "write_file", "list_directory"]
    assert metadata["cacheable"] is False

@pytest.mark.asyncio
async def test_fs_setup_success_path_exists(tmp_path: Path):
    sandbox_dir = tmp_path / "existing_sandbox"
    sandbox_dir.mkdir()
    tool = SandboxedFileSystemTool()
    await tool.setup(config={"sandbox_base_path": str(sandbox_dir)})
    assert tool._sandbox_root == sandbox_dir.resolve()
    assert sandbox_dir.is_dir()

@pytest.mark.asyncio
async def test_fs_setup_success_creates_path(tmp_path: Path):
    sandbox_dir_to_create = tmp_path / "new_sandbox_to_be_created_by_setup"
    tool = SandboxedFileSystemTool()

    await tool.setup(config={"sandbox_base_path": str(sandbox_dir_to_create)})

    assert tool._sandbox_root == sandbox_dir_to_create.resolve()
    assert sandbox_dir_to_create.exists(), "Sandbox directory should have been created by tool's setup"
    assert sandbox_dir_to_create.is_dir()


@pytest.mark.asyncio
async def test_fs_setup_fail_path_is_file(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    file_path = tmp_path / "iam_a_file.txt"
    file_path.write_text("content")
    tool = SandboxedFileSystemTool()
    await tool.setup(config={"sandbox_base_path": str(file_path)})
    assert tool._sandbox_root is None
    assert f"Configured 'sandbox_base_path' ({file_path.resolve()}) exists but is not a directory." in caplog.text

@pytest.mark.asyncio
async def test_fs_setup_fail_no_path_configured(caplog: pytest.LogCaptureFixture):
    tool = SandboxedFileSystemTool()
    await tool.setup(config={})
    assert tool._sandbox_root is None
    assert "'sandbox_base_path' not configured. Tool will be disabled." in caplog.text

@pytest.mark.asyncio
async def test_fs_resolve_secure_path_success(fs_tool_with_sandbox: AsyncGenerator[SandboxedFileSystemTool, None]):
    tool = await anext(fs_tool_with_sandbox)
    assert tool._sandbox_root is not None, "Sandbox root should be initialized by the fixture"
    resolved = tool._resolve_secure_path("test.txt")
    assert resolved == tool._sandbox_root / "test.txt"
    resolved_subdir = tool._resolve_secure_path("subdir/another.md")
    assert resolved_subdir == tool._sandbox_root / "subdir" / "another.md"

@pytest.mark.asyncio
async def test_fs_resolve_secure_path_fail_traversal(fs_tool_with_sandbox: AsyncGenerator[SandboxedFileSystemTool, None]):
    tool = await anext(fs_tool_with_sandbox)
    assert tool._sandbox_root is not None, "Sandbox root should be initialized by the fixture"
    with pytest.raises(PermissionError, match="Path traversal attempt detected"):
        tool._resolve_secure_path("../../../etc/hosts")
    with pytest.raises(PermissionError, match="Path traversal attempt detected"):
        tool._resolve_secure_path("some_dir/../../test_sandbox/../file")


@pytest.mark.asyncio
async def test_fs_resolve_secure_path_fail_absolute_path(fs_tool_with_sandbox: AsyncGenerator[SandboxedFileSystemTool, None]):
    tool = await anext(fs_tool_with_sandbox)
    assert tool._sandbox_root is not None, "Sandbox root should be initialized by the fixture"
    with pytest.raises(ValueError, match="Relative path cannot be absolute"):
        tool._resolve_secure_path("/etc/hosts")

@pytest.mark.asyncio
async def test_fs_execute_sandbox_not_initialized(mock_fs_key_provider: MockFSKeyProvider):
    tool_no_setup = SandboxedFileSystemTool()
    result = await tool_no_setup.execute({"operation": "read_file", "path": "test.txt"}, mock_fs_key_provider)
    assert result["success"] is False
    assert result["message"] == "Sandbox not initialized. Check configuration."

@pytest.mark.asyncio
async def test_fs_execute_read_file_success(fs_tool_with_sandbox: AsyncGenerator[SandboxedFileSystemTool, None], mock_fs_key_provider: MockFSKeyProvider):
    tool = await anext(fs_tool_with_sandbox)
    assert tool._sandbox_root is not None, "Sandbox root should be initialized by the fixture"
    file_in_sandbox = tool._sandbox_root / "readable.txt"
    expected_content = "Content to be read."
    async with aiofiles.open(file_in_sandbox, "w") as f:
        await f.write(expected_content)

    result = await tool.execute({"operation": "read_file", "path": "readable.txt"}, mock_fs_key_provider)
    assert result["success"] is True
    assert result["content"] == expected_content

@pytest.mark.asyncio
async def test_fs_execute_read_file_not_found(fs_tool_with_sandbox: AsyncGenerator[SandboxedFileSystemTool, None], mock_fs_key_provider: MockFSKeyProvider):
    tool = await anext(fs_tool_with_sandbox)
    assert tool._sandbox_root is not None, "Sandbox root should be initialized by the fixture"
    result = await tool.execute({"operation": "read_file", "path": "non_existent_file.txt"}, mock_fs_key_provider)
    assert result["success"] is False
    assert result["message"] == "File not found: non_existent_file.txt"

@pytest.mark.asyncio
async def test_fs_execute_write_file_success_create_and_read_back(fs_tool_with_sandbox: AsyncGenerator[SandboxedFileSystemTool, None], mock_fs_key_provider: MockFSKeyProvider):
    tool = await anext(fs_tool_with_sandbox)
    assert tool._sandbox_root is not None, "Sandbox root should be initialized by the fixture"
    file_path_str = "newly_written.txt"
    content = "This is fresh content."

    write_result = await tool.execute({"operation": "write_file", "path": file_path_str, "content": content}, mock_fs_key_provider)
    assert write_result["success"] is True
    assert (tool._sandbox_root / file_path_str).exists()

    read_result = await tool.execute({"operation": "read_file", "path": file_path_str}, mock_fs_key_provider)
    assert read_result["success"] is True
    assert read_result["content"] == content

@pytest.mark.asyncio
async def test_fs_execute_write_file_fail_no_overwrite(fs_tool_with_sandbox: AsyncGenerator[SandboxedFileSystemTool, None], mock_fs_key_provider: MockFSKeyProvider):
    tool = await anext(fs_tool_with_sandbox)
    assert tool._sandbox_root is not None, "Sandbox root should be initialized by the fixture"
    existing_file = tool._sandbox_root / "exists.txt"
    async with aiofiles.open(existing_file, "w") as f: await f.write("original")

    result = await tool.execute({"operation": "write_file", "path": "exists.txt", "content": "new", "overwrite": False}, mock_fs_key_provider)
    assert result["success"] is False
    assert "File exists and overwrite is false" in result["message"] # type: ignore
    async with aiofiles.open(existing_file, "r") as f: assert await f.read() == "original"

@pytest.mark.asyncio
async def test_fs_execute_write_file_creates_parent_dirs(fs_tool_with_sandbox: AsyncGenerator[SandboxedFileSystemTool, None], mock_fs_key_provider: MockFSKeyProvider):
    tool = await anext(fs_tool_with_sandbox)
    assert tool._sandbox_root is not None, "Sandbox root should be initialized by the fixture"
    deep_path_str = "level1/level2/deep_file.log"
    content = "Deep thoughts here."

    write_result = await tool.execute({"operation": "write_file", "path": deep_path_str, "content": content}, mock_fs_key_provider)
    assert write_result["success"] is True

    full_path_in_sandbox = tool._sandbox_root / deep_path_str
    assert full_path_in_sandbox.exists()
    assert full_path_in_sandbox.parent.exists()
    assert full_path_in_sandbox.parent.is_dir()
    assert full_path_in_sandbox.parent.parent.exists()
    assert full_path_in_sandbox.parent.parent.is_dir()
    async with aiofiles.open(full_path_in_sandbox, "r") as f:
        assert await f.read() == content

@pytest.mark.asyncio
async def test_fs_execute_list_directory_success(fs_tool_with_sandbox: AsyncGenerator[SandboxedFileSystemTool, None], mock_fs_key_provider: MockFSKeyProvider):
    tool = await anext(fs_tool_with_sandbox)
    assert tool._sandbox_root is not None, "Sandbox root should be initialized by the fixture"
    (tool._sandbox_root / "file1.txt").write_text("f1")
    (tool._sandbox_root / "file2.log").write_text("f2")
    (tool._sandbox_root / "sub").mkdir()
    (tool._sandbox_root / "sub" / "subfile.md").write_text("sf")

    result = await tool.execute({"operation": "list_directory", "path": "."}, mock_fs_key_provider)
    assert result["success"] is True
    assert sorted(result["file_list"]) == sorted(["file1.txt", "file2.log", "sub"]) # type: ignore

    result_sub = await tool.execute({"operation": "list_directory", "path": "sub"}, mock_fs_key_provider)
    assert result_sub["success"] is True
    assert result_sub["file_list"] == ["subfile.md"]

@pytest.mark.asyncio
async def test_fs_execute_unknown_operation(fs_tool_with_sandbox: AsyncGenerator[SandboxedFileSystemTool, None], mock_fs_key_provider: MockFSKeyProvider):
    tool = await anext(fs_tool_with_sandbox)
    assert tool._sandbox_root is not None, "Sandbox root should be initialized by the fixture"
    result = await tool.execute({"operation": "delete_everything", "path": "."}, mock_fs_key_provider)
    assert result["success"] is False
    assert result["message"] == "Unknown operation: delete_everything"
