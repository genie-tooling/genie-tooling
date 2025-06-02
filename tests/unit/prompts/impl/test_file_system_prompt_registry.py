import logging
from pathlib import Path
from unittest.mock import patch

import aiofiles
import pytest
from genie_tooling.prompts.impl.file_system_prompt_registry import (
    FileSystemPromptRegistryPlugin,
)

REGISTRY_LOGGER_NAME = "genie_tooling.prompts.impl.file_system_prompt_registry"

@pytest.fixture
async def fs_prompt_registry(tmp_path: Path) -> FileSystemPromptRegistryPlugin:
    registry = FileSystemPromptRegistryPlugin()
    # Create a temporary base path for prompts within the test's tmp_path
    prompt_dir = tmp_path / "test_prompts"
    prompt_dir.mkdir()
    await registry.setup(config={"base_path": str(prompt_dir)})
    return registry

@pytest.mark.asyncio
async def test_setup_creates_directory_if_not_exists(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=REGISTRY_LOGGER_NAME)
    non_existent_dir = tmp_path / "new_prompts_dir"
    registry = FileSystemPromptRegistryPlugin()
    await registry.setup(config={"base_path": str(non_existent_dir)})
    assert non_existent_dir.is_dir()
    assert f"Base path '{non_existent_dir.resolve()}' does not exist or is not a directory. Creating it." in caplog.text
    assert registry._base_path == non_existent_dir.resolve()

@pytest.mark.asyncio
async def test_setup_uses_existing_directory(tmp_path: Path):
    existing_dir = tmp_path / "prompts_already_here"
    existing_dir.mkdir()
    registry = FileSystemPromptRegistryPlugin()
    await registry.setup(config={"base_path": str(existing_dir)})
    assert registry._base_path == existing_dir.resolve()

@pytest.mark.asyncio
async def test_setup_base_path_is_file_error(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=REGISTRY_LOGGER_NAME)
    file_path = tmp_path / "iam_a_file.prompt"
    file_path.touch()
    registry = FileSystemPromptRegistryPlugin()
    await registry.setup(config={"base_path": str(file_path)})
    assert registry._base_path is None # Should be None on error
    assert f"Error setting up base path '{file_path.resolve()}'" in caplog.text

@pytest.mark.asyncio
async def test_get_template_content_success(fs_prompt_registry: FileSystemPromptRegistryPlugin):
    registry = await fs_prompt_registry
    assert registry._base_path is not None
    prompt_file = registry._base_path / "my_test_prompt.prompt"
    prompt_content = "Hello, {{ name }}!"
    async with aiofiles.open(prompt_file, "w") as f:
        await f.write(prompt_content)

    content = await registry.get_template_content("my_test_prompt")
    assert content == prompt_content

@pytest.mark.asyncio
async def test_get_template_content_with_version(fs_prompt_registry: FileSystemPromptRegistryPlugin):
    registry = await fs_prompt_registry
    assert registry._base_path is not None
    prompt_v1_file = registry._base_path / "versioned_prompt.v1.prompt"
    prompt_v1_content = "Version 1: {{ item }}"
    async with aiofiles.open(prompt_v1_file, "w") as f:
        await f.write(prompt_v1_content)

    prompt_latest_file = registry._base_path / "versioned_prompt.prompt"
    prompt_latest_content = "Latest Version: {{ item }}"
    async with aiofiles.open(prompt_latest_file, "w") as f:
        await f.write(prompt_latest_content)

    content_v1 = await registry.get_template_content("versioned_prompt", version="1")
    assert content_v1 == prompt_v1_content

    content_latest_explicit = await registry.get_template_content("versioned_prompt")
    assert content_latest_explicit == prompt_latest_content

    # Test fallback if specific version not found but unversioned exists
    content_v2_fallback = await registry.get_template_content("versioned_prompt", version="2")
    assert content_v2_fallback == prompt_latest_content


@pytest.mark.asyncio
async def test_get_template_content_not_found(fs_prompt_registry: FileSystemPromptRegistryPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=REGISTRY_LOGGER_NAME)
    registry = await fs_prompt_registry
    content = await registry.get_template_content("non_existent_prompt")
    assert content is None
    assert "Template 'non_existent_prompt' (version: any) not found" in caplog.text

@pytest.mark.asyncio
async def test_get_template_content_base_path_not_set(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=REGISTRY_LOGGER_NAME)
    registry_no_setup = FileSystemPromptRegistryPlugin() # No setup call
    content = await registry_no_setup.get_template_content("any_prompt")
    assert content is None
    assert "Base path not configured or invalid. Cannot load template." in caplog.text

@pytest.mark.asyncio
async def test_get_template_content_read_error(fs_prompt_registry: FileSystemPromptRegistryPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=REGISTRY_LOGGER_NAME)
    registry = await fs_prompt_registry
    assert registry._base_path is not None
    prompt_file = registry._base_path / "error_prompt.prompt"
    # Create the file so it's found
    async with aiofiles.open(prompt_file, "w") as f:
        await f.write("content")

    with patch("aiofiles.open", side_effect=IOError("Simulated read error")):
        content = await registry.get_template_content("error_prompt")
    assert content is None
    assert f"Error reading template file '{prompt_file}'" in caplog.text

@pytest.mark.asyncio
async def test_list_available_templates_success(fs_prompt_registry: FileSystemPromptRegistryPlugin):
    registry = await fs_prompt_registry
    assert registry._base_path is not None
    (registry._base_path / "prompt1.prompt").write_text("p1")
    (registry._base_path / "prompt2.v1.prompt").write_text("p2v1")
    (registry._base_path / "prompt2.v2.prompt").write_text("p2v2")
    sub_dir = registry._base_path / "subdir"
    sub_dir.mkdir()
    (sub_dir / "sub_prompt.prompt").write_text("sp")

    templates = await registry.list_available_templates()
    assert len(templates) == 4 # prompt1, prompt2.v1, prompt2.v2, sub_prompt

    names_versions = {(t["name"], t["version"]) for t in templates}
    assert ("prompt1", None) in names_versions
    assert ("prompt2", "1") in names_versions
    assert ("prompt2", "2") in names_versions
    assert ("sub_prompt", None) in names_versions

@pytest.mark.asyncio
async def test_list_available_templates_empty(fs_prompt_registry: FileSystemPromptRegistryPlugin):
    registry = await fs_prompt_registry
    templates = await registry.list_available_templates()
    assert len(templates) == 0

@pytest.mark.asyncio
async def test_list_available_templates_base_path_not_set(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=REGISTRY_LOGGER_NAME)
    registry_no_setup = FileSystemPromptRegistryPlugin()
    templates = await registry_no_setup.list_available_templates()
    assert len(templates) == 0
    assert "Base path not configured. Cannot list templates." in caplog.text

@pytest.mark.asyncio
async def test_list_available_templates_rglob_error(fs_prompt_registry: FileSystemPromptRegistryPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=REGISTRY_LOGGER_NAME)
    registry = await fs_prompt_registry
    assert registry._base_path is not None

    with patch.object(Path, "rglob", side_effect=OSError("Simulated rglob error")):
        templates = await registry.list_available_templates()

    assert len(templates) == 0
    assert f"Error listing templates in '{registry._base_path}': Simulated rglob error" in caplog.text

@pytest.mark.asyncio
async def test_teardown(fs_prompt_registry: FileSystemPromptRegistryPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=REGISTRY_LOGGER_NAME)
    registry = await fs_prompt_registry
    await registry.teardown()
    assert f"{registry.plugin_id}: Teardown complete." in caplog.text
