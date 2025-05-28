### tests/unit/tools/impl/test_code_execution_tool.py
"""Unit tests for the GenericCodeExecutionTool."""
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
# Updated import paths for CodeExecutionResult and CodeExecutor
from genie_tooling.code_executors.abc import CodeExecutionResult, CodeExecutor
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.impl.code_execution_tool import GenericCodeExecutionTool

# --- Mock Components ---

class MockCodeExecutor(CodeExecutor):
    """A mock CodeExecutor for testing GenericCodeExecutionTool."""
    def __init__(
        self,
        plugin_id: str,
        executor_id: str,
        description: str,
        supported_languages: List[str],
        execute_result: CodeExecutionResult
    ):
        self.plugin_id = plugin_id
        self.executor_id = executor_id
        self.description = description
        self.supported_languages = supported_languages
        self._execute_result = execute_result
        self.setup_called_with_config: Optional[Dict[str, Any]] = None
        self.teardown_called = False

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.setup_called_with_config = config

    async def teardown(self) -> None:
        self.teardown_called = True

    async def execute_code(
        self,
        language: str,
        code: str,
        timeout_seconds: int,
        input_data: Optional[Dict[str, Any]] = None
    ) -> CodeExecutionResult:
        if language not in self.supported_languages:
            return CodeExecutionResult("", f"Language {language} not supported by {self.executor_id}", None, "Unsupported language", 0.0)
        return self._execute_result


@pytest.fixture
def mock_plugin_manager_for_code_exec(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.list_discovered_plugin_classes = MagicMock(return_value={})
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture
def mock_key_provider_for_code_exec(mocker) -> KeyProvider:
    kp = mocker.AsyncMock(spec=KeyProvider)
    return kp


# --- Test Cases ---

@pytest.mark.asyncio
async def test_generic_code_exec_tool_init_and_get_metadata_no_executors(
    mock_plugin_manager_for_code_exec: PluginManager
):
    """Test initialization and metadata when no executors are found."""
    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    # Prime the cache or ensure discovery is called if necessary
    await tool._get_available_executors()

    metadata = await tool.get_metadata()
    assert metadata["identifier"] == "generic_code_execution_tool"
    assert "No code executors currently available." in metadata["description_human"]
    assert "Supported languages: None" in metadata["description_llm"]
    assert metadata["input_schema"]["properties"]["language"].get("enum") is None
    assert metadata["input_schema"]["properties"]["executor_id"].get("enum") is None


@pytest.mark.asyncio
async def test_generic_code_exec_tool_get_metadata_with_executors(
    mock_plugin_manager_for_code_exec: PluginManager
):
    """Test metadata generation when executors are present."""
    py_executor_res = CodeExecutionResult("py_out", "", None, None, 10.0)
    js_executor_res = CodeExecutionResult("js_out", "", None, None, 12.0)

    mock_py_executor = MockCodeExecutor(
        plugin_id="py_exec_plugin", executor_id="python_sandbox_v1",
        description="Secure Python sandbox.", supported_languages=["python", "python3"],
        execute_result=py_executor_res
    )
    mock_js_executor = MockCodeExecutor(
        plugin_id="js_exec_plugin", executor_id="nodejs_sandbox_v1",
        description="Secure Node.js sandbox.", supported_languages=["javascript", "js"],
        execute_result=js_executor_res
    )

    mock_plugin_manager_for_code_exec.list_discovered_plugin_classes.return_value = {
        "py_exec_plugin": MockCodeExecutor, # type: ignore # Class for discovery
        "js_exec_plugin": MockCodeExecutor, # type: ignore # Class for discovery
    }

    # Simulate get_plugin_instance returning our mock instances
    async def get_instance_side_effect(plugin_id_req: str, config: Optional[Dict[str, Any]] = None):
        if plugin_id_req == "py_exec_plugin":
            # Simulate setup being called by PluginManager or by the tool's discovery
            await mock_py_executor.setup(config)
            return mock_py_executor
        if plugin_id_req == "js_exec_plugin":
            await mock_js_executor.setup(config)
            return mock_js_executor
        return None
    mock_plugin_manager_for_code_exec.get_plugin_instance.side_effect = get_instance_side_effect

    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    await tool._get_available_executors() # Populate cache

    metadata = await tool.get_metadata()
    assert "python_sandbox_v1" in metadata["description_human"]
    assert "nodejs_sandbox_v1" in metadata["description_human"]
    assert "python" in metadata["input_schema"]["properties"]["language"]["enum"]
    assert "javascript" in metadata["input_schema"]["properties"]["language"]["enum"]
    assert "python_sandbox_v1" in metadata["input_schema"]["properties"]["executor_id"]["enum"]
    assert "nodejs_sandbox_v1" in metadata["input_schema"]["properties"]["executor_id"]["enum"]


@pytest.mark.asyncio
async def test_execute_python_code_auto_select_executor(
    mock_plugin_manager_for_code_exec: PluginManager,
    mock_key_provider_for_code_exec: KeyProvider
):
    """Test successful Python code execution with auto-selected executor."""
    expected_stdout = "Hello Python!"
    py_executor_res = CodeExecutionResult(expected_stdout, "", None, None, 25.5)
    mock_py_executor = MockCodeExecutor(
        plugin_id="py_exec", executor_id="py_sandbox", description="Py",
        supported_languages=["python"], execute_result=py_executor_res
    )

    mock_plugin_manager_for_code_exec.list_discovered_plugin_classes.return_value = {"py_exec": type(mock_py_executor)} # type: ignore
    mock_plugin_manager_for_code_exec.get_plugin_instance.return_value = mock_py_executor

    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    params = {"language": "python", "code_to_run": "print('Hello Python!')"}

    result_dict = await tool.execute(params, mock_key_provider_for_code_exec)

    assert result_dict["stdout"] == expected_stdout
    assert result_dict["error"] is None
    assert result_dict["execution_time_ms"] == 25.5


@pytest.mark.asyncio
async def test_execute_code_with_specific_executor_id(
    mock_plugin_manager_for_code_exec: PluginManager,
    mock_key_provider_for_code_exec: KeyProvider
):
    """Test execution when a specific executor ID is requested."""
    expected_output = "Specific exec works"
    exec_res = CodeExecutionResult(expected_output, "", None, None, 30.0)
    mock_executor1 = MockCodeExecutor("exec1_plug", "executor_one", "Desc1", ["python"], exec_res)

    # Irrelevant executor also discovered to test selection
    mock_executor2 = MockCodeExecutor("exec2_plug", "executor_two", "Desc2", ["javascript"], CodeExecutionResult("", "", None, None, 1.0))

    mock_plugin_manager_for_code_exec.list_discovered_plugin_classes.return_value = {
        "exec1_plug": type(mock_executor1),
        "exec2_plug": type(mock_executor2)
    }
    async def get_instance_side_effect(plugin_id_req: str, config=None):
        if plugin_id_req == "exec1_plug": return mock_executor1
        if plugin_id_req == "exec2_plug": return mock_executor2
        return None
    mock_plugin_manager_for_code_exec.get_plugin_instance.side_effect = get_instance_side_effect

    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    params = {"language": "python", "code_to_run": "...", "executor_id": "executor_one"}
    result_dict = await tool.execute(params, mock_key_provider_for_code_exec)

    assert result_dict["stdout"] == expected_output
    assert result_dict["error"] is None


@pytest.mark.asyncio
async def test_execute_unsupported_language(
    mock_plugin_manager_for_code_exec: PluginManager,
    mock_key_provider_for_code_exec: KeyProvider
):
    """Test execution attempt with a language not supported by any available executor."""
    mock_py_executor = MockCodeExecutor("py_exec", "py_s", "Py", ["python"], CodeExecutionResult("", "", None, None, 1.0))
    mock_plugin_manager_for_code_exec.list_discovered_plugin_classes.return_value = {"py_exec": type(mock_py_executor)}
    mock_plugin_manager_for_code_exec.get_plugin_instance.return_value = mock_py_executor

    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    params = {"language": "ruby", "code_to_run": "puts 'hello'"}
    result_dict = await tool.execute(params, mock_key_provider_for_code_exec)

    assert result_dict["error"] == "No suitable executor."
    assert "No available executor found that supports language 'ruby'" in result_dict["stderr"]


@pytest.mark.asyncio
async def test_execute_requested_executor_not_found(
    mock_plugin_manager_for_code_exec: PluginManager,
    mock_key_provider_for_code_exec: KeyProvider
):
    """Test when a requested executor ID does not exist."""
    # Ensure list_discovered_plugin_classes is empty or get_plugin_instance returns None for the ID
    mock_plugin_manager_for_code_exec.list_discovered_plugin_classes.return_value = {}
    mock_plugin_manager_for_code_exec.get_plugin_instance.return_value = None

    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    params = {"language": "python", "code_to_run": "...", "executor_id": "non_existent_executor"}
    result_dict = await tool.execute(params, mock_key_provider_for_code_exec)

    assert result_dict["error"] == "Executor not found."
    assert "Requested executor 'non_existent_executor' not found or not available." in result_dict["stderr"]


@pytest.mark.asyncio
async def test_execute_requested_executor_does_not_support_language(
    mock_plugin_manager_for_code_exec: PluginManager,
    mock_key_provider_for_code_exec: KeyProvider
):
    """Test when requested executor exists but doesn't support the language."""
    mock_js_executor = MockCodeExecutor("js_exec", "js_sandbox", "JS", ["javascript"], CodeExecutionResult("", "", None, None, 1.0))
    mock_plugin_manager_for_code_exec.list_discovered_plugin_classes.return_value = {"js_exec": type(mock_js_executor)}
    mock_plugin_manager_for_code_exec.get_plugin_instance.return_value = mock_js_executor

    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    params = {"language": "python", "code_to_run": "...", "executor_id": "js_sandbox"}
    result_dict = await tool.execute(params, mock_key_provider_for_code_exec)

    assert result_dict["error"] == "Executor language mismatch."
    assert "Requested executor 'js_sandbox' does not support language 'python'" in result_dict["stderr"]


@pytest.mark.asyncio
async def test_execute_executor_itself_raises_error(
    mock_plugin_manager_for_code_exec: PluginManager,
    mock_key_provider_for_code_exec: KeyProvider
):
    """Test when the chosen executor's execute_code method raises an unhandled exception."""
    mock_erroring_executor = AsyncMock(spec=CodeExecutor)
    # Manually set attributes required by the protocol if spec doesn't cover them for isinstance checks in tool
    mock_erroring_executor.plugin_id = "error_exec"
    mock_erroring_executor.executor_id = "error_sandbox"
    mock_erroring_executor.description = "Errors"
    mock_erroring_executor.supported_languages = ["python"]
    # Ensure setup is an async def
    mock_erroring_executor.setup = AsyncMock()
    mock_erroring_executor.teardown = AsyncMock()

    mock_erroring_executor.execute_code.side_effect = RuntimeError("Executor crashed internally!")

    mock_plugin_manager_for_code_exec.list_discovered_plugin_classes.return_value = {"error_exec": type(mock_erroring_executor)}
    mock_plugin_manager_for_code_exec.get_plugin_instance.return_value = mock_erroring_executor

    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    params = {"language": "python", "code_to_run": "..."}
    result_dict = await tool.execute(params, mock_key_provider_for_code_exec)

    assert "Critical error in execution process: Executor crashed internally!" in result_dict["error"]
    assert "Tool-level execution error: Executor crashed internally!" in result_dict["stderr"]


@pytest.mark.asyncio
async def test_teardown_calls_executor_teardown(mock_plugin_manager_for_code_exec: PluginManager):
    """Test that the tool's teardown calls teardown on its cached executors."""
    mock_executor = MockCodeExecutor("exec_plug", "exec_id", "Desc", ["python"], CodeExecutionResult("", "", None, None, 1.0))

    # Mock list_discovered_plugin_classes to include the class of mock_executor
    mock_plugin_manager_for_code_exec.list_discovered_plugin_classes.return_value = {"exec_plug": type(mock_executor)}
    # Mock get_plugin_instance to return the mock_executor instance
    mock_plugin_manager_for_code_exec.get_plugin_instance.return_value = mock_executor

    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    await tool._get_available_executors() # Populate cache
    assert mock_executor in tool._available_executors_cache # type: ignore

    await tool.teardown()
    assert mock_executor.teardown_called is True
    assert tool._available_executors_cache is None
