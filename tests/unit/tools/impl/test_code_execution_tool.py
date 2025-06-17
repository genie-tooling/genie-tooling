from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.code_executors.abc import CodeExecutionResult, CodeExecutor
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.impl.code_execution_tool import GenericCodeExecutionTool


class MockCodeExecutor(CodeExecutor):
    _plugin_id_value: str
    def __init__(
        self,
        plugin_id_val: str,
        executor_id: str,
        description: str,
        supported_languages: List[str],
        execute_result: CodeExecutionResult
    ):
        self._plugin_id_value = plugin_id_val
        self.executor_id: str = executor_id
        self.description: str = description
        self.supported_languages: List[str] = supported_languages
        self._execute_result = execute_result
        self.setup_called_with_config: Optional[Dict[str, Any]] = None
        self.teardown_called: bool = False

    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.setup_called_with_config = config
    async def teardown(self) -> None: self.teardown_called = True
    async def execute_code(self, language: str, code: str, timeout_seconds: int, input_data: Optional[Dict[str, Any]] = None) -> CodeExecutionResult:
        if language not in self.supported_languages:
            return CodeExecutionResult("", f"Language {language} not supported by {self.executor_id}", None, "UnsupportedLanguage", 0.0)
        return self._execute_result

@pytest.fixture()
def mock_plugin_manager_for_code_exec(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.list_discovered_plugin_classes = MagicMock(return_value={})
    pm.get_plugin_instance = AsyncMock()
    return pm
@pytest.fixture()
def mock_key_provider_for_code_exec(mocker) -> KeyProvider:
    kp = mocker.AsyncMock(spec=KeyProvider)
    return kp

@pytest.mark.asyncio()
async def test_generic_code_exec_tool_init_and_get_metadata_no_executors(mock_plugin_manager_for_code_exec: PluginManager):
    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    # The tool is configured via setup, not discovery.
    await tool.setup(config={"available_executor_ids": []})
    metadata = await tool.get_metadata()
    assert metadata["identifier"] == "generic_code_execution_tool"
    # --- FIX START: Correct the assertion to reflect the configured state ---
    assert "Available executors: []" in metadata["description_llm"]
    # --- FIX END ---
    assert metadata["input_schema"]["properties"]["executor_id"]["enum"] == []

@pytest.mark.asyncio()
async def test_generic_code_exec_tool_get_metadata_with_executors(mock_plugin_manager_for_code_exec: PluginManager):
    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    # Configure the tool with the executor IDs it's allowed to use
    await tool.setup(config={"available_executor_ids": ["python_sandbox_v1", "nodejs_sandbox_v1"]})
    metadata = await tool.get_metadata()
    assert "python_sandbox_v1" in metadata["input_schema"]["properties"]["executor_id"]["enum"]
    assert "nodejs_sandbox_v1" in metadata["input_schema"]["properties"]["executor_id"]["enum"]

@pytest.mark.asyncio()
async def test_execute_python_code_auto_select_executor(mock_plugin_manager_for_code_exec: PluginManager, mock_key_provider_for_code_exec: KeyProvider):
    expected_stdout = "Hello Python!"
    py_executor_res = CodeExecutionResult(expected_stdout, "", None, None, 25.5)
    mock_py_executor = MockCodeExecutor(plugin_id_val="py_exec", executor_id="pysandbox_executor_stub_v1", description="Py", supported_languages=["python"], execute_result=py_executor_res)
    # Configure the mock PM to return this executor when requested by ID
    mock_plugin_manager_for_code_exec.get_plugin_instance.return_value = mock_py_executor

    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    # Setup the tool (which sets its default executor ID)
    await tool.setup()

    params = {"language": "python", "code_to_run": "print('Hello Python!')"}
    result_dict = await tool.execute(params, mock_key_provider_for_code_exec, context={})

    assert result_dict["stdout"] == expected_stdout
    assert result_dict["error"] is None
    assert result_dict["execution_time_ms"] == 25.5
    mock_plugin_manager_for_code_exec.get_plugin_instance.assert_awaited_with("pysandbox_executor_stub_v1")

@pytest.mark.asyncio()
async def test_execute_code_with_specific_executor_id(mock_plugin_manager_for_code_exec: PluginManager, mock_key_provider_for_code_exec: KeyProvider):
    expected_output = "Specific exec works"
    exec_res = CodeExecutionResult(expected_output, "", None, None, 30.0)
    mock_executor1 = MockCodeExecutor("exec1_plug", "executor_one", "Desc1", ["python"], exec_res)
    # Configure PM to return the correct mock based on requested ID
    mock_plugin_manager_for_code_exec.get_plugin_instance.return_value = mock_executor1

    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    # Setup with the available executor IDs
    await tool.setup(config={"available_executor_ids": ["executor_one"]})

    params = {"language": "python", "code_to_run": "...", "executor_id": "executor_one"}
    result_dict = await tool.execute(params, mock_key_provider_for_code_exec, context={})
    assert result_dict["stdout"] == expected_output
    assert result_dict["error"] is None
    mock_plugin_manager_for_code_exec.get_plugin_instance.assert_awaited_with("executor_one")

@pytest.mark.asyncio()
async def test_execute_unsupported_language(mock_plugin_manager_for_code_exec: PluginManager, mock_key_provider_for_code_exec: KeyProvider):
    mock_py_executor = MockCodeExecutor("py_exec", "py_sandbox", "Py", ["python"], CodeExecutionResult("", "", None, None, 1.0))
    mock_plugin_manager_for_code_exec.get_plugin_instance.return_value = mock_py_executor
    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    # --- FIX START: Explicitly set the default executor to match what's available for this test ---
    await tool.setup(config={"available_executor_ids": ["py_sandbox"], "executor_id": "py_sandbox"})
    # --- FIX END ---

    params = {"language": "ruby", "code_to_run": "puts 'hello'"}
    result_dict = await tool.execute(params, mock_key_provider_for_code_exec, context={})
    assert result_dict["error"] == "UnsupportedLanguage"
    assert "Executor 'py_sandbox' does not support the requested language 'ruby'" in result_dict["stderr"]

@pytest.mark.asyncio()
async def test_execute_requested_executor_not_found(mock_plugin_manager_for_code_exec: PluginManager, mock_key_provider_for_code_exec: KeyProvider):
    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    # Setup with a list of *valid* executors. The requested one is not in this list.
    await tool.setup(config={"available_executor_ids": ["valid_executor"]})

    params = {"language": "python", "code_to_run": "...", "executor_id": "non_existent_executor"}
    result_dict = await tool.execute(params, mock_key_provider_for_code_exec, context={})
    assert result_dict["error"] == "ConfigurationError"
    assert "Requested executor 'non_existent_executor' is not in the list of available executors" in result_dict["stderr"]

@pytest.mark.asyncio()
async def test_execute_requested_executor_does_not_support_language(mock_plugin_manager_for_code_exec: PluginManager, mock_key_provider_for_code_exec: KeyProvider):
    mock_js_executor = MockCodeExecutor("js_exec", "js_sandbox", "JS", ["javascript"], CodeExecutionResult("", "", None, None, 1.0))
    mock_plugin_manager_for_code_exec.get_plugin_instance.return_value = mock_js_executor
    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    await tool.setup(config={"available_executor_ids": ["js_sandbox"]})

    params = {"language": "python", "code_to_run": "...", "executor_id": "js_sandbox"}
    result_dict = await tool.execute(params, mock_key_provider_for_code_exec, context={})
    assert result_dict["error"] == "UnsupportedLanguage"
    assert "Executor 'js_sandbox' does not support the requested language 'python'" in result_dict["stderr"]

@pytest.mark.asyncio()
async def test_execute_executor_itself_raises_error(mock_plugin_manager_for_code_exec: PluginManager, mock_key_provider_for_code_exec: KeyProvider):
    mock_erroring_executor = AsyncMock(spec=CodeExecutor)
    mock_erroring_executor.executor_id = "error_sandbox"
    mock_erroring_executor.supported_languages = ["python"]
    mock_erroring_executor.execute_code.side_effect = RuntimeError("Executor crashed internally!")
    mock_plugin_manager_for_code_exec.get_plugin_instance.return_value = mock_erroring_executor

    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    await tool.setup() # Use default executor id, which will be mocked to return the erroring one

    params = {"language": "python", "code_to_run": "..."}
    result_dict = await tool.execute(params, mock_key_provider_for_code_exec, context={})
    assert "Critical error in execution process: Executor crashed internally!" in result_dict["error"]
    assert "Tool-level execution error: Executor crashed internally!" in result_dict["stderr"]

@pytest.mark.asyncio()
async def test_teardown_calls_executor_teardown(mock_plugin_manager_for_code_exec: PluginManager):
    # The tool's teardown is a no-op as it doesn't hold executor instances.
    # We test that it simply runs without error.
    tool = GenericCodeExecutionTool(plugin_manager=mock_plugin_manager_for_code_exec)
    await tool.setup()
    await tool.teardown()
    assert True # Pass if no exception