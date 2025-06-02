### tests/unit/code_executors/impl/test_secure_docker_executor.py
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.code_executors.impl.secure_docker_executor import (
    DEFAULT_BASH_IMAGE,
    DEFAULT_NODE_IMAGE,
    DEFAULT_PYTHON_IMAGE,
    SecureDockerExecutor,
)


# --- Mock Docker Error Types ---
class DockerAPIErrorMock(Exception): pass
class DockerContainerErrorMock(Exception):
    def __init__(self, message, exit_status=1, command=None, image=None, stderr_bytes=b"mocked container error bytes"):
        super().__init__(message)
        self.exit_status = exit_status
        self.command = command
        self.image = image
        self.stderr = stderr_bytes
class DockerImageNotFoundMock(Exception): pass


@pytest.fixture
def mock_docker_client_fixture() -> MagicMock:
    client = MagicMock(name="MockDockerClientInstance")
    client.ping = MagicMock(return_value=True)
    client.images = MagicMock()
    client.images.pull = MagicMock()
    client.containers = MagicMock()
    client.containers.run = MagicMock(return_value=MagicMock(name="DefaultMockContainer"))
    client.close = MagicMock()
    return client

@pytest.fixture
def secure_docker_executor(mock_docker_client_fixture: MagicMock) -> SecureDockerExecutor: # Changed to sync fixture
    """
    Provides a SecureDockerExecutor instance that is set up.
    The docker module and DOCKER_AVAILABLE are patched for its setup.
    """
    mock_docker_client_fixture.containers.run.side_effect = None
    mock_docker_client_fixture.containers.run.return_value = MagicMock(name="DefaultMockContainerForFixtureSetup")

    with patch("genie_tooling.code_executors.impl.secure_docker_executor.DOCKER_AVAILABLE", True), \
         patch("genie_tooling.code_executors.impl.secure_docker_executor.docker") as mock_docker_module_in_executor_code:

        mock_docker_module_in_executor_code.from_env = MagicMock(return_value=mock_docker_client_fixture)
        mock_docker_module_in_executor_code.errors = MagicMock()
        mock_docker_module_in_executor_code.errors.APIError = DockerAPIErrorMock
        mock_docker_module_in_executor_code.errors.ContainerError = DockerContainerErrorMock
        mock_docker_module_in_executor_code.errors.ImageNotFound = DockerImageNotFoundMock

        executor = SecureDockerExecutor()
        # Run the async setup method using asyncio.run() because this fixture is synchronous
        asyncio.run(executor.setup())
    return executor


# --- Tests ---

@pytest.mark.asyncio
async def test_setup_docker_not_available(caplog: pytest.LogCaptureFixture):
    executor = SecureDockerExecutor()
    with patch("genie_tooling.code_executors.impl.secure_docker_executor.DOCKER_AVAILABLE", False):
        await executor.setup()
    assert executor._docker_client is None
    assert "Docker SDK not available. Executor disabled." in caplog.text

@pytest.mark.asyncio
async def test_setup_docker_client_init_fails(caplog: pytest.LogCaptureFixture):
    executor = SecureDockerExecutor()
    with patch("genie_tooling.code_executors.impl.secure_docker_executor.DOCKER_AVAILABLE", True), \
         patch("genie_tooling.code_executors.impl.secure_docker_executor.docker") as mock_docker_module_in_executor_code:
        mock_docker_module_in_executor_code.from_env.side_effect = Exception("Docker daemon down")
        await executor.setup()
    assert executor._docker_client is None
    assert "Failed to initialize Docker client: Docker daemon down" in caplog.text

@pytest.mark.asyncio
async def test_setup_pull_images_on_setup(mock_docker_client_fixture: MagicMock):
    with patch("genie_tooling.code_executors.impl.secure_docker_executor.DOCKER_AVAILABLE", True), \
         patch("genie_tooling.code_executors.impl.secure_docker_executor.docker") as mock_docker_module_in_executor_code:
        mock_docker_module_in_executor_code.from_env = MagicMock(return_value=mock_docker_client_fixture)
        mock_docker_module_in_executor_code.errors = MagicMock()
        mock_docker_module_in_executor_code.errors.APIError = DockerAPIErrorMock

        executor = SecureDockerExecutor()
        mock_docker_client_fixture.images.pull.reset_mock()
        await executor.setup(config={"pull_images_on_setup": True})

    mock_docker_client_fixture.images.pull.assert_any_call(DEFAULT_PYTHON_IMAGE)
    mock_docker_client_fixture.images.pull.assert_any_call(DEFAULT_NODE_IMAGE)
    mock_docker_client_fixture.images.pull.assert_any_call(DEFAULT_BASH_IMAGE)

@pytest.mark.asyncio
async def test_setup_pull_images_api_error(mock_docker_client_fixture: MagicMock, caplog: pytest.LogCaptureFixture):
    mock_docker_client_fixture.images.pull.side_effect = DockerAPIErrorMock("Pull failed")

    with patch("genie_tooling.code_executors.impl.secure_docker_executor.DOCKER_AVAILABLE", True), \
         patch("genie_tooling.code_executors.impl.secure_docker_executor.docker") as mock_docker_module_in_executor_code:
        mock_docker_module_in_executor_code.from_env = MagicMock(return_value=mock_docker_client_fixture)
        mock_docker_module_in_executor_code.errors = MagicMock()
        mock_docker_module_in_executor_code.errors.APIError = DockerAPIErrorMock

        executor = SecureDockerExecutor()
        await executor.setup(config={"pull_images_on_setup": True})

    assert f"Unexpected error pulling image {DEFAULT_PYTHON_IMAGE} for python: Pull failed" in caplog.text
    mock_docker_client_fixture.images.pull.side_effect = None


@pytest.mark.asyncio
async def test_execute_code_docker_client_not_initialized(secure_docker_executor: SecureDockerExecutor): # No await
    executor = secure_docker_executor
    executor._docker_client = None
    result = await executor.execute_code("python", "print('hi')", 10)
    assert result.error == "ExecutorSetupError"

@pytest.mark.asyncio
async def test_execute_code_unsupported_language(secure_docker_executor: SecureDockerExecutor): # No await
    executor = secure_docker_executor
    result = await executor.execute_code("ruby", "puts 'hello'", 10)
    assert result.error == "UnsupportedLanguage"

@pytest.mark.asyncio
async def test_execute_code_io_error_writing_script(secure_docker_executor: SecureDockerExecutor): # No await
    executor = secure_docker_executor
    with patch("pathlib.Path.write_text", side_effect=IOError("Disk full")):
        result = await executor.execute_code("python", "print('hi')", 10)
    assert result.error == "FileIOError"

@pytest.mark.asyncio
async def test_execute_code_other_setup_error(secure_docker_executor: SecureDockerExecutor): # No await
    executor = secure_docker_executor
    with patch("os.chmod", side_effect=Exception("chmod failed")):
        result = await executor.execute_code("python", "print('hi')", 10)
    assert result.error == "SetupError"


@pytest.mark.asyncio
@patch("asyncio.get_event_loop")
async def test_execute_code_success_python(mock_get_loop: MagicMock, secure_docker_executor: SecureDockerExecutor, mock_docker_client_fixture: MagicMock): # No await
    executor = secure_docker_executor
    assert executor._docker_client is mock_docker_client_fixture

    mock_container = MagicMock(name="MockContainer")
    mock_container.wait = MagicMock(return_value={"StatusCode": 0})
    mock_container.logs = MagicMock(side_effect=[b"output from python", b""])
    mock_container.stop = MagicMock()
    mock_container.remove = MagicMock()
    mock_docker_client_fixture.containers.run.return_value = mock_container
    mock_docker_client_fixture.containers.run.side_effect = None

    mock_event_loop = AsyncMock()
    mock_event_loop.run_in_executor = AsyncMock(return_value={"StatusCode": 0})
    mock_event_loop.time = MagicMock(side_effect=[10.0, 20.0])
    mock_get_loop.return_value = mock_event_loop

    result = await executor.execute_code("python", "print('output from python')", 10)
    assert result.stdout == "output from python"
    assert result.execution_time_ms == (20.0 - 10.0) * 1000

@pytest.mark.asyncio
@patch("asyncio.get_event_loop")
async def test_execute_code_timeout(mock_get_loop: MagicMock, secure_docker_executor: SecureDockerExecutor, mock_docker_client_fixture: MagicMock): # No await
    executor = secure_docker_executor
    assert executor._docker_client is mock_docker_client_fixture

    mock_container = MagicMock(name="MockContainerTimeout")
    mock_container.stop = MagicMock()
    mock_container.remove = MagicMock()
    mock_docker_client_fixture.containers.run.return_value = mock_container
    mock_docker_client_fixture.containers.run.side_effect = None

    mock_event_loop = AsyncMock()
    mock_event_loop.run_in_executor = AsyncMock(side_effect=asyncio.TimeoutError)
    mock_event_loop.time = MagicMock(side_effect=[5.0, 8.0])
    mock_get_loop.return_value = mock_event_loop

    result = await executor.execute_code("python", "import time; time.sleep(5)", 1)
    assert result.error == "Timeout"

@pytest.mark.asyncio
@patch("asyncio.get_event_loop")
async def test_execute_code_container_error(mock_get_loop: MagicMock, secure_docker_executor: SecureDockerExecutor, mock_docker_client_fixture: MagicMock): # No await
    executor = secure_docker_executor
    assert executor._docker_client is mock_docker_client_fixture

    mock_container = MagicMock(name="MockContainerError")
    mock_container.remove = MagicMock()
    mock_docker_client_fixture.containers.run.return_value = mock_container
    mock_docker_client_fixture.containers.run.side_effect = None

    mock_event_loop = AsyncMock()
    container_error_exception = DockerContainerErrorMock(
        message="Container exited with non-zero",
        exit_status=1,
        stderr_bytes=b"container error output from exception"
    )
    mock_event_loop.run_in_executor = AsyncMock(side_effect=container_error_exception)
    mock_event_loop.time = MagicMock(side_effect=[1.0, 2.0])
    mock_get_loop.return_value = mock_event_loop

    result = await executor.execute_code("python", "exit(1)", 10)

    assert "WaitError: " in result.error # type: ignore
    assert "Container exited with non-zero" in result.error # type: ignore
    assert "container error output from exception" in result.stderr

@pytest.mark.asyncio
async def test_execute_code_image_not_found(secure_docker_executor: SecureDockerExecutor, mock_docker_client_fixture: MagicMock): # No await
    executor = secure_docker_executor
    assert executor._docker_client is mock_docker_client_fixture
    with patch("genie_tooling.code_executors.impl.secure_docker_executor.DockerImageNotFound", DockerImageNotFoundMock):
        mock_docker_client_fixture.containers.run.side_effect = DockerImageNotFoundMock(f"Image {DEFAULT_PYTHON_IMAGE} not found")
        result = await executor.execute_code("python", "print('hi')", 10)
    assert f"DockerImageNotFound: Image '{DEFAULT_PYTHON_IMAGE}' not found." in result.error # type: ignore
    mock_docker_client_fixture.containers.run.side_effect = None

@pytest.mark.asyncio
async def test_execute_code_docker_api_error_on_run(secure_docker_executor: SecureDockerExecutor, mock_docker_client_fixture: MagicMock): # No await
    executor = secure_docker_executor
    assert executor._docker_client is mock_docker_client_fixture
    with patch("genie_tooling.code_executors.impl.secure_docker_executor.DockerAPIError", DockerAPIErrorMock), \
         patch("genie_tooling.code_executors.impl.secure_docker_executor.DockerImageNotFound", type("OtherINF", (Exception,), {})):
        mock_docker_client_fixture.containers.run.side_effect = DockerAPIErrorMock("Permission denied")
        result = await executor.execute_code("python", "print('hi')", 10)
    assert "DockerAPIError: Permission denied" in result.error # type: ignore
    mock_docker_client_fixture.containers.run.side_effect = None

@pytest.mark.asyncio
async def test_execute_code_general_docker_error_on_run(secure_docker_executor: SecureDockerExecutor, mock_docker_client_fixture: MagicMock): # No await
    executor = secure_docker_executor
    assert executor._docker_client is mock_docker_client_fixture
    with patch("genie_tooling.code_executors.impl.secure_docker_executor.DockerAPIError", type("OtherAPI", (Exception,), {})), \
         patch("genie_tooling.code_executors.impl.secure_docker_executor.DockerImageNotFound", type("OtherINF", (Exception,), {})):
        mock_docker_client_fixture.containers.run.side_effect = Exception("Some other docker error")
        result = await executor.execute_code("python", "print('hi')", 10)
    assert "GeneralDockerError: Some other docker error" in result.error # type: ignore
    mock_docker_client_fixture.containers.run.side_effect = None

@pytest.mark.asyncio
@patch("asyncio.get_event_loop")
async def test_execute_code_container_remove_fails(mock_get_loop: MagicMock, secure_docker_executor: SecureDockerExecutor, mock_docker_client_fixture: MagicMock, caplog: pytest.LogCaptureFixture): # No await
    executor = secure_docker_executor
    assert executor._docker_client is mock_docker_client_fixture

    mock_container = MagicMock(name="MockContainerRemoveFail")
    mock_container.wait = MagicMock(return_value={"StatusCode": 0})
    mock_container.logs = MagicMock(return_value=b"output")
    with patch("genie_tooling.code_executors.impl.secure_docker_executor.DockerAPIError", DockerAPIErrorMock):
        mock_container.remove = MagicMock(side_effect=DockerAPIErrorMock("Failed to remove"))
        mock_docker_client_fixture.containers.run.return_value = mock_container
        mock_docker_client_fixture.containers.run.side_effect = None

        mock_event_loop = AsyncMock()
        mock_event_loop.run_in_executor = AsyncMock(return_value={"StatusCode": 0})
        mock_event_loop.time = MagicMock(side_effect=[1.0, 2.0])
        mock_get_loop.return_value = mock_event_loop

        await executor.execute_code("python", "print('hi')", 10)
    assert "Failed to remove container" in caplog.text
    assert "Failed to remove" in caplog.text

@pytest.mark.asyncio
async def test_teardown_client_none():
    executor = SecureDockerExecutor()
    executor._docker_client = None
    await executor.teardown()

@pytest.mark.asyncio
async def test_teardown_client_close_error(secure_docker_executor: SecureDockerExecutor, mock_docker_client_fixture: MagicMock, caplog: pytest.LogCaptureFixture): # No await
    executor = secure_docker_executor
    assert executor._docker_client is mock_docker_client_fixture
    mock_docker_client_fixture.close.side_effect = Exception("Close failed")
    await executor.teardown()
    assert "Error closing Docker client: Close failed" in caplog.text
    assert executor._docker_client is None
