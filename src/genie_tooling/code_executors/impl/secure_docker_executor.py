"""
SecureDockerExecutor: Executes code within isolated Docker containers.
"""
import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from genie_tooling.code_executors.abc import CodeExecutionResult, CodeExecutor

logger = logging.getLogger(__name__)

try:
    import docker
    from docker.errors import APIError as DockerAPIError
    from docker.errors import ContainerError as DockerContainerError
    from docker.errors import ImageNotFound as DockerImageNotFound
    DOCKER_AVAILABLE = True
except ImportError:
    docker = None # type: ignore
    DockerAPIError = Exception # type: ignore
    _MockDockerContainerError_Base = type("DockerContainerError", (Exception,), {})
    class DockerContainerError(_MockDockerContainerError_Base): # type: ignore
        def __init__(self, message, exit_status=1, command=None, image=None, stderr_bytes=b"mocked container error bytes"):
            super().__init__(message)
            self.exit_status = exit_status
            self.command = command
            self.image = image
            self.stderr = stderr_bytes # Ensure this attribute exists

    DockerImageNotFound = Exception # type: ignore
    DOCKER_AVAILABLE = False
    logger.warning(
        "SecureDockerExecutor: 'docker' library not installed. "
        "This executor will not be functional. Please install it: pip install docker"
    )

DEFAULT_PYTHON_IMAGE = "python:3.11-slim"
DEFAULT_NODE_IMAGE = "node:20-slim"
DEFAULT_BASH_IMAGE = "bash:latest"

class SecureDockerExecutor(CodeExecutor):
    plugin_id: str = "secure_docker_executor_v1"
    executor_id: str = "secure_docker_executor_v1"
    description: str = (
        "Executes code in isolated Docker containers for enhanced security. "
        "Supports Python, JavaScript, and Bash by default."
    )
    supported_languages: List[str] = ["python", "javascript", "bash"]

    _docker_client: Optional[Any] = None
    _language_images: Dict[str, str]
    _default_network_mode: str = "none"
    _default_mem_limit: str = "128m"
    _default_cpu_shares: Optional[int] = None
    _default_pids_limit: int = 100

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not DOCKER_AVAILABLE:
            logger.error(f"{self.plugin_id}: Docker SDK not available. Executor disabled.")
            return

        cfg = config or {}
        try:
            self._docker_client = docker.from_env() # type: ignore
            self._docker_client.ping() # type: ignore
            logger.info(f"{self.plugin_id}: Docker client initialized and connected.")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize Docker client: {e}. Is Docker running?", exc_info=True)
            self._docker_client = None
            return

        self._language_images = {
            "python": cfg.get("python_docker_image", DEFAULT_PYTHON_IMAGE),
            "javascript": cfg.get("node_docker_image", DEFAULT_NODE_IMAGE),
            "bash": cfg.get("bash_docker_image", DEFAULT_BASH_IMAGE),
        }
        self._default_network_mode = cfg.get("default_network_mode", self._default_network_mode)
        self._default_mem_limit = cfg.get("default_mem_limit", self._default_mem_limit)
        self._default_cpu_shares = cfg.get("default_cpu_shares", self._default_cpu_shares)
        self._default_pids_limit = int(cfg.get("default_pids_limit", self._default_pids_limit))

        pull_images_on_setup = cfg.get("pull_images_on_setup", False)
        if pull_images_on_setup:
            logger.info(f"{self.plugin_id}: Attempting to pull configured Docker images...")
            for lang, image_name in self._language_images.items():
                try:
                    logger.info(f"Pulling image for {lang}: {image_name}")
                    self._docker_client.images.pull(image_name) # type: ignore
                except DockerAPIError as e_pull: # type: ignore
                    logger.warning(f"Failed to pull image {image_name} for {lang}: {e_pull}")
                except Exception as e_gen_pull:
                    logger.warning(f"Unexpected error pulling image {image_name} for {lang}: {e_gen_pull}")


    async def execute_code(
        self,
        language: str,
        code: str,
        timeout_seconds: int,
        input_data: Optional[Dict[str, Any]] = None
    ) -> CodeExecutionResult:
        if not self._docker_client:
            return CodeExecutionResult("", "Docker client not initialized.", None, "ExecutorSetupError", 0.0)

        lang_lower = language.lower()
        image_name = self._language_images.get(lang_lower)
        if not image_name:
            return CodeExecutionResult("", f"Language '{language}' not supported.", None, "UnsupportedLanguage", 0.0)

        temp_dir_host_str: Optional[str] = None
        try:
            temp_dir_host_str = tempfile.mkdtemp(prefix="genie_docker_exec_")
            temp_dir_host = Path(temp_dir_host_str)
            temp_dir_container = "/app"

            script_filename: str
            command_to_run: List[str]

            if lang_lower == "python":
                script_filename = "script.py"
                command_to_run = ["python", f"{temp_dir_container}/{script_filename}"]
            elif lang_lower == "javascript":
                script_filename = "script.js"
                command_to_run = ["node", f"{temp_dir_container}/{script_filename}"]
            elif lang_lower == "bash":
                script_filename = "script.sh"
                command_to_run = ["/bin/bash", f"{temp_dir_container}/{script_filename}"]
            else:
                return CodeExecutionResult("", f"Internal error: Language '{language}' command setup failed.", None, "InternalError", 0.0)

            host_script_path = temp_dir_host / script_filename
            host_script_path.write_text(code, encoding="utf-8")
            os.chmod(host_script_path, 0o755)

        except IOError as e_io:
            if temp_dir_host_str:
                shutil.rmtree(temp_dir_host_str, ignore_errors=True)
            return CodeExecutionResult("", f"Failed to write code to temporary file or set permissions: {e_io}", None, "FileIOError", 0.0)
        except Exception as e_setup:
            if temp_dir_host_str:
                shutil.rmtree(temp_dir_host_str, ignore_errors=True)
            return CodeExecutionResult("", f"Unexpected error during script setup: {e_setup}", None, "SetupError", 0.0)

        container_name = f"genie-exec-{lang_lower}-{uuid.uuid4().hex[:8]}"
        volumes_map = {str(temp_dir_host): {"bind": temp_dir_container, "mode": "rw"}}

        container_run_params: Dict[str, Any] = {
            "image": image_name, "command": command_to_run, "name": container_name,
            "volumes": volumes_map, "working_dir": temp_dir_container, "detach": True,
            "network_mode": self._default_network_mode, "mem_limit": self._default_mem_limit,
            "pids_limit": self._default_pids_limit,
            "security_opt": ["no-new-privileges"], "cap_drop": ["ALL"],
        }
        if self._default_cpu_shares is not None:
            container_run_params["cpu_shares"] = self._default_cpu_shares

        start_time = asyncio.get_event_loop().time()
        container = None
        stdout_str, stderr_str, exec_error = "", "", None

        try:
            logger.debug(f"Running container '{container_name}' with image '{image_name}'. Cmd: {command_to_run}")
            container = self._docker_client.containers.run(**container_run_params) # type: ignore

            async def wait_for_container_async(cont, timeout_val):
                return await asyncio.get_event_loop().run_in_executor(None, cont.wait, timeout=timeout_val)

            try:
                await asyncio.wait_for(wait_for_container_async(container, timeout_seconds), timeout=timeout_seconds + 2.0)
            except asyncio.TimeoutError:
                logger.warning(f"Container '{container_name}' execution timed out after ~{timeout_seconds}s. Stopping.")
                if container:
                    try: container.stop(timeout=5)
                    except Exception as e_stop: logger.warning(f"Error stopping timed-out container {container_name}: {e_stop}")
                exec_error = "Timeout"
            except DockerContainerError as e_cont_err: # type: ignore
                logger.warning(f"Container '{container_name}' exited with error: {e_cont_err.exit_status}. Stderr: {e_cont_err.stderr.decode('utf-8', 'replace') if hasattr(e_cont_err, 'stderr') and e_cont_err.stderr else 'N/A'}") # type: ignore
                if hasattr(e_cont_err, "stderr") and e_cont_err.stderr: # type: ignore
                    stderr_str += e_cont_err.stderr.decode("utf-8", "replace") # type: ignore
                exec_error = f"ContainerError (Exit Code: {e_cont_err.exit_status})" # type: ignore
            except Exception as e_wait_other:
                logger.error(f"Error waiting for container '{container_name}': {e_wait_other}", exc_info=True)
                exec_error = f"WaitError: {str(e_wait_other)}"
                # Check if the wrapped exception has stderr (like DockerContainerError)
                if hasattr(e_wait_other, "stderr") and e_wait_other.stderr and isinstance(e_wait_other.stderr, bytes): # type: ignore
                    stderr_str += e_wait_other.stderr.decode("utf-8", "replace") # type: ignore


            if container and not exec_error:
                stdout_bytes = container.logs(stdout=True, stderr=False)
                stderr_bytes = container.logs(stdout=False, stderr=True)
                stdout_str = stdout_bytes.decode("utf-8", "replace") if stdout_bytes else ""
                stderr_str += stderr_bytes.decode("utf-8", "replace") if stderr_bytes else ""

        except DockerImageNotFound: # type: ignore
            exec_error = f"DockerImageNotFound: Image '{image_name}' not found."
            logger.error(f"{self.plugin_id}: {exec_error}")
        except DockerAPIError as e_api: # type: ignore
            exec_error = f"DockerAPIError: {str(e_api)}"
            logger.error(f"{self.plugin_id}: {exec_error}", exc_info=True)
        except Exception as e_general_docker:
            exec_error = f"GeneralDockerError: {str(e_general_docker)}"
            logger.error(f"{self.plugin_id}: {exec_error}", exc_info=True)
        finally:
            if container:
                try: container.remove(force=True)
                except DockerAPIError as e_remove: logger.warning(f"Failed to remove container '{container_name}': {e_remove}") # type: ignore
            if temp_dir_host_str:
                shutil.rmtree(temp_dir_host_str, ignore_errors=True)

        end_time = asyncio.get_event_loop().time()
        execution_time_ms = (end_time - start_time) * 1000
        return CodeExecutionResult(stdout_str, stderr_str, None, exec_error, execution_time_ms)

    async def teardown(self) -> None:
        if self._docker_client:
            try:
                self._docker_client.close() # type: ignore
                logger.info(f"{self.plugin_id}: Docker client closed.")
            except Exception as e:
                logger.error(f"{self.plugin_id}: Error closing Docker client: {e}", exc_info=True)
        self._docker_client = None
