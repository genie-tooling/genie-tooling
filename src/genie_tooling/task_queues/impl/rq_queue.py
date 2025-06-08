import asyncio
import functools
import logging
from typing import Any, Dict, Optional, Tuple

from genie_tooling.task_queues.abc import DistributedTaskQueuePlugin, TaskStatus

logger = logging.getLogger(__name__)

RQ_AVAILABLE = False
Job = None
Queue = None
Redis = None
NoSuchJobError = None
Retry = None
send_stop_job_command = None

try:
    from redis import Redis as SyncRedis
    from rq import Queue as RQQueue
    from rq.command import send_stop_job_command as rq_send_stop_job_command
    from rq.exceptions import NoSuchJobError as RQNSError
    from rq.job import Job as RQJob
    from rq.retry import Retry as RQRetry
    RQ_AVAILABLE = True
    Job = RQJob # type: ignore
    Queue = RQQueue # type: ignore
    Redis = SyncRedis # type: ignore
    NoSuchJobError = RQNSError # type: ignore
    Retry = RQRetry # type: ignore
    send_stop_job_command = rq_send_stop_job_command
except ImportError:
    logger.warning(
        "RedisQueueTaskPlugin: 'rq' or 'redis' library not installed. "
        "This plugin will not be functional. Please install them: poetry add rq redis"
    )


class RedisQueueTaskPlugin(DistributedTaskQueuePlugin):
    plugin_id: str = "redis_queue_task_plugin_v1"
    description: str = "Integrates with RQ (Redis Queue) for distributed tasks."

    _redis_conn: Optional[Any] = None
    _queues: Dict[str, Any] = {}
    _default_queue_name: str = "default"
    _redis_url: Optional[str] = None # Added for robust setup

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not RQ_AVAILABLE or not Queue or not Redis or not Job or not NoSuchJobError:
            logger.error(f"{self.plugin_id}: RQ or Redis library not available. Cannot initialize.")
            self._redis_conn = None
            self._queues = {}
            return
        cfg = config or {}
        self._redis_url = cfg.get("redis_url") # No default here
        if not self._redis_url:
            logger.info(f"{self.plugin_id}: 'redis_url' for RQ not configured. Plugin will be disabled.")
            self._redis_conn = None
            self._queues = {}
            return

        self._default_queue_name = cfg.get("default_queue_name", "default")

        try:
            self._redis_conn = Redis.from_url(self._redis_url) # type: ignore
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._redis_conn.ping) # type: ignore
            self._queues[self._default_queue_name] = Queue(self._default_queue_name, connection=self._redis_conn) # type: ignore
            logger.info(f"{self.plugin_id}: Connected to Redis for RQ at {self._redis_url}. Default queue: '{self._default_queue_name}'.")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to connect to Redis or initialize default RQ queue: {e}", exc_info=True)
            if self._redis_conn:
                try:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._redis_conn.close)
                except Exception: # nosec
                    pass
            self._redis_conn = None
            self._queues = {}

    def _get_rq_queue(self, queue_name: Optional[str] = None) -> Optional[Any]: # rq.Queue
        if not self._redis_conn or not Queue:
            logger.debug(f"{self.plugin_id}: Cannot get RQ queue, Redis connection or RQ.Queue not available.")
            return None
        target_queue_name = queue_name or self._default_queue_name
        if target_queue_name not in self._queues:
            try:
                self._queues[target_queue_name] = Queue(target_queue_name, connection=self._redis_conn) # type: ignore
            except Exception as e:
                logger.error(f"{self.plugin_id}: Failed to initialize RQ queue '{target_queue_name}': {e}", exc_info=True)
                return None
        return self._queues.get(target_queue_name)

    async def submit_task(
        self, task_name: str, args: Tuple = (), kwargs: Optional[Dict[str, Any]] = None,
        queue_name: Optional[str] = None, task_options: Optional[Dict[str, Any]] = None
    ) -> str:
        if not self._redis_conn:
            raise RuntimeError(f"{self.plugin_id}: Redis connection not available for task submission.")
        rq_queue = self._get_rq_queue(queue_name)
        if not rq_queue:
            raise RuntimeError(f"{self.plugin_id}: RQ queue '{queue_name or self._default_queue_name}' could not be initialized.")

        task_actual_kwargs = kwargs or {}
        options_for_enqueue = task_options or {}
        enqueue_specific_kwargs = {}
        if options_for_enqueue.get("job_timeout", options_for_enqueue.get("timeout")) is not None:
            enqueue_specific_kwargs["job_timeout"] = options_for_enqueue.get("job_timeout", options_for_enqueue.get("timeout"))
        if options_for_enqueue.get("retry") is not None:
            enqueue_specific_kwargs["retry"] = options_for_enqueue.get("retry")
        if options_for_enqueue.get("description", f"Genie Task: {task_name}") is not None:
            enqueue_specific_kwargs["description"] = options_for_enqueue.get("description", f"Genie Task: {task_name}")
        try:
            loop = asyncio.get_running_loop()
            partial_enqueue_call = functools.partial(rq_queue.enqueue, task_name, *args, **task_actual_kwargs, **enqueue_specific_kwargs) # type: ignore
            job = await loop.run_in_executor(None, partial_enqueue_call)
            if not job or not job.id:
                raise RuntimeError("RQ enqueue did not return a job with an ID.")
            return job.id
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error submitting task '{task_name}' to RQ: {e}", exc_info=True)
            raise

    async def get_task_status(self, task_id: str, queue_name: Optional[str] = None) -> TaskStatus:
        if not self._redis_conn or not Job or not NoSuchJobError:
            logger.debug(f"{self.plugin_id}: Cannot get task status, Redis/RQ components not available.")
            return "unknown"
        try:
            loop = asyncio.get_running_loop()
            partial_fetch = functools.partial(Job.fetch, task_id, connection=self._redis_conn) # type: ignore
            job = await loop.run_in_executor(None, partial_fetch)
            partial_get_status = functools.partial(job.get_status, refresh=True) # type: ignore
            job_status_str = await loop.run_in_executor(None, partial_get_status)
            status_map: Dict[str, TaskStatus] = {"queued": "pending", "started": "running", "deferred": "pending", "finished": "success", "failed": "failure", "scheduled": "pending", "canceled": "revoked"}
            return status_map.get(job_status_str.lower(), "unknown")
        except NoSuchJobError: # type: ignore
            logger.debug(f"{self.plugin_id}: Task ID '{task_id}' not found in RQ.")
            return "unknown"
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error getting RQ task status for '{task_id}': {e}", exc_info=True)
            return "unknown"

    async def get_task_result(self, task_id: str, queue_name: Optional[str] = None, timeout_seconds: Optional[float] = None) -> Any:
        if not self._redis_conn or not Job or not NoSuchJobError:
            raise RuntimeError(f"{self.plugin_id}: Redis connection or RQ Job type not available for result fetching.")
        try:
            loop = asyncio.get_running_loop()
            partial_fetch = functools.partial(Job.fetch, task_id, connection=self._redis_conn) # type: ignore
            job = await loop.run_in_executor(None, partial_fetch)
            if timeout_seconds is not None and timeout_seconds > 0:
                start_time = asyncio.get_event_loop().time()
                while not await loop.run_in_executor(None, lambda j=job: j.is_finished) and not await loop.run_in_executor(None, lambda j=job: j.is_failed) and not await loop.run_in_executor(None, lambda j=job: j.is_canceled):
                    if (asyncio.get_event_loop().time() - start_time) > timeout_seconds:
                        raise TimeoutError(f"Timeout waiting for RQ task '{task_id}' result.")
                    await asyncio.sleep(0.1)
            if await loop.run_in_executor(None, lambda j=job: j.is_failed):
                exc_info = await loop.run_in_executor(None, lambda j=job: j.exc_info)
                raise RuntimeError(f"RQ task '{job.id}' failed. Info: {exc_info}") # type: ignore
            if await loop.run_in_executor(None, lambda j=job: j.is_canceled):
                raise RuntimeError(f"RQ task '{job.id}' was canceled.") # type: ignore
            return await loop.run_in_executor(None, lambda j=job: j.result)
        except NoSuchJobError:
            raise KeyError(f"RQ Task ID '{task_id}' not found.") from None # type: ignore
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error getting RQ task result for '{task_id}': {e}", exc_info=True)
            raise

    async def revoke_task(self, task_id: str, queue_name: Optional[str] = None, terminate: bool = False) -> bool:
        if not RQ_AVAILABLE or not self._redis_conn or not Job or not NoSuchJobError or not send_stop_job_command:
            logger.debug(f"{self.plugin_id}: Cannot revoke task, Redis/RQ components not available.")
            return False
        try:
            loop = asyncio.get_running_loop()
            partial_fetch = functools.partial(Job.fetch, task_id, connection=self._redis_conn) # type: ignore
            job = await loop.run_in_executor(None, partial_fetch)
            if await loop.run_in_executor(None, lambda j=job: j.is_started) and terminate: # type: ignore
                partial_stop_cmd = functools.partial(send_stop_job_command, self._redis_conn, job.id) # type: ignore
                await loop.run_in_executor(None, partial_stop_cmd)
                logger.info(f"{self.plugin_id}: Sent stop command for running RQ job '{task_id}'.")
                return True
            elif not await loop.run_in_executor(None, lambda j=job: j.is_finished) and not await loop.run_in_executor(None, lambda j=job: j.is_failed) and not await loop.run_in_executor(None, lambda j=job: j.is_canceled): # type: ignore
                await loop.run_in_executor(None, job.cancel)
                logger.info(f"{self.plugin_id}: Canceled queued/deferred RQ job '{task_id}'.")
                return True # type: ignore
            logger.info(f"{self.plugin_id}: RQ job '{task_id}' already finished, failed, or canceled. No action taken for revoke.")
            return True
        except NoSuchJobError:
            logger.warning(f"{self.plugin_id}: Task ID '{task_id}' not found for revoke. Considered successful.")
            return True # type: ignore
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error revoking RQ task '{task_id}': {e}", exc_info=True)
            return False

    async def teardown(self) -> None:
        if self._redis_conn:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._redis_conn.close) # type: ignore
            except Exception as e:
                logger.error(f"{self.plugin_id}: Error closing Redis connection for RQ: {e}", exc_info=True)
            finally:
                self._redis_conn = None
        self._queues.clear()
        logger.info(f"{self.plugin_id}: Teardown complete.")
