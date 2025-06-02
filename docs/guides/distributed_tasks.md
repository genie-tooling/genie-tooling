# Using Distributed Task Queues

Genie Tooling supports offloading tasks, such as long-running tool executions, to distributed task queues like Celery or RQ. This allows your main agent loop to remain responsive.

## Core Concepts

*   **`TaskQueueInterface` (`genie.task_queue`)**: Facade for submitting and managing distributed tasks.
*   **`DistributedTaskQueuePlugin`**: Plugin for a specific task queue system (e.g., Celery, RQ).
    *   Built-in: `CeleryTaskQueuePlugin` (alias: `celery_task_queue`), `RedisQueueTaskPlugin` (alias: `rq_task_queue` - currently a stub).
*   **`DistributedTaskInvocationStrategy`**: An `InvocationStrategy` that uses a task queue plugin to execute tools.

## Configuration

Enable and configure via `FeatureSettings`:

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        task_queue="celery", # or "rq", or "none"
        task_queue_celery_broker_url="redis://localhost:6379/1",
        task_queue_celery_backend_url="redis://localhost:6379/2",
    ),
    # distributed_task_queue_configurations={
    #     "celery_task_queue_v1": {
    #         "celery_app_name": "my_genie_worker_app",
    #         # "celery_include_task_paths": ["my_project.worker_tasks"]
    #     }
    # }
)
```

## Using `genie.task_queue` (Direct Submission - Advanced)

```python
task_id = await genie.task_queue.submit_task(
    task_name="my_project.tasks.long_computation", # Name of task registered with Celery/RQ
    args=(10, 20),
    kwargs={"operation": "multiply"}
)
if task_id:
    print(f"Task submitted: {task_id}")
    # Later...
    # status = await genie.task_queue.get_task_status(task_id)
    # result = await genie.task_queue.get_task_result(task_id, timeout_seconds=30)
```

## Using `DistributedTaskInvocationStrategy` for Tools

To automatically offload specific tool executions:
1.  Ensure a task queue is configured (e.g., `features.task_queue = "celery"`).
2.  Define a Celery/RQ task in your worker environment that can execute Genie tools. A generic task like `genie_tooling.worker_tasks.execute_genie_tool_task` might be provided by Genie or your application. This task would typically receive `tool_id`, `params`, and potentially serialized key information.
3.  When calling `genie.execute_tool`, specify the `DistributedTaskInvocationStrategy`:

```python
# Assuming 'distributed_task_invocation_strategy_v1' is registered
# and configured to use your chosen task queue plugin.
# The strategy itself needs to be configured with the task_queue_plugin_id.

# This requires more setup in MiddlewareConfig for the strategy itself,
# or the strategy needs to discover the default task queue manager from Genie.
# For now, this is a conceptual example.

# result_or_task_ref = await genie.execute_tool(
#     "my_long_running_tool",
#     param1="value",
#     strategy_id="distributed_task_invocation_strategy_v1"
# )
```
Detailed configuration of the `DistributedTaskInvocationStrategy` and worker-side task definitions are crucial and will be expanded here.

**Note:** Securely passing API keys and context to distributed workers requires careful consideration.
