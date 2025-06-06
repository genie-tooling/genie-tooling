# Using Distributed Task Queues

Genie Tooling supports offloading tasks, such as long-running tool executions, to distributed task queues like Celery or RQ. This allows your main agent loop to remain responsive.

## Core Concepts

*   **`TaskQueueInterface` (`genie.task_queue`)**: Facade for submitting and managing distributed tasks.
*   **`DistributedTaskQueuePlugin`**: Plugin for a specific task queue system.
    *   Built-in: `CeleryTaskQueuePlugin` (alias: `celery_task_queue`), `RedisQueueTaskPlugin` (alias: `rq_task_queue`).
*   **`DistributedTaskInvocationStrategy`**: An `InvocationStrategy` that uses a task queue plugin to execute tools. (Note: This strategy is currently more conceptual and may require further application-side setup for robust use.)

## Configuration

Enable and configure your chosen task queue system via `FeatureSettings`:

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

# Example for Celery
app_config_celery = MiddlewareConfig(
    features=FeatureSettings(
        task_queue="celery", 
        task_queue_celery_broker_url="redis://localhost:6379/1", # Your Celery broker
        task_queue_celery_backend_url="redis://localhost:6379/2", # Your Celery result backend
    ),
    # Optional: Further Celery-specific configurations
    distributed_task_queue_configurations={
        "celery_task_queue_v1": { # Canonical ID
            "celery_app_name": "my_genie_worker_app",
            # "celery_include_task_paths": ["my_project.worker_tasks"] # Paths for Celery to find tasks
        }
    }
)

# Example for RQ (Redis Queue)
app_config_rq = MiddlewareConfig(
    features=FeatureSettings(
        task_queue="rq" 
        # RQ typically uses a single Redis connection, configured in its plugin settings
    ),
    distributed_task_queue_configurations={
        "redis_queue_task_plugin_v1": { # Canonical ID
            "redis_url": "redis://localhost:6379/3", # Redis for RQ
            "default_queue_name": "genie-rq-default"
        }
    }
)
```
Ensure your chosen task queue broker (e.g., Redis, RabbitMQ) is running and accessible.

## Using `genie.task_queue` (Direct Task Submission)

This interface allows you to directly submit tasks to the configured queue. You need to have tasks defined and registered with your Celery/RQ worker environment.

```python
# Assuming 'genie' is initialized with a task queue configured (e.g., Celery)
# and you have a Celery task defined as 'my_project.tasks.long_computation'

task_id = await genie.task_queue.submit_task(
    task_name="my_project.tasks.long_computation", 
    args=(10, 20),
    kwargs={"operation": "multiply"},
    task_options={"countdown": 5} # Example: Celery-specific option
)

if task_id:
    print(f"Task submitted with ID: {task_id}")
    
    # Poll for status and result (example polling loop)
    for _ in range(30): # Try for 30 seconds
        status = await genie.task_queue.get_task_status(task_id)
        print(f"Task {task_id} status: {status}")
        if status == "success":
            result = await genie.task_queue.get_task_result(task_id)
            print(f"Task result: {result}")
            break
        elif status in ["failure", "revoked"]:
            print(f"Task failed or was revoked.")
            # Optionally try to get error details if it failed
            # result = await genie.task_queue.get_task_result(task_id) 
            break
        await asyncio.sleep(1)
else:
    print("Failed to submit task.")
```

## Using `DistributedTaskInvocationStrategy` for Tools (Conceptual)

The `DistributedTaskInvocationStrategy` aims to automatically offload tool executions to a task queue. This is a more advanced setup.

1.  **Configure Task Queue**: Ensure a task queue (e.g., Celery) is configured in `FeatureSettings`.
2.  **Worker Task**: Define a generic task in your Celery/RQ worker environment that can execute Genie tools. This task would typically:
    *   Receive `tool_id`, `tool_params`, and potentially serialized key information or context.
    *   Instantiate a minimal Genie environment (or have one pre-configured).
    *   Execute the specified tool with the given parameters.
    *   Return the tool's result.
    *   Example task name: `genie_tooling.worker_tasks.execute_genie_tool_task` (this is a placeholder; you'd implement this).
3.  **Configure the Strategy**: In `MiddlewareConfig`, you would configure the `DistributedTaskInvocationStrategy` to use your chosen task queue plugin and specify the name of your generic worker task.
    ```python
    # In MiddlewareConfig:
    # invocation_strategy_configurations={
    #     "distributed_task_invocation_strategy_v1": {
    #         "task_queue_plugin_id": "celery_task_queue_v1", # Or "redis_queue_task_plugin_v1"
    #         "worker_tool_execution_task_name": "my_project.worker_tasks.execute_genie_tool_task"
    #     }
    # }
    ```
4.  **Execute Tool with Strategy**:
    ```python
    # result_or_task_ref = await genie.execute_tool(
    #     "my_long_running_tool",
    #     param1="value",
    #     strategy_id="distributed_task_invocation_strategy_v1" 
    #     # The strategy might return a task ID or future immediately,
    #     # or poll internally and return the final result.
    # )
    ```

**Note on Security and Context:** Securely passing API keys and sensitive context to distributed workers requires careful design. Options include:
*   Workers fetching credentials themselves from a secure vault.
*   Using short-lived, task-specific credentials.
*   Carefully serializing only necessary and safe context information.

The direct `genie.task_queue.submit_task` method gives you more control over what gets sent to the worker.
