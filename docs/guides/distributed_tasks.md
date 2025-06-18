# Using Distributed Task Queues (`genie.task_queue`)

Genie Tooling supports offloading tasks, such as long-running tool executions, to distributed task queues like Celery or RQ. This allows your main agent loop to remain responsive.

## Core Concepts

*   **`TaskQueueInterface` (`genie.task_queue`)**: Facade for submitting and managing distributed tasks.
*   **`DistributedTaskQueuePlugin`**: Plugin for a specific task queue system.
    *   Built-in: `CeleryTaskQueuePlugin` (alias: `celery`), `RedisQueueTaskPlugin` (alias: `rq`).

## Configuration

Enable and configure your chosen task queue system via `FeatureSettings`.

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

# Example for Celery
app_config_celery = MiddlewareConfig(
    features=FeatureSettings(
        task_queue="celery",
        task_queue_celery_broker_url="redis://localhost:6379/1", # Your Celery broker
        task_queue_celery_backend_url="redis://localhost:6379/2", # Your Celery result backend
    )
)

# Example for RQ (Redis Queue)
app_config_rq = MiddlewareConfig(
    features=FeatureSettings(
        task_queue="rq"
    ),
    # RQ plugin specific config
    distributed_task_queue_configurations={
        "redis_queue_task_plugin_v1": { # Canonical ID
            "redis_url": "redis://localhost:6379/3", # Redis for RQ
            "default_queue_name": "genie-rq-default"
        }
    }
)
```
Ensure your chosen task queue broker (e.g., Redis, RabbitMQ) is running and accessible.

## Using `genie.task_queue`

This interface allows you to directly submit tasks to the configured queue. You must have tasks defined and registered with your Celery/RQ worker environment.

```python
# Assuming 'genie' is initialized with a task queue configured
# and you have a worker task defined as 'my_project.tasks.long_computation'

task_id = await genie.task_queue.submit_task(
    task_name="my_project.tasks.long_computation",
    args=(10, 20),
    kwargs={"operation": "multiply"},
    task_options={"countdown": 5} # Example: Celery-specific option
)

if task_id:
    print(f"Task submitted with ID: {task_id}")

    # Poll for status and result (example polling loop)
    for _ in range(30):
        status = await genie.task_queue.get_task_status(task_id)
        print(f"Task {task_id} status: {status}")
        if status == "success":
            result = await genie.task_queue.get_task_result(task_id)
            print(f"Task result: {result}")
            break
        elif status in ["failure", "revoked"]:
            print(f"Task failed or was revoked.")
            break
        await asyncio.sleep(1)
```

**Note on Worker Implementation:** A common pattern is to create a generic worker task that can initialize a minimal Genie environment to execute tools. This keeps your web-facing application lightweight while offloading heavy work.

**Note on Security and Context:** Securely passing API keys and sensitive context to distributed workers requires careful design. Options include workers fetching credentials themselves from a secure vault or using short-lived, task-specific credentials. The direct `genie.task_queue.submit_task` method gives you full control over what gets sent to the worker.
