# Tutorial: Distributed Tasks (E22)

This tutorial corresponds to the example file `examples/E22_distributed_task_example.py`.

It demonstrates how to offload long-running tasks to a distributed queue like Celery or RQ. It shows how to:
- Configure a `DistributedTaskQueuePlugin` for Celery or RQ.
- Use `genie.task_queue.submit_task()` to send a task to a worker.
- Use `genie.task_queue.get_task_status()` and `genie.task_queue.get_task_result()` to monitor and retrieve results.

**Note**: This example focuses on the client-side API. A running Celery/RQ worker environment is required to execute the tasks.

## Example Code

--8<-- "examples/E22_distributed_task_example.py"
