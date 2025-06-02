# examples/E22_distributed_task_example.py
"""
Example: Distributed Task Offloading with Celery (Conceptual)
-------------------------------------------------------------
This example outlines how one might use the Distributed Task Queue feature
with Celery. It assumes:
1. Celery is installed (`poetry add celery redis`).
2. A Celery worker is running and configured to find tasks.
3. A task (e.g., `execute_genie_tool_task`) is defined for the worker
   that can instantiate a minimal Genie environment or directly execute a tool.

This example focuses on the Genie client-side configuration and submission.
The worker-side task implementation is beyond this basic example.

To Run (Conceptual - requires worker setup):
1. Start Redis: `docker run -d -p 6379:6379 redis`
2. Start a Celery worker (details depend on your task definitions).
3. Run this script: `poetry run python examples/E22_distributed_task_example.py`
"""
import asyncio
import logging

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie

# Name of the generic task defined in your Celery worker environment
# This task would be responsible for loading the tool and executing it.
# Example: my_project.worker_tasks.execute_tool_remotely
REMOTE_TOOL_EXEC_TASK_NAME = "genie_tooling.worker_tasks.execute_genie_tool_task" # Placeholder

async def run_distributed_task_demo():
    print("--- Distributed Task Offloading Demo (Celery - Conceptual) ---")
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("genie_tooling").setLevel(logging.DEBUG)


    app_config = MiddlewareConfig(
        features=FeatureSettings(
            # LLM and other features might be needed by the tool itself on the worker
            llm="none",
            command_processor="none",

            # Configure Celery as the task queue
            task_queue="celery",
            task_queue_celery_broker_url="redis://localhost:6379/1",
            task_queue_celery_backend_url="redis://localhost:6379/2",
        ),
        # Configuration for the CeleryTaskQueuePlugin itself
        distributed_task_queue_configurations={
            "celery_task_queue_v1": {
                "celery_app_name": "genie_example_tasks",
                # "celery_include_task_paths": ["path.to.your.worker.tasks"] # If needed
            }
        },
        # Configuration for the DistributedTaskInvocationStrategy
        # This strategy needs to know which task queue plugin to use.
        # This can be set here or it could default to the one configured in features.
        # invocation_strategy_configurations={
        #     "distributed_task_invocation_strategy_v1": {
        #         "task_queue_plugin_id": "celery_task_queue_v1" # or alias "celery"
        #     }
        # }
    )

    genie: Optional[Genie] = None
    try:
        genie = await Genie.create(config=app_config)
        print("Genie initialized with Celery task queue support.")

        # --- 1. Direct Task Submission (if genie.task_queue interface is used) ---
        print("\n--- Submitting a generic task directly (conceptual) ---")
        # This assumes a task named 'add' is defined in your Celery workers.
        # task_id_add = await genie.task_queue.submit_task(
        #     task_name="your_project.tasks.add", # Replace with your actual task name
        #     args=(5, 7)
        # )
        # if task_id_add:
        #     print(f"Addition task submitted with ID: {task_id_add}")
        #     # Poll for result (simplified)
        #     for _ in range(10): # Max 10 attempts
        #         status = await genie.task_queue.get_task_status(task_id_add)
        #         print(f"Task {task_id_add} status: {status}")
        #         if status == "success":
        #             result = await genie.task_queue.get_task_result(task_id_add)
        #             print(f"Task {task_id_add} result: {result}")
        #             break
        #         elif status in ["failure", "revoked"]:
        #             print(f"Task {task_id_add} ended with status: {status}")
        #             break
        #         await asyncio.sleep(1)
        # else:
        #     print("Failed to submit addition task.")

        # --- 2. Tool Execution via DistributedTaskInvocationStrategy ---
        print("\n--- Executing a tool via DistributedTaskInvocationStrategy (conceptual) ---")
        # This requires:
        #   a) The 'distributed_task_invocation_strategy_v1' to be registered.
        #   b) A generic worker task (e.g., REMOTE_TOOL_EXEC_TASK_NAME) that can
        #      receive tool_id, params, and execute the tool.
        #   c) The strategy needs to be configured to use the 'celery_task_queue_v1'.

        # For this example, we'll simulate the direct submission part that the strategy would do.
        # A real scenario would involve:
        # await genie.execute_tool(
        #     "calculator_tool",
        #     num1=100, num2=5, operation="divide",
        #     strategy_id="distributed_task_invocation_strategy_v1"
        # )

        tool_exec_params = {
            "tool_id": "calculator_tool",
            "tool_params": {"num1": 200, "num2": 25, "operation": "multiply"},
            # "key_provider_info": {...} # Securely handle keys
            "context_info": {"user_id": "demo_user"}
        }

        task_id_tool = await genie.task_queue.submit_task(
            task_name=REMOTE_TOOL_EXEC_TASK_NAME, # This task needs to exist on your worker
            kwargs=tool_exec_params
        )

        if task_id_tool:
            print(f"Tool execution task for 'calculator_tool' submitted with ID: {task_id_tool}")
            # Simplified polling for demonstration
            result_output = "Polling for result..."
            for i in range(15): # Poll for up to 15 seconds
                status = await genie.task_queue.get_task_status(task_id_tool)
                print(f"  (Poll {i+1}) Task {task_id_tool} status: {status}")
                if status == "success":
                    tool_result = await genie.task_queue.get_task_result(task_id_tool)
                    result_output = f"Tool task '{task_id_tool}' successful. Result: {tool_result}"
                    break
                elif status in ["failure", "revoked"]:
                    result_output = f"Tool task '{task_id_tool}' failed or revoked. Status: {status}"
                    try:
                        error_details = await genie.task_queue.get_task_result(task_id_tool)
                        result_output += f" Details: {error_details}"
                    except Exception:
                        pass # Details might not be available
                    break
                await asyncio.sleep(1)
            print(result_output)
        else:
            print("Failed to submit tool execution task.")


    except Exception as e:
        print(f"An error occurred: {e}")
        logging.exception("Distributed task demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    # Note: This example is conceptual and requires a running Celery worker
    # configured with the specified tasks to execute successfully.
    print("This example is conceptual and requires a configured Celery (or other queue) worker.")
    print("It demonstrates the client-side API usage.")
    asyncio.run(run_distributed_task_demo())
