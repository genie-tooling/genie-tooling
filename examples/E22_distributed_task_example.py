# examples/E22_distributed_task_example.py
"""
Example: Distributed Task Offloading (Conceptual)
-------------------------------------------------------------
This example outlines how one might use the Distributed Task Queue feature
with Celery or RQ. It assumes:
1. Celery/RQ and a broker (e.g., Redis) are installed.
2. A worker is running and configured to find tasks.
3. A task (e.g., `execute_genie_tool_task`) is defined for the worker
   that can instantiate a minimal Genie environment or directly execute a tool.

This example focuses on the Genie client-side configuration and submission.
The worker-side task implementation is beyond this basic example.

To Run (Conceptual - requires worker setup):
1. Start Redis: `docker run -d -p 6379:6379 redis`
2. Start a Celery or RQ worker (details depend on your task definitions).
3. Run this script: `poetry run python examples/E22_distributed_task_example.py`
"""
import asyncio
import logging
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie

REMOTE_TOOL_EXEC_TASK_NAME = "genie_tooling.worker_tasks.execute_genie_tool_task" # Placeholder

async def run_distributed_task_demo():
    print("--- Distributed Task Offloading Demo (Conceptual) ---")
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)

    # --- Configuration for Celery ---
    app_config_celery = MiddlewareConfig(
        features=FeatureSettings(
            llm="none",
            command_processor="none",
            task_queue="celery",
            task_queue_celery_broker_url="redis://localhost:6379/1",
            task_queue_celery_backend_url="redis://localhost:6379/2",
        ),
        distributed_task_queue_configurations={
            "celery_task_queue_v1": {
                "celery_app_name": "genie_example_tasks_celery",
            }
        }
    )

    # --- Configuration for RQ (Redis Queue) ---
    app_config_rq = MiddlewareConfig(
        features=FeatureSettings(
            llm="none",
            command_processor="none",
            task_queue="rq",
        ),
        distributed_task_queue_configurations={
            "redis_queue_task_plugin_v1": { # Canonical ID for RQ plugin
                "redis_url": "redis://localhost:6379/3", # Separate Redis DB for RQ
                "default_queue_name": "genie-rq-jobs"
            }
        }
    )
    
    # Select the configuration to use
    app_config = app_config_celery
    # app_config = app_config_rq # Uncomment to test with RQ

    print(f"Using task queue: {app_config.features.task_queue}")

    genie: Optional[Genie] = None
    try:
        genie = await Genie.create(config=app_config)
        print(f"Genie initialized with {app_config.features.task_queue} task queue support.")

        tool_exec_params = {
            "tool_id": "calculator_tool", 
            "tool_params": {"num1": 200, "num2": 25, "operation": "multiply"},
            "context_info": {"user_id": "demo_user"}
        }

        task_id_tool = await genie.task_queue.submit_task(
            task_name=REMOTE_TOOL_EXEC_TASK_NAME,
            kwargs=tool_exec_params
        )

        if task_id_tool:
            print(f"Tool execution task for 'calculator_tool' submitted with ID: {task_id_tool}")
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
                        pass # Error details might not be available or might raise
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
    print("This example is conceptual and requires a configured Celery/RQ worker.")
    print("It demonstrates the client-side API usage for both.")
    asyncio.run(run_distributed_task_demo())
