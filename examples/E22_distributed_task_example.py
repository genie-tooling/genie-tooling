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
from typing import Optional  # Added Optional

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
            llm="none",
            command_processor="none",
            task_queue="celery",
            task_queue_celery_broker_url="redis://localhost:6379/1",
            task_queue_celery_backend_url="redis://localhost:6379/2",
        ),
        distributed_task_queue_configurations={
            "celery_task_queue_v1": {
                "celery_app_name": "genie_example_tasks",
            }
        },
        tool_configurations={
            # If the worker task needs to execute a tool, ensure it's enabled
            # in the worker's Genie config (not necessarily here on client-side)
            # For this conceptual client, we don't need to enable tools here.
        }
    )

    genie: Optional[Genie] = None
    try:
        genie = await Genie.create(config=app_config)
        print("Genie initialized with Celery task queue support.")

        tool_exec_params = {
            "tool_id": "calculator_tool", # Tool to be executed by the worker
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
            for i in range(15):
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
                        pass
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
    print("This example is conceptual and requires a configured Celery (or other queue) worker.")
    print("It demonstrates the client-side API usage.")
    asyncio.run(run_distributed_task_demo())
