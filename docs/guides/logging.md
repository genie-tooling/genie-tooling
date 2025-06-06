# Logging

Genie Tooling uses the standard Python `logging` module for its internal logging. This allows application developers to integrate Genie's logs into their existing logging infrastructure seamlessly.

## Library Logger Name

The root logger name for all messages originating from the Genie Tooling library is:

```
genie_tooling
```

Submodules within the library will use child loggers of this root logger (e.g., `genie_tooling.tools.manager`, `genie_tooling.llm_providers.impl.ollama_provider`).

## Configuring Logging in Your Application

By default, Genie Tooling adds a `logging.NullHandler()` to its root logger (`genie_tooling`). This prevents "No handlers could be found for logger 'genie_tooling'" warnings if your application doesn't explicitly configure logging for the library.

To see logs from Genie Tooling, you need to configure a handler and set a log level for the `"genie_tooling"` logger (or one of its parent loggers, like the root logger) in your application code.

**Example: Basic Console Logging**

Here's how you can enable basic console logging for Genie Tooling messages:

```python
import logging

# Get the Genie Tooling library logger
library_logger = logging.getLogger("genie_tooling")

# Set the desired log level (e.g., DEBUG for verbose output, INFO for general, WARNING for issues)
library_logger.setLevel(logging.DEBUG) 

# Create a handler (e.g., StreamHandler to output to console)
console_handler = logging.StreamHandler()

# Optional: Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(module)s:%(lineno)d)')
console_handler.setFormatter(formatter)

# Add the handler to the library logger
# Check if a similar handler already exists to avoid duplicates if this code runs multiple times
if not any(isinstance(h, logging.StreamHandler) for h in library_logger.handlers):
    library_logger.addHandler(console_handler)

# Optional: Prevent Genie logs from propagating to the root logger if it also has handlers
# library_logger.propagate = False 

# Now, when you use Genie, its logs will appear on the console.
# from genie_tooling.genie import Genie
# ... your Genie setup and usage ...
```

**Common Log Levels:**

*   `logging.DEBUG`: Detailed information, typically of interest only when diagnosing problems.
*   `logging.INFO`: Confirmation that things are working as expected, high-level operational messages.
*   `logging.WARNING`: An indication that something unexpected happened, or indicative of some problem in the near future (e.g., ‘disk space low’). The software is still working as expected.
*   `logging.ERROR`: Due to a more serious problem, the software has not been able to perform some function.
*   `logging.CRITICAL`: A serious error, indicating that the program itself may be unable to continue running.

You can integrate Genie's logging with more advanced logging setups, such as logging to files, using structured logging (e.g., JSON format), or sending logs to external monitoring services, just as you would for any other Python library. The `DefaultLogAdapter` plugin, if configured, can also influence logging behavior, including redaction.
