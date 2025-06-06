# Installation

This guide explains how to install Genie Tooling and its dependencies.

## Prerequisites

*   Python 3.10 or higher.
*   [Poetry](https://python-poetry.org/docs/#installation) for dependency management and packaging.

## Installation Steps

1.  **Clone the Repository (Optional, if contributing or running examples locally):**
    If you want to work with the source code or run the examples directly from the repository:
    ```bash
    git clone https://github.com/genie-tooling/genie-tooling.git
    cd genie-tooling
    ```

2.  **Install using Poetry:**

    *   **To install the core library and all optional dependencies (recommended for full functionality and running all examples):**
        Navigate to the cloned repository directory (if applicable) or where your project that depends on Genie Tooling is located.
        ```bash
        poetry install --all-extras
        ```

    *   **To install only the core library dependencies:**
        ```bash
        poetry install
        ```
        This will install the essential components of Genie Tooling. Some plugins requiring external libraries (e.g., specific LLM providers, vector stores, or tools) will not be functional unless their respective optional dependencies are installed.

    *   **To install specific optional dependencies (extras):**
        You can install only the extras you need. Extras are defined in the `pyproject.toml` file.
        For example, to install support for Ollama, OpenAI, Qdrant, Celery, and the internal Llama.cpp provider:
        ```bash
        poetry install --extras "ollama openai qdrant celery llama_cpp_internal"
        ```
        Common extras include:
        *   `web_tools`: For tools that interact with web pages (e.g., `WebPageLoader`, `GoogleSearchTool`).
        *   `openai_services`: For the OpenAI LLM provider and embedding generator.
        *   `local_rag`: For local RAG components like Sentence Transformers and FAISS.
        *   `distributed_rag`: For distributed RAG components like ChromaDB and Qdrant clients.
        *   `ollama`: For the Ollama LLM provider.
        *   `gemini`: For the Google Gemini LLM provider.
        *   `secure_exec`: For the `SecureDockerExecutor`.
        *   `observability`: For OpenTelemetry tracing and metrics.
        *   `prompts`: For advanced prompt templating engines like Jinja2.
        *   `task_queues`: For distributed task queue support (Celery, RQ).
        *   `llama_cpp_server`: For the Llama.cpp server-based LLM provider.
        *   `llama_cpp_internal`: For the Llama.cpp internal (direct library use) LLM provider.

    *   **Adding Genie Tooling as a dependency to your existing project:**
        If you are integrating Genie Tooling into your own Poetry-managed project, you can add it:
        ```bash
        poetry add genie-tooling # For the core library
        
        # To add with specific extras:
        poetry add genie-tooling -E ollama -E local_rag 
        # or
        poetry add genie-tooling[ollama,local_rag]
        ```

## Verifying Installation

After installation, you can try running one of the examples, such as the simple Ollama chat example (ensure Ollama is running):

```bash
# From the root of the genie-tooling repository if cloned:
poetry run python examples/E02_ollama_chat_example.py 
```

If the example runs without import errors and interacts with Ollama successfully, your basic installation is working.
