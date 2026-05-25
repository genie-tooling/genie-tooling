# Tutorial: Math Proof Assistant Agent

This tutorial corresponds to the example file `examples/run_math_bot.py`.

It demonstrates how to run the `MathProofAssistantAgent`, a stateful, interactive agent designed to help users explore complex mathematical concepts and proofs. It combines several core `genie-tooling` capabilities:

-   **Deep Research**: It uses the `DeepResearchAgent` as a sub-process to find and summarize external knowledge.
-   **Symbolic Math**: It leverages the `SymbolicMathTool` (powered by SymPy) to test hypotheses and simplify expressions.
-   **RAG Memory**: It saves all findings—research summaries, user insights, hypothesis test results—into a persistent RAG collection, building a project-specific knowledge base over time.
-   **Intent Classification**: It uses an LLM to understand the user's intent in a conversational setting, deciding whether to research, test a hypothesis, or continue with a plan.

## Running the Example

**Prerequisites**:
1.  Both `genie-tooling` and `genie-tooling-agents` installed.
2.  A local GGUF model file downloaded.
3.  Set the `LLAMA_CPP_INTERNAL_MODEL_PATH` environment variable or use the `--model-path` command-line argument.

```bash
# From the root of the genie-tooling-agents repository
poetry run python examples/run_math_bot.py --model-path /path/to/your/model.gguf
```

The script will launch an interactive terminal UI where you can guide the agent's research and proof-building process.

