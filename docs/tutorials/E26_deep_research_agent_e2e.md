# Tutorial: Deep Research Agent E2E (E26)

This tutorial corresponds to the example file `examples/E26_deep_research_agent_e2e.py`.

It demonstrates how to configure and run the `DeepResearchAgent`, a more advanced agent that uses a plan-and-execute strategy. It shows how to:
- Configure the agent to use specific tools for web search (DuckDuckGo) and academic paper search (ArXiv).
- Configure "deep retrieval" tools (`WebPageScraperTool`, `ArxivPDFTextExtractorTool`) that the agent can choose to use for in-depth analysis of promising search results.
- Observe the agent's planning, execution, and final report generation process.

## Example Code

--8<-- "examples/E26_deep_research_agent_e2e.py"
