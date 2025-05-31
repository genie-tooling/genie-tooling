# Genie Facade and Interfaces

## Genie Facade

::: genie_tooling.genie.Genie
    handler: python
    options:
      show_root_heading: true

## Core Interfaces

These interfaces are accessed via attributes on the `Genie` facade instance (e.g., `genie.llm`, `genie.rag`).

::: genie_tooling.interfaces.LLMInterface
    handler: python
    options:
      show_root_heading: true
      
::: genie_tooling.interfaces.RAGInterface
    handler: python
    options:
      show_root_heading: true

::: genie_tooling.interfaces.ObservabilityInterface
    handler: python
    options:
      show_root_heading: true

::: genie_tooling.interfaces.HITLInterface
    handler: python
    options:
      show_root_heading: true

::: genie_tooling.interfaces.UsageTrackingInterface
    handler: python
    options:
      show_root_heading: true

::: genie_tooling.interfaces.PromptInterface
    handler: python
    options:
      show_root_heading: true

::: genie_tooling.interfaces.ConversationInterface
    handler: python
    options:
      show_root_heading: true
