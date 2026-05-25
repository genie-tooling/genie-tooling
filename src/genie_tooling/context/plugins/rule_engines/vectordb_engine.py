"""VectorDBRuleEnginePlugin: semantic-search rule engine for fuzzy matching.

Two distinct operating modes, controlled by the ``use_llm_descriptions``
config flag:

* **Deterministic mode (default, audit-safe)** — `use_llm_descriptions=False`.
  Indexes each rule using its static ``description`` field (or a structured
  fallback derived from the rule's predicate/conditions/actions if no
  description is provided). Loading a given rule set produces the same
  embedding inputs every time, so rule selection is reproducible across
  process restarts and across LLM model upgrades. This is what corporate
  audit / governance teams need.

* **LLM-enriched mode** — `use_llm_descriptions=True`. Uses an LLM call
  at load-time to generate a richer semantic description of each rule
  for embedding. Produces better fuzzy-match accuracy at the cost of
  non-determinism: the same rule indexed under two different LLM model
  versions will produce different embeddings and therefore different
  match results. **Not suitable for audit-bound deployments.** Setup
  emits a loud warning when this mode is selected so operators can
  notice if it's accidentally on in a production-tagged environment.

Configuration::

    rules_path: str (optional, default = bundled rules dir)
    collection_name: str (default "cqs_rules_semantic_index")
    use_llm_descriptions: bool (default False — audit-safe)
"""
from __future__ import annotations

import asyncio
import importlib.resources
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import yaml

from genie_tooling.context.protocols import RuleEnginePlugin
from genie_tooling.context.types import RuleObject

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


def _default_rules_path() -> Path:
    return Path(str(importlib.resources.files("genie_tooling.context") / "rules"))


def _deterministic_rule_text(rule: RuleObject) -> str:
    """Build a stable, deterministic embedding input from a rule's static
    fields. Same rule → same text → same embedding → same retrieval
    behavior, audit-safe across process restarts and model versions.

    Prefers the author-supplied ``description`` field. When absent, falls
    back to a structured derivation from the predicate + conditions +
    actions so a rule with no description is still meaningfully searchable.
    """
    description = (rule.get("description") or "").strip()
    if description:
        return description

    parts: List[str] = []
    rule_id = rule.get("rule_id", "")
    predicate = rule.get("predicate", "")
    parts.append(f"Rule {rule_id} matches predicate '{predicate}'.")
    conditions = rule.get("conditions") or []
    if conditions:
        formatted = [
            f"{cond[0]} {cond[1]} {cond[2]}"
            for cond in conditions
            if isinstance(cond, (list, tuple)) and len(cond) >= 3
        ]
        if formatted:
            parts.append("Conditions: " + "; ".join(formatted) + ".")
    # Surface the rule's intent via its actions in a stable order.
    actions = rule.get("actions") or []
    action_summaries: List[str] = []
    for action in actions:
        if isinstance(action, (list, tuple)) and len(action) == 4:
            target, op, key, value = action
            action_summaries.append(f"{target}.{key} = {value}")
    if action_summaries:
        parts.append("Effects: " + "; ".join(action_summaries) + ".")
    return " ".join(parts)


class VectorDBRuleEnginePlugin(RuleEnginePlugin):
    """A semantic rule engine that uses a vector database to find the most relevant rule.

    See module docstring for the deterministic-vs-LLM-enriched mode
    trade-off. Defaults to deterministic mode for audit safety.
    """

    plugin_id: str = "vectordb_rule_engine_v1"
    description: str = (
        "Finds relevant context rules using semantic vector search. "
        "Deterministic by default (audit-safe); opt into LLM-enriched "
        "descriptions via use_llm_descriptions=True at cost of "
        "non-determinism across LLM model versions."
    )

    _rules_path: Path
    _rules_in_memory: Dict[str, RuleObject]
    _collection_name: str
    _use_llm_descriptions: bool = False

    async def setup(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        path_override = cfg.get("rules_path")
        self._rules_path = Path(path_override) if path_override else _default_rules_path()
        self._rules_in_memory = {}
        self._collection_name = cfg.get(
            "collection_name", "cqs_rules_semantic_index"
        )
        self._use_llm_descriptions = bool(cfg.get("use_llm_descriptions", False))

        if self._use_llm_descriptions:
            logger.warning(
                f"[{self.plugin_id}] use_llm_descriptions=True: rule index "
                "will be built from LLM-generated text. This is NOT "
                "deterministic across LLM model versions — re-indexing under "
                "a different model will produce different match behavior. "
                "Do NOT use in audit-bound deployments; prefer the default "
                "deterministic mode or the FileSystemRuleEnginePlugin."
            )
        logger.info(
            f"[{self.plugin_id}] Initialized. Rules path: '{self._rules_path}'. "
            f"Collection: '{self._collection_name}'. "
            f"Mode: {'LLM-enriched' if self._use_llm_descriptions else 'deterministic'}."
        )

    async def _rule_to_text(self, rule: RuleObject, genie: "Genie") -> str:
        """
        Generates the text that gets embedded for semantic search.

        Deterministic mode: derives the text from the rule's static fields.
        LLM-enriched mode: asks the model to summarize the rule.
        """
        if not self._use_llm_descriptions:
            return _deterministic_rule_text(rule)

        rule_json = json.dumps(rule, indent=2)
        prompt = f"""
        Based on the following rule object, generate a concise, natural language paragraph
        that describes the situation this rule is designed for and what it does.
        This description will be used for semantic search.

        Rule Object:
        ```json
        {rule_json}
        ```

        Example Description:
        "This rule is for when a user asks a factual 'what is' question. It prioritizes finding a direct answer using a knowledge lookup engine and presenting it concisely, like an encyclopedia entry."
        """
        try:
            response = await genie.llm.generate(
                prompt, temperature=0.2, max_tokens=150
            )
            description = response.get("text", rule.get("description", ""))
            return description.strip()
        except Exception as e:
            logger.error(
                f"[{self.plugin_id}] LLM call to describe rule '{rule.get('rule_id')}' failed: {e}. "
                "Falling back to deterministic description."
            )
            return _deterministic_rule_text(rule)

    @staticmethod
    def _read_rule_file_sync(file_path: Path) -> Any:
        """Synchronous helper; called via asyncio.to_thread from load_rules."""
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    async def load_rules(self, genie: "Genie") -> bool:
        """
        Loads rules from the filesystem and indexes them into a vector store
        using the provided Genie instance.
        """
        is_dir = await asyncio.to_thread(self._rules_path.is_dir)
        if not is_dir:
            logger.error(
                f"[{self.plugin_id}] Rules path '{self._rules_path}' is not a valid directory."
            )
            return False

        logger.info(
            f"[{self.plugin_id}] Loading rules from '{self._rules_path}' and indexing for semantic search..."
        )
        rule_paths = await asyncio.to_thread(lambda: list(self._rules_path.glob("*.yml")))
        for file_path in rule_paths:
            try:
                rule_data = await asyncio.to_thread(self._read_rule_file_sync, file_path)
                if isinstance(rule_data, dict):
                    rule_id = rule_data.get("rule_id")
                    if not rule_id:
                        logger.warning(
                            f"Skipping rule from {file_path} due to missing 'rule_id'."
                        )
                        continue
                    self._rules_in_memory[rule_id] = rule_data

                    rule_text = await self._rule_to_text(rule_data, genie)

                    # Use genie.rag.index_text which is a convenience method for single text indexing
                    await genie.rag.index_text(
                        text=rule_text,
                        collection_name=self._collection_name,
                        metadata={"rule_id": rule_id, "source_file": str(file_path)},
                    )
            except Exception as e:
                logger.error(
                    f"[{self.plugin_id}] Failed to load, parse, or index rule file {file_path}: {e}"
                )

        logger.info(
            f"[{self.plugin_id}] Finished loading and indexing {len(self._rules_in_memory)} rules."
        )
        return True

    async def evaluate(
        self, inferred_context: Dict[str, Any], query_predicate: str, genie: "Genie"
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Finds relevant rules by performing a semantic search against the
        indexed rule descriptions.
        """
        context_query_text = (
            f"The user's action is about '{query_predicate}'. "
            f"The inferred context is: {json.dumps(inferred_context, indent=2)}."
        )
        logger.debug(f"[{self.plugin_id}] Semantic search query: {context_query_text}")

        try:
            search_results = await genie.rag.search(
                query=context_query_text,
                collection_name=self._collection_name,
                top_k=5,
            )
        except Exception as e:
            logger.error(
                f"[{self.plugin_id}] RAG search for rules failed: {e}", exc_info=True
            )
            return []

        ranked_rules = []
        for chunk in search_results:
            rule_id = chunk.metadata.get("rule_id")
            if rule_id and rule_id in self._rules_in_memory:
                rule_object = self._rules_in_memory[rule_id]
                score = chunk.score
                ranked_rules.append((rule_object, score))
            else:
                logger.warning(
                    f"[{self.plugin_id}] RAG returned a rule with ID '{rule_id}' that is not in memory."
                )

        if ranked_rules:
            logger.info(
                f"Found {len(ranked_rules)} semantically relevant rules. Top hit: '{ranked_rules[0][0].get('rule_id')}' with score {ranked_rules[0][1]:.4f}."
            )

        return ranked_rules
