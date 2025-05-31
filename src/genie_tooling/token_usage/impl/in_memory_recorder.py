"""InMemoryTokenUsageRecorderPlugin: Stores token usage in memory."""
import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from genie_tooling.token_usage.abc import TokenUsageRecorderPlugin
from genie_tooling.token_usage.types import TokenUsageRecord

logger = logging.getLogger(__name__)

class InMemoryTokenUsageRecorderPlugin(TokenUsageRecorderPlugin):
    plugin_id: str = "in_memory_token_usage_recorder_v1"
    description: str = "Stores LLM token usage records in an in-memory list."

    _records: List[TokenUsageRecord]
    _lock: asyncio.Lock

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._records = []
        self._lock = asyncio.Lock()
        logger.info(f"{self.plugin_id}: Initialized. Records will be stored in memory.")

    async def record_usage(self, record: TokenUsageRecord) -> None:
        async with self._lock:
            # Ensure timestamp if not provided
            if "timestamp" not in record:
                record["timestamp"] = time.time()
            self._records.append(record)
        # logger.debug(f"{self.plugin_id}: Recorded usage for model {record.get('model_name')}")

    async def get_summary(self, filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # This is a basic summary. More complex filtering/aggregation could be added.
        async with self._lock:
            records_to_summarize = self._records
            if filter_criteria: # Basic filtering example
                # This is a naive filter; real applications might need more robust filtering
                def matches(rec):
                    return all(rec.get(k) == v for k, v in filter_criteria.items())
                records_to_summarize = [r for r in self._records if matches(r)]

            summary: Dict[str, Any] = {
                "total_records": len(records_to_summarize),
                "total_prompt_tokens": sum(r.get("prompt_tokens", 0) or 0 for r in records_to_summarize),
                "total_completion_tokens": sum(r.get("completion_tokens", 0) or 0 for r in records_to_summarize),
                "total_tokens_overall": sum(r.get("total_tokens", 0) or 0 for r in records_to_summarize),
                "by_model": defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0, "count": 0})
            }
            for r in records_to_summarize:
                model = r.get("model_name", "unknown_model")
                summary["by_model"][model]["prompt"] += r.get("prompt_tokens", 0) or 0
                summary["by_model"][model]["completion"] += r.get("completion_tokens", 0) or 0
                summary["by_model"][model]["total"] += r.get("total_tokens", 0) or 0
                summary["by_model"][model]["count"] += 1
        return summary

    async def clear_records(self, filter_criteria: Optional[Dict[str, Any]] = None) -> bool:
        async with self._lock:
            if filter_criteria:
                # Naive filter for clearing, similar to summary
                def matches(rec):
                    return all(rec.get(k) == v for k, v in filter_criteria.items())
                self._records = [r for r in self._records if not matches(r)]
            else:
                self._records.clear()
        logger.info(f"{self.plugin_id}: Records cleared (Criteria: {filter_criteria is not None}).")
        return True

    async def teardown(self) -> None:
        await self.clear_records()
        logger.debug(f"{self.plugin_id}: Teardown complete, records cleared.")
