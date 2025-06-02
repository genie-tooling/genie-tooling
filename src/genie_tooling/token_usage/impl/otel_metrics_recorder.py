"""OpenTelemetryMetricsTokenRecorderPlugin: Emits token usage as OTel metrics."""
import logging
from typing import Any, Dict, Optional

from genie_tooling import __version__
from genie_tooling.token_usage.abc import TokenUsageRecorderPlugin
from genie_tooling.token_usage.types import TokenUsageRecord

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics
    OTEL_METRICS_AVAILABLE = True
except ImportError:
    OTEL_METRICS_AVAILABLE = False
    metrics = None # type: ignore
    logger.warning(
        "OpenTelemetryMetricsTokenRecorderPlugin: 'opentelemetry-sdk' or 'opentelemetry-api' "
        "not installed (or metrics components missing). This recorder will be a no-op."
    )

class OpenTelemetryMetricsTokenRecorderPlugin(TokenUsageRecorderPlugin):
    plugin_id: str = "otel_metrics_token_recorder_v1"
    description: str = "Records LLM token usage as OpenTelemetry metrics."

    _meter: Optional["metrics.Meter"] = None
    _prompt_tokens_counter: Optional["metrics.Counter"] = None
    _completion_tokens_counter: Optional["metrics.Counter"] = None
    _total_tokens_counter: Optional["metrics.Counter"] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not OTEL_METRICS_AVAILABLE or not metrics:
            logger.warning(f"{self.plugin_id}: OpenTelemetry metrics libraries not available. This recorder will be a no-op.")
            return

        # Assumes a global MeterProvider is already configured (e.g., by OpenTelemetryTracerPlugin or application)
        # If not, this plugin might need to initialize a basic one.
        try:
            meter_provider = metrics.get_meter_provider()
            self._meter = meter_provider.get_meter(self.plugin_id, __version__)
        except Exception as e:
            logger.error(f"{self.plugin_id}: Could not get OTel Meter. Ensure MeterProvider is configured. Error: {e}", exc_info=True)
            self._meter = None
            return

        self._prompt_tokens_counter = self._meter.create_counter(
            name="llm.request.tokens.prompt",
            unit="1",
            description="Number of prompt tokens sent to the LLM."
        )
        self._completion_tokens_counter = self._meter.create_counter(
            name="llm.request.tokens.completion",
            unit="1",
            description="Number of completion tokens received from the LLM."
        )
        self._total_tokens_counter = self._meter.create_counter(
            name="llm.request.tokens.total",
            unit="1",
            description="Total number of tokens for an LLM request."
        )
        logger.info(f"{self.plugin_id}: OpenTelemetry Metrics Recorder initialized.")

    async def record_usage(self, record: TokenUsageRecord) -> None:
        if not self._meter or not self._prompt_tokens_counter or \
           not self._completion_tokens_counter or not self._total_tokens_counter:
            logger.debug(f"{self.plugin_id}: OTel Meter or counters not initialized. Skipping metric recording.")
            return

        attributes = {
            "llm.provider.id": record.get("provider_id", "unknown"),
            "llm.model.name": record.get("model_name", "unknown"),
        }
        if record.get("call_type"):
            attributes["llm.call_type"] = record["call_type"]
        if record.get("user_id"):
            attributes["genie.client.user_id"] = record["user_id"]
        if record.get("session_id"):
            attributes["genie.client.session_id"] = record["session_id"]
        if record.get("custom_tags"):
            for tag_key, tag_val in record["custom_tags"].items():
                attributes[f"genie.tag.{tag_key}"] = str(tag_val) # Ensure string value

        if record.get("prompt_tokens") is not None:
            self._prompt_tokens_counter.add(record["prompt_tokens"], attributes=attributes)
        if record.get("completion_tokens") is not None:
            self._completion_tokens_counter.add(record["completion_tokens"], attributes=attributes)
        if record.get("total_tokens") is not None:
            self._total_tokens_counter.add(record["total_tokens"], attributes=attributes)

    async def get_summary(self, filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.warning(f"{self.plugin_id}: get_summary is not applicable for an OTel metrics exporter. "
                       "Summaries should be viewed in your OTel metrics backend (e.g., Prometheus, Grafana).")
        return {"status": "Metrics are exported via OTel; summary not available here."}

    async def clear_records(self, filter_criteria: Optional[Dict[str, Any]] = None) -> bool:
        logger.warning(f"{self.plugin_id}: clear_records is not applicable for an OTel metrics exporter.")
        return False

    async def teardown(self) -> None:
        # OTel Meter and Counter lifecycle is typically managed by the MeterProvider.
        # No specific shutdown needed for individual instruments here.
        self._meter = None
        self._prompt_tokens_counter = None
        self._completion_tokens_counter = None
        self._total_tokens_counter = None
        logger.debug(f"{self.plugin_id}: Teardown complete.")
