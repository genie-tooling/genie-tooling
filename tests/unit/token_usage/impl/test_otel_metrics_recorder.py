### tests/unit/token_usage/impl/test_otel_metrics_recorder.py
import logging
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.token_usage.impl.otel_metrics_recorder import (
    OTEL_METRICS_AVAILABLE,
    OpenTelemetryMetricsTokenRecorderPlugin,
)
from genie_tooling.token_usage.types import TokenUsageRecord

RECORDER_LOGGER_NAME = "genie_tooling.token_usage.impl.otel_metrics_recorder"

# Mock OTel metrics types if not available
if not OTEL_METRICS_AVAILABLE:
    class MockOtelMetrics:
        def get_meter_provider(self):
            meter_provider_mock = MagicMock(name="MockMeterProviderInstance")
            meter_mock = MagicMock(name="MockMeterInstanceFromMockProvider")

            # This will be further refined in the fixture
            meter_mock.create_counter = MagicMock(name="MockCreateCounterMethod")
            meter_provider_mock.get_meter = MagicMock(return_value=meter_mock)
            return meter_provider_mock
    metrics_module_mock_for_no_otel = MockOtelMetrics()
else:
    from opentelemetry import metrics  # type: ignore
    metrics_module_mock_for_otel_available = metrics


@pytest.fixture()
def mock_otel_meter_components():
    # Create the specific counter mocks ONCE.
    # These are the instances we want the plugin to use and our tests to assert against.
    prompt_counter_mock = MagicMock(name="MockPromptCounterInstance")
    prompt_counter_mock.add = MagicMock(name="MockAddMethod_Prompt")

    completion_counter_mock = MagicMock(name="MockCompletionCounterInstance")
    completion_counter_mock.add = MagicMock(name="MockAddMethod_Completion")

    total_counter_mock = MagicMock(name="MockTotalCounterInstance")
    total_counter_mock.add = MagicMock(name="MockAddMethod_Total")

    # This is the single mock meter instance that will be used.
    the_one_true_mock_meter = MagicMock(name="SharedMockMeterInstanceForAllCases")

    # Configure its create_counter method's side_effect
    def create_counter_side_effect(name, unit, description):
        if name == "llm.request.tokens.prompt":
            return prompt_counter_mock
        if name == "llm.request.tokens.completion":
            return completion_counter_mock
        if name == "llm.request.tokens.total":
            return total_counter_mock
        # Fallback for any other unexpected counter creation
        fallback_counter = MagicMock(name=f"FallbackMockCounter_{name}")
        fallback_counter.add = MagicMock(name=f"FallbackMockAdd_{name}")
        return fallback_counter

    the_one_true_mock_meter.create_counter = MagicMock(side_effect=create_counter_side_effect)

    mock_get_meter_method = MagicMock(return_value=the_one_true_mock_meter)
    mock_meter_provider_instance = MagicMock(name="MockMeterProviderInstance")
    mock_meter_provider_instance.get_meter = mock_get_meter_method
    mock_get_meter_provider_func = MagicMock(return_value=mock_meter_provider_instance)

    metrics_to_patch = metrics_module_mock_for_otel_available if OTEL_METRICS_AVAILABLE else metrics_module_mock_for_no_otel

    with patch("genie_tooling.token_usage.impl.otel_metrics_recorder.metrics", metrics_to_patch), \
         patch("genie_tooling.token_usage.impl.otel_metrics_recorder.metrics.get_meter_provider", mock_get_meter_provider_func):
        yield {
            "meter": the_one_true_mock_meter,
            "prompt_counter": prompt_counter_mock,
            "completion_counter": completion_counter_mock,
            "total_counter": total_counter_mock,
        }

@pytest.fixture()
async def otel_metrics_recorder(mock_otel_meter_components) -> OpenTelemetryMetricsTokenRecorderPlugin:
    recorder = OpenTelemetryMetricsTokenRecorderPlugin()
    await recorder.setup()
    return recorder


@pytest.mark.asyncio()
async def test_setup_otel_not_available(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=RECORDER_LOGGER_NAME)
    with patch("genie_tooling.token_usage.impl.otel_metrics_recorder.OTEL_METRICS_AVAILABLE", False):
        recorder_no_otel = OpenTelemetryMetricsTokenRecorderPlugin()
        await recorder_no_otel.setup()
    assert recorder_no_otel._meter is None
    assert "OpenTelemetry metrics libraries not available. This recorder will be a no-op." in caplog.text

@pytest.mark.asyncio()
async def test_setup_initializes_counters(otel_metrics_recorder: OpenTelemetryMetricsTokenRecorderPlugin, mock_otel_meter_components):
    recorder = await otel_metrics_recorder
    assert recorder._meter is mock_otel_meter_components["meter"]
    # These assertions should now pass because create_counter in the plugin's setup
    # will return the exact same mock instances created in the fixture.
    assert recorder._prompt_tokens_counter is mock_otel_meter_components["prompt_counter"]
    assert recorder._completion_tokens_counter is mock_otel_meter_components["completion_counter"]
    assert recorder._total_tokens_counter is mock_otel_meter_components["total_counter"]

@pytest.mark.asyncio()
async def test_record_usage_adds_to_counters(otel_metrics_recorder: OpenTelemetryMetricsTokenRecorderPlugin, mock_otel_meter_components):
    recorder = await otel_metrics_recorder
    record: TokenUsageRecord = {
        "provider_id": "test_prov", "model_name": "test_model",
        "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
        "call_type": "chat", "user_id": "user1", "session_id": "sess1",
        "custom_tags": {"project": "alpha"}
    }
    expected_attrs = {
        "llm.provider.id": "test_prov", "llm.model.name": "test_model",
        "llm.call_type": "chat", "genie.client.user_id": "user1",
        "genie.client.session_id": "sess1", "genie.tag.project": "alpha"
    }
    await recorder.record_usage(record)
    # These assertions should now work because recorder._prompt_tokens_counter.add
    # is the same mock method as mock_otel_meter_components["prompt_counter"].add
    mock_otel_meter_components["prompt_counter"].add.assert_called_once_with(100, attributes=expected_attrs)
    mock_otel_meter_components["completion_counter"].add.assert_called_once_with(50, attributes=expected_attrs)
    mock_otel_meter_components["total_counter"].add.assert_called_once_with(150, attributes=expected_attrs)

@pytest.mark.asyncio()
async def test_record_usage_partial_tokens(otel_metrics_recorder: OpenTelemetryMetricsTokenRecorderPlugin, mock_otel_meter_components):
    recorder = await otel_metrics_recorder
    record: TokenUsageRecord = {"provider_id": "p", "model_name": "m", "prompt_tokens": 75}
    await recorder.record_usage(record)
    mock_otel_meter_components["prompt_counter"].add.assert_called_once_with(75, attributes={"llm.provider.id": "p", "llm.model.name": "m"})
    mock_otel_meter_components["completion_counter"].add.assert_not_called()
    mock_otel_meter_components["total_counter"].add.assert_not_called()

@pytest.mark.asyncio()
async def test_get_summary_and_clear_records_log_warning(otel_metrics_recorder: OpenTelemetryMetricsTokenRecorderPlugin, caplog: pytest.LogCaptureFixture):
    recorder = await otel_metrics_recorder
    caplog.set_level(logging.WARNING, logger=RECORDER_LOGGER_NAME)

    summary = await recorder.get_summary()
    assert "summary not available here" in summary.get("status", "")
    assert "get_summary is not applicable" in caplog.text
    caplog.clear()

    cleared = await recorder.clear_records()
    assert cleared is False
    assert "clear_records is not applicable" in caplog.text

@pytest.mark.asyncio()
async def test_teardown_nullifies_otel_objects(otel_metrics_recorder: OpenTelemetryMetricsTokenRecorderPlugin):
    recorder = await otel_metrics_recorder
    assert recorder._meter is not None
    await recorder.teardown()
    assert recorder._meter is None
    assert recorder._prompt_tokens_counter is None
