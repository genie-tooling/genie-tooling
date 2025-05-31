### tests/unit/token_usage/impl/test_in_memory_recorder.py
import asyncio
import logging
import time

import pytest
from genie_tooling.token_usage.impl.in_memory_recorder import (
    InMemoryTokenUsageRecorderPlugin,
)
from genie_tooling.token_usage.types import TokenUsageRecord
from typing import List
RECORDER_LOGGER_NAME = "genie_tooling.token_usage.impl.in_memory_recorder"


@pytest.fixture
async def mem_recorder() -> InMemoryTokenUsageRecorderPlugin:
    recorder = InMemoryTokenUsageRecorderPlugin()
    await recorder.setup()
    return recorder


@pytest.mark.asyncio
async def test_setup_initializes_correctly(mem_recorder: InMemoryTokenUsageRecorderPlugin):
    recorder = await mem_recorder
    assert recorder._records == []
    assert isinstance(recorder._lock, asyncio.Lock)


@pytest.mark.asyncio
async def test_record_usage_adds_record(mem_recorder: InMemoryTokenUsageRecorderPlugin):
    recorder = await mem_recorder
    record1: TokenUsageRecord = {
        "provider_id": "test_provider",
        "model_name": "model_x",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "call_type": "chat",
    }
    await recorder.record_usage(record1)
    assert len(recorder._records) == 1
    assert recorder._records[0]["provider_id"] == "test_provider"
    assert "timestamp" in recorder._records[0]  # Timestamp should be added

    record2: TokenUsageRecord = {
        "provider_id": "test_provider",
        "model_name": "model_y",
        "prompt_tokens": 20,
        "total_tokens": 20,
        "timestamp": 12345.678,
    }
    await recorder.record_usage(record2)
    assert len(recorder._records) == 2
    assert recorder._records[1]["timestamp"] == 12345.678


@pytest.mark.asyncio
async def test_get_summary_no_records(mem_recorder: InMemoryTokenUsageRecorderPlugin):
    recorder = await mem_recorder
    summary = await recorder.get_summary()
    assert summary["total_records"] == 0
    assert summary["total_prompt_tokens"] == 0
    assert summary["total_completion_tokens"] == 0
    assert summary["total_tokens_overall"] == 0
    assert summary["by_model"] == {}


@pytest.mark.asyncio
async def test_get_summary_with_records(mem_recorder: InMemoryTokenUsageRecorderPlugin):
    recorder = await mem_recorder
    records: List[TokenUsageRecord] = [
        {"provider_id": "p1", "model_name": "m1", "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        {"provider_id": "p1", "model_name": "m2", "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        {"provider_id": "p2", "model_name": "m1", "prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
        {"provider_id": "p1", "model_name": "m1", "prompt_tokens": 5, "total_tokens": 5}, # Only prompt tokens
    ]
    for rec in records:
        await recorder.record_usage(rec)

    summary = await recorder.get_summary()
    assert summary["total_records"] == 4
    assert summary["total_prompt_tokens"] == 10 + 20 + 30 + 5
    assert summary["total_completion_tokens"] == 5 + 10 + 15 + 0
    assert summary["total_tokens_overall"] == 15 + 30 + 45 + 5

    assert summary["by_model"]["m1"]["prompt"] == 10 + 30 + 5
    assert summary["by_model"]["m1"]["completion"] == 5 + 15
    assert summary["by_model"]["m1"]["total"] == 15 + 45 + 5
    assert summary["by_model"]["m1"]["count"] == 3

    assert summary["by_model"]["m2"]["prompt"] == 20
    assert summary["by_model"]["m2"]["completion"] == 10
    assert summary["by_model"]["m2"]["total"] == 30
    assert summary["by_model"]["m2"]["count"] == 1


@pytest.mark.asyncio
async def test_get_summary_with_filter(mem_recorder: InMemoryTokenUsageRecorderPlugin):
    recorder = await mem_recorder
    records: List[TokenUsageRecord] = [
        {"provider_id": "p1", "model_name": "m1", "prompt_tokens": 10, "total_tokens": 10, "user_id": "userA"},
        {"provider_id": "p1", "model_name": "m2", "prompt_tokens": 20, "total_tokens": 20, "user_id": "userB"},
        {"provider_id": "p2", "model_name": "m1", "prompt_tokens": 30, "total_tokens": 30, "user_id": "userA"},
    ]
    for rec in records:
        await recorder.record_usage(rec)

    summary_user_a = await recorder.get_summary(filter_criteria={"user_id": "userA"})
    assert summary_user_a["total_records"] == 2
    assert summary_user_a["total_prompt_tokens"] == 10 + 30
    assert summary_user_a["by_model"]["m1"]["count"] == 2

    summary_model_m2 = await recorder.get_summary(filter_criteria={"model_name": "m2"})
    assert summary_model_m2["total_records"] == 1
    assert summary_model_m2["total_prompt_tokens"] == 20


@pytest.mark.asyncio
async def test_clear_records_no_filter(mem_recorder: InMemoryTokenUsageRecorderPlugin):
    recorder = await mem_recorder
    await recorder.record_usage({"provider_id": "p", "model_name": "m", "total_tokens": 1})
    assert len(recorder._records) == 1
    cleared = await recorder.clear_records()
    assert cleared is True
    assert len(recorder._records) == 0


@pytest.mark.asyncio
async def test_clear_records_with_filter(mem_recorder: InMemoryTokenUsageRecorderPlugin):
    recorder = await mem_recorder
    await recorder.record_usage({"provider_id": "p1", "model_name": "m1", "user_id": "uA"})
    await recorder.record_usage({"provider_id": "p1", "model_name": "m2", "user_id": "uB"})
    await recorder.record_usage({"provider_id": "p2", "model_name": "m1", "user_id": "uA"})

    assert len(recorder._records) == 3
    cleared = await recorder.clear_records(filter_criteria={"user_id": "uA"})
    assert cleared is True
    assert len(recorder._records) == 1
    assert recorder._records[0]["user_id"] == "uB"


@pytest.mark.asyncio
async def test_teardown_clears_records(mem_recorder: InMemoryTokenUsageRecorderPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=RECORDER_LOGGER_NAME)
    recorder = await mem_recorder
    await recorder.record_usage({"provider_id": "p", "model_name": "m", "total_tokens": 1})
    assert len(recorder._records) == 1
    await recorder.teardown()
    assert len(recorder._records) == 0
    assert f"{recorder.plugin_id}: Teardown complete, records cleared." in caplog.text


@pytest.mark.asyncio
async def test_concurrent_record_and_summary(mem_recorder: InMemoryTokenUsageRecorderPlugin):
    recorder = await mem_recorder
    num_records = 100

    async def record_concurrently():
        for i in range(num_records // 2):
            await recorder.record_usage({"provider_id": "p_conc", "model_name": f"m_conc_{i%2}", "total_tokens": i})
            if i % 10 == 0:
                await asyncio.sleep(0.001) # Yield control

    async def get_summary_concurrently():
        summaries = []
        for _ in range(num_records // 2):
            summaries.append(await recorder.get_summary())
            if _ % 10 == 0:
                await asyncio.sleep(0.001) # Yield control
        return summaries

    task1 = asyncio.create_task(record_concurrently())
    task2 = asyncio.create_task(get_summary_concurrently())

    await asyncio.gather(task1, task2)

    final_summary = await recorder.get_summary()
    assert final_summary["total_records"] == num_records // 2
    # Further assertions on summary content could be made if needed,
    # but the main point is to check for race conditions/deadlocks.
    # The exact sum of tokens might be tricky if reads happen mid-write without perfect atomicity
    # in the test logic itself, but the recorder's lock should protect its internal state.
    assert final_summary["by_model"]["m_conc_0"]["count"] > 0
    assert final_summary["by_model"]["m_conc_1"]["count"] > 0