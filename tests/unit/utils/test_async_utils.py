### tests/unit/utils/test_async_utils.py
import asyncio
from typing import AsyncIterable, List, TypeVar

import pytest
from genie_tooling.utils.async_utils import abatch_iterable, acollect

T = TypeVar("T")


async def simple_async_generator(*items: T) -> AsyncIterable[T]:
    """Helper async generator for tests."""
    for item in items:
        await asyncio.sleep(0)  # Yield control to event loop
        yield item


async def failing_async_generator(num_good_items: int, error_message: str = "Test error") -> AsyncIterable[int]:
    """Helper async generator that fails after yielding some items."""
    for i in range(num_good_items):
        await asyncio.sleep(0)
        yield i
    raise RuntimeError(error_message)


@pytest.mark.asyncio
class TestACollect:
    async def test_acollect_empty_iterable(self):
        """Test acollect with an empty async iterable."""
        result = await acollect(simple_async_generator())
        assert result == []

    async def test_acollect_with_items(self):
        """Test acollect with a few items."""
        items = [1, "two", {"three": 3}]
        result = await acollect(simple_async_generator(*items))
        assert result == items

    async def test_acollect_with_various_types(self):
        """Test acollect with mixed data types."""
        items = [None, True, 3.14, (1, 2)]
        result = await acollect(simple_async_generator(*items))
        assert result == items

    async def test_acollect_propagates_exception(self):
        """Test that acollect propagates exceptions from the iterable."""
        with pytest.raises(RuntimeError, match="Test error during acollect"):
            await acollect(failing_async_generator(2, "Test error during acollect"))


@pytest.mark.asyncio
class TestABatchIterable:
    async def test_abatch_empty_iterable(self):
        """Test abatch_iterable with an empty async iterable."""
        batches = [batch async for batch in abatch_iterable(simple_async_generator(), 3)]
        assert batches == []

    async def test_abatch_exact_multiple(self):
        """Test abatch_iterable where item count is an exact multiple of batch_size."""
        items = [1, 2, 3, 4, 5, 6]
        batch_size = 3
        expected_batches = [[1, 2, 3], [4, 5, 6]]
        batches = [batch async for batch in abatch_iterable(simple_async_generator(*items), batch_size)]
        assert batches == expected_batches

    async def test_abatch_not_exact_multiple(self):
        """Test abatch_iterable where item count is not an exact multiple (last batch smaller)."""
        items = [1, 2, 3, 4, 5]
        batch_size = 3
        expected_batches = [[1, 2, 3], [4, 5]]
        batches = [batch async for batch in abatch_iterable(simple_async_generator(*items), batch_size)]
        assert batches == expected_batches

    async def test_abatch_batch_size_one(self):
        """Test abatch_iterable with batch_size of 1."""
        items = ["a", "b", "c"]
        batch_size = 1
        expected_batches = [["a"], ["b"], ["c"]]
        batches = [batch async for batch in abatch_iterable(simple_async_generator(*items), batch_size)]
        assert batches == expected_batches

    async def test_abatch_batch_size_larger_than_items(self):
        """Test abatch_iterable where batch_size is larger than the total number of items."""
        items = [10, 20]
        batch_size = 5
        expected_batches = [[10, 20]]
        batches = [batch async for batch in abatch_iterable(simple_async_generator(*items), batch_size)]
        assert batches == expected_batches

    async def test_abatch_invalid_batch_size_zero(self):
        """Test abatch_iterable with batch_size of 0, expecting ValueError."""
        with pytest.raises(ValueError, match="batch_size must be a positive integer."):
            async for _ in abatch_iterable(simple_async_generator(1, 2), 0):
                pass  # pragma: no cover

    async def test_abatch_invalid_batch_size_negative(self):
        """Test abatch_iterable with a negative batch_size, expecting ValueError."""
        with pytest.raises(ValueError, match="batch_size must be a positive integer."):
            async for _ in abatch_iterable(simple_async_generator(1, 2), -2):
                pass  # pragma: no cover

    async def test_abatch_propagates_exception_from_iterable(self):
        """Test that abatch_iterable propagates exceptions from the source iterable."""
        batch_size = 2
        collected_batches: List[List[int]] = []
        with pytest.raises(RuntimeError, match="Test error during abatch"):
            async for batch in abatch_iterable(failing_async_generator(3, "Test error during abatch"), batch_size):
                collected_batches.append(batch)
        # Check that items before the error were processed correctly
        assert collected_batches == [[0, 1]]

    async def test_abatch_with_single_item_and_batch_size_one(self):
        """Test abatch_iterable with a single item and batch_size of 1."""
        items = [42]
        batch_size = 1
        expected_batches = [[42]]
        batches = [batch async for batch in abatch_iterable(simple_async_generator(*items), batch_size)]
        assert batches == expected_batches

    async def test_abatch_with_single_item_and_large_batch_size(self):
        """Test abatch_iterable with a single item and batch_size larger than 1."""
        items = [99]
        batch_size = 10
        expected_batches = [[99]]
        batches = [batch async for batch in abatch_iterable(simple_async_generator(*items), batch_size)]
        assert batches == expected_batches
