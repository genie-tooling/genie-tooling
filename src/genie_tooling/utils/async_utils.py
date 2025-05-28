"""Utility functions for asynchronous operations."""
from typing import AsyncIterable, AsyncIterator, List, TypeVar

T = TypeVar("T")

async def acollect(async_iterable: AsyncIterable[T]) -> List[T]:
    """Collects all items from an async iterable into a list."""
    return [item async for item in async_iterable]

async def abatch_iterable(async_iterable: AsyncIterable[T], batch_size: int) -> AsyncIterator[List[T]]:
    """
    Batches items from an async iterable.

    Args:
        async_iterable: The async iterable to batch.
        batch_size: The desired size of each batch.

    Yields:
        Lists of items, each of `batch_size` (except possibly the last one).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    current_batch: List[T] = []
    async for item in async_iterable:
        current_batch.append(item)
        if len(current_batch) >= batch_size:
            yield current_batch
            current_batch = []
    if current_batch: # Yield any remaining items in the last batch
        yield current_batch

# Example of a retry decorator (can be more sophisticated)
# import functools
# import logging
# logger = logging.getLogger(__name__)

# def async_retry(max_attempts: int = 3, delay_seconds: float = 1.0, backoff_factor: float = 2.0,
#                 retry_on_exceptions: tuple = (Exception,)):
#     def decorator(async_func):
#         @functools.wraps(async_func)
#         async def wrapper(*args, **kwargs):
#             attempts = 0
#             current_delay = delay_seconds
#             while attempts < max_attempts:
#                 try:
#                     return await async_func(*args, **kwargs)
#                 except retry_on_exceptions as e:
#                     attempts += 1
#                     if attempts >= max_attempts:
#                         logger.error(f"Async function {async_func.__name__} failed after {max_attempts} attempts. Last error: {e}", exc_info=True)
#                         raise
#                     logger.warning(f"Async function {async_func.__name__} attempt {attempts} failed: {e}. Retrying in {current_delay:.2f}s...")
#                     await asyncio.sleep(current_delay)
#                     current_delay *= backoff_factor
#         return wrapper
#     return decorator
