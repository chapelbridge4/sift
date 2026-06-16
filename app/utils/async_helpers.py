"""
Async utilities for batch processing and concurrent operations.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any, Callable, Iterable, List, TypeVar

from loguru import logger

T = TypeVar('T')
R = TypeVar('R')


class AsyncBatchProcessor:
    """Handles batch processing of items with async/thread pool support."""

    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers

    async def process_in_batches(
        self,
        items: List[T],
        process_func: Callable[[T], R],
        use_threads: bool = True
    ) -> List[R]:
        """
        Process items in batches using thread/process pool.

        Args:
            items: List of items to process
            process_func: Function to apply to each item
            use_threads: Use ThreadPoolExecutor if True, else ProcessPoolExecutor

        Returns:
            List of processed results
        """
        if not items:
            return []

        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size

        logger.info(f"Processing {len(items)} items in {total_batches} batches")

        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.debug(f"Processing batch {batch_num}/{total_batches}")

            if use_threads:
                batch_results = await self._process_batch_threaded(batch, process_func)
            else:
                batch_results = await self._process_batch_async(batch, process_func)

            results.extend(batch_results)

        return results

    async def _process_batch_threaded(
        self,
        batch: List[T],
        process_func: Callable[[T], R]
    ) -> List[R]:
        """Process batch using ThreadPoolExecutor."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                loop.run_in_executor(executor, process_func, item)
                for item in batch
            ]
            return await asyncio.gather(*futures)

    async def _process_batch_async(
        self,
        batch: List[T],
        process_func: Callable[[T], Any]
    ) -> List[R]:
        """Process batch using async gather."""
        tasks = [process_func(item) for item in batch]
        return await asyncio.gather(*tasks)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_calls: int, time_window: float):
        """
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            now = time.time()
            # Remove old calls outside time window
            self.calls = [call for call in self.calls if now - call < self.time_window]

            if len(self.calls) >= self.max_calls:
                # Calculate sleep time
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    self.calls = []

            self.calls.append(time.time())


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {current_delay}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries} retry attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper
    return decorator


async def gather_with_concurrency(n: int, *tasks):
    """
    Gather tasks with limited concurrency.

    Args:
        n: Maximum number of concurrent tasks
        *tasks: Async tasks to run

    Returns:
        List of results from all tasks
    """
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


def chunks(iterable: Iterable[T], size: int) -> Iterable[List[T]]:
    """
    Split an iterable into chunks of specified size.

    Args:
        iterable: Input iterable
        size: Chunk size

    Yields:
        Chunks of the specified size
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


class ProgressTracker:
    """Track progress of async operations."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0

        logger.info(
            f"{self.description}: {self.current}/{self.total} "
            f"({percentage:.1f}%) - {rate:.1f} items/sec"
        )

    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.current >= self.total
