# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Utility helpers for skills and code execution.

These utilities provide common patterns for tool composition code,
inspired by the TypeScript implementation in agent-codemode-claude-poc.

Key utilities:
- wait_for: Wait for an async condition with polling
- retry: Retry an async function on failure
- run_with_timeout: Execute with a timeout
- parallel: Run multiple async functions in parallel
- RateLimiter: Simple rate limiter for tool calls

These are useful for skills that compose multiple tools.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional, TypeVar, Union

T = TypeVar("T")

# Type aliases for conditions and functions
AsyncCondition = Callable[[], Union[bool, "asyncio.Future[bool]"]]
AsyncFunction = Callable[[], "asyncio.Future[T]"]


async def wait_for(
    condition: AsyncCondition,
    interval_seconds: float = 1.0,
    timeout_seconds: Optional[float] = None,
    description: Optional[str] = None,
) -> None:
    """Wait for a condition to become true.
    
    This is the "More Powerful Control" pattern from Code Mode:
    instead of burning tokens with repeated LLM calls checking conditions,
    write code that polls in a loop.
    
    Example:
        # Wait for a file to appear
        await wait_for(
            lambda: Path("/tmp/results.json").exists(),
            interval_seconds=2,
            timeout_seconds=60,
        )
        
        # Wait for a Slack message (hypothetical)
        async def check_slack():
            messages = await slack.list_messages({"channel": "general"})
            return any(m["text"].startswith("READY") for m in messages)
        
        await wait_for(check_slack, interval_seconds=5, timeout_seconds=300)
    
    Args:
        condition: Callable (sync or async) that returns True when ready.
        interval_seconds: Time between checks (default: 1 second).
        timeout_seconds: Maximum time to wait (None = wait forever).
        description: Optional description for error messages.
    
    Raises:
        TimeoutError: If timeout is reached before condition is true.
    """
    start_time = asyncio.get_event_loop().time()
    
    while True:
        # Check condition (handle both sync and async)
        result = condition()
        if asyncio.iscoroutine(result):
            result = await result
        
        if result:
            return
        
        # Check timeout
        if timeout_seconds is not None:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout_seconds:
                desc = description or "condition"
                raise TimeoutError(
                    f"Timeout waiting for {desc} after {timeout_seconds}s"
                )
        
        # Wait before next check
        await asyncio.sleep(interval_seconds)


async def retry(
    fn: AsyncFunction[T],
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 1.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> T:
    """Retry an async function on failure.
    
    Useful for handling transient failures when calling external tools
    or APIs.
    
    Example:
        # Retry an API call with exponential backoff
        result = await retry(
            lambda: api.fetch_data({"id": "123"}),
            max_attempts=5,
            delay_seconds=1,
            backoff_factor=2,  # 1s, 2s, 4s, 8s, 16s
        )
        
        # Retry with custom exception handling
        async def fetch_with_logging():
            return await api.fetch_data({"id": "123"})
        
        result = await retry(
            fetch_with_logging,
            max_attempts=3,
            on_retry=lambda e, i: print(f"Attempt {i} failed: {e}")
        )
    
    Args:
        fn: Async function to retry.
        max_attempts: Maximum number of attempts (default: 3).
        delay_seconds: Initial delay between attempts (default: 1s).
        backoff_factor: Multiply delay by this after each attempt (default: 1).
        exceptions: Tuple of exception types to catch (default: Exception).
        on_retry: Optional callback called on each retry with (exception, attempt).
    
    Returns:
        Result of the function if it succeeds.
    
    Raises:
        The last exception if all attempts fail.
    """
    last_exception: Optional[Exception] = None
    current_delay = delay_seconds
    
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except exceptions as e:
            last_exception = e
            
            if attempt < max_attempts:
                if on_retry:
                    on_retry(e, attempt)
                await asyncio.sleep(current_delay)
                current_delay *= backoff_factor
    
    if last_exception:
        raise last_exception
    raise RuntimeError(f"Retry failed after {max_attempts} attempts")


async def run_with_timeout(
    fn: AsyncFunction[T],
    timeout_seconds: float,
    description: Optional[str] = None,
) -> T:
    """Run an async function with a timeout.
    
    Example:
        result = await run_with_timeout(
            lambda: fetch_large_file({"url": "..."}),
            timeout_seconds=30,
            description="fetching file",
        )
    
    Args:
        fn: Async function to run.
        timeout_seconds: Maximum execution time.
        description: Optional description for error message.
    
    Returns:
        Result of the function.
    
    Raises:
        asyncio.TimeoutError: If the function doesn't complete in time.
    """
    try:
        return await asyncio.wait_for(fn(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        desc = description or "operation"
        raise TimeoutError(f"Timeout: {desc} did not complete in {timeout_seconds}s")


async def parallel(*tasks: AsyncFunction[Any]) -> list[Any]:
    """Run multiple async functions in parallel.
    
    Convenience wrapper around asyncio.gather.
    
    Example:
        results = await parallel(
            lambda: fetch_file("a.txt"),
            lambda: fetch_file("b.txt"),
            lambda: fetch_file("c.txt"),
        )
    
    Args:
        *tasks: Async functions to run.
    
    Returns:
        List of results in the same order as tasks.
    """
    return await asyncio.gather(*(t() for t in tasks))


class RateLimiter:
    """Simple rate limiter for tool calls.
    
    Useful when calling external APIs with rate limits.
    
    Example:
        limiter = RateLimiter(calls_per_second=5)
        
        for url in urls:
            await limiter.acquire()
            result = await fetch_url({"url": url})
    """
    
    def __init__(self, calls_per_second: float):
        """Initialize the rate limiter.
        
        Args:
            calls_per_second: Maximum calls per second.
        """
        self._interval = 1.0 / calls_per_second
        self._last_call: float = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Wait until a call is allowed."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self._last_call
            
            if time_since_last < self._interval:
                await asyncio.sleep(self._interval - time_since_last)
            
            self._last_call = asyncio.get_event_loop().time()


__all__ = [
    "wait_for",
    "retry",
    "run_with_timeout",
    "parallel",
    "RateLimiter",
]
