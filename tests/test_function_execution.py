from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from skyagent.exceptions import SkyAgentDetrimentalError
from skyagent.function_executor import FunctionCall
from skyagent.function_executor import FunctionExecutor


# =======================
# Dummy Functions for Tests
# =======================


# Two inline (synchronous) functions
def inline_func1():
    time.sleep(0.5)
    return "inline1"


def inline_func2():
    time.sleep(0.5)
    return "inline2"


# Two async functions


async def async_func1():
    await asyncio.sleep(0.5)
    return "async1"


async def async_func2():
    await asyncio.sleep(0.5)
    return "async2"


# Two compute-heavy functions (synchronous)


def compute_heavy_func1():
    time.sleep(0.5)
    return "compute1"


def compute_heavy_func2():
    time.sleep(0.5)
    return "compute2"


# Two compute-heavy async functions


async def async_compute_heavy_func1():
    await asyncio.sleep(0.5)
    return "async_compute1"


async def async_compute_heavy_func2():
    await asyncio.sleep(0.5)
    return "async_compute2"


# Failing functions for each category:


def inline_fail():
    time.sleep(0.5)
    raise Exception("inline failure")


async def async_fail():
    await asyncio.sleep(0.5)
    raise Exception("async failure")


def compute_heavy_fail():
    time.sleep(0.5)
    raise Exception("compute heavy failure")


async def async_compute_heavy_fail():
    await asyncio.sleep(0.5)
    raise Exception("async compute heavy failure")


# =======================
# Pytest Test Functions
# =======================


@pytest.mark.asyncio
async def test_results_mapping():
    """
    Test that all functions (inline, async, compute-heavy, compute-heavy async)
    return results with the correct function name and call_id.
    """
    executor = FunctionExecutor(timeout=5.0, num_processes=4)
    calls = [
        FunctionCall(
            function=inline_func1,
            arguments={},
            function_name="inline_func1",
            call_id=str(uuid.uuid4()),
        ),
        FunctionCall(
            function=inline_func2,
            arguments={},
            function_name="inline_func2",
            call_id=str(uuid.uuid4()),
        ),
        FunctionCall(
            function=async_func1,
            arguments={},
            function_name="async_func1",
            call_id=str(uuid.uuid4()),
        ),
        FunctionCall(
            function=async_func2,
            arguments={},
            function_name="async_func2",
            call_id=str(uuid.uuid4()),
        ),
        FunctionCall(
            function=compute_heavy_func1,
            arguments={},
            function_name="compute_heavy_func1",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
        FunctionCall(
            function=compute_heavy_func2,
            arguments={},
            function_name="compute_heavy_func2",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
        FunctionCall(
            function=async_compute_heavy_func1,
            arguments={},
            function_name="async_compute_heavy_func1",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
        FunctionCall(
            function=async_compute_heavy_func2,
            arguments={},
            function_name="async_compute_heavy_func2",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
    ]
    results = await executor.execute_all(calls)

    mapping = {res.function_name: res.result for res in results}
    assert mapping["inline_func1"] == "inline1"
    assert mapping["inline_func2"] == "inline2"
    assert mapping["async_func1"] == "async1"
    assert mapping["async_func2"] == "async2"
    assert mapping["compute_heavy_func1"] == "compute1"
    assert mapping["compute_heavy_func2"] == "compute2"
    assert mapping["async_compute_heavy_func1"] == "async_compute1"
    assert mapping["async_compute_heavy_func2"] == "async_compute2"

    call_ids = {call.call_id for call in calls}
    result_ids = {res.call_id for res in results}
    assert call_ids == result_ids


@pytest.mark.asyncio
async def test_concurrency():
    """
    Test that functions (both compute heavy and async) run concurrently.
    Each function sleeps 0.5 seconds; running four concurrently should take near 0.5s.
    """
    executor = FunctionExecutor(timeout=5.0, num_processes=4)
    calls = [
        FunctionCall(
            function=async_func1,
            arguments={},
            function_name="async_func1",
            call_id=str(uuid.uuid4()),
            compute_heavy=False,
        ),
        FunctionCall(
            function=async_func2,
            arguments={},
            function_name="async_func2",
            call_id=str(uuid.uuid4()),
            compute_heavy=False,
        ),
        FunctionCall(
            function=compute_heavy_func1,
            arguments={},
            function_name="compute_heavy_func1",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
        FunctionCall(
            function=compute_heavy_func2,
            arguments={},
            function_name="compute_heavy_func2",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
        FunctionCall(
            function=async_compute_heavy_func1,
            arguments={},
            function_name="async_compute_heavy_func1",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
        FunctionCall(
            function=async_compute_heavy_func2,
            arguments={},
            function_name="async_compute_heavy_func2",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
    ]

    start = time.monotonic()
    results = await executor.execute_all(calls)
    elapsed = time.monotonic() - start

    assert elapsed < 0.6, f"Elapsed time was {elapsed:.2f}s, expected near 0.5s"

    mapping = {res.function_name: res.result for res in results}
    assert mapping["async_func1"] == "async1"
    assert mapping["async_func2"] == "async2"
    assert mapping["compute_heavy_func1"] == "compute1"
    assert mapping["compute_heavy_func2"] == "compute2"
    assert mapping["async_compute_heavy_func1"] == "async_compute1"
    assert mapping["async_compute_heavy_func2"] == "async_compute2"


# Additional tests can target individual categories if desired.


@pytest.mark.asyncio
async def test_inline_failure_handling():
    """
    Test that a failure in an inline function is handled correctly.
    """
    executor = FunctionExecutor(timeout=5.0, num_processes=2)
    calls = [
        FunctionCall(
            function=inline_func1,
            arguments={},
            function_name="inline_func1",
            call_id=str(uuid.uuid4()),
        ),
        FunctionCall(
            function=inline_fail,
            arguments={},
            function_name="inline_fail",
            call_id=str(uuid.uuid4()),
        ),
    ]
    with pytest.raises(SkyAgentDetrimentalError) as exc_info:
        await executor.execute_all(calls)
    error_msg = str(exc_info.value)
    assert "inline_fail" in error_msg, "Error message should mention inline_fail"


@pytest.mark.asyncio
async def test_async_failure_handling():
    """
    Test that a failure in an async function is handled correctly.
    """
    executor = FunctionExecutor(timeout=5.0, num_processes=2)
    calls = [
        FunctionCall(
            function=async_func1,
            arguments={},
            function_name="async_func1",
            call_id=str(uuid.uuid4()),
        ),
        FunctionCall(
            function=async_fail,
            arguments={},
            function_name="async_fail",
            call_id=str(uuid.uuid4()),
        ),
    ]
    with pytest.raises(SkyAgentDetrimentalError) as exc_info:
        await executor.execute_all(calls)
    error_msg = str(exc_info.value)
    assert "async_fail" in error_msg, "Error message should mention async_fail"


@pytest.mark.asyncio
async def test_compute_heavy_failure_handling():
    """
    Test that a failure in a compute-heavy function is handled correctly.
    """
    executor = FunctionExecutor(timeout=5.0, num_processes=2)
    calls = [
        FunctionCall(
            function=compute_heavy_func1,
            arguments={},
            function_name="compute_heavy_func1",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
        FunctionCall(
            function=compute_heavy_fail,
            arguments={},
            function_name="compute_heavy_fail",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
    ]
    with pytest.raises(SkyAgentDetrimentalError) as exc_info:
        await executor.execute_all(calls)
    error_msg = str(exc_info.value)
    assert (
        "compute_heavy_fail" in error_msg
    ), "Error message should mention compute_heavy_fail"


@pytest.mark.asyncio
async def test_async_compute_heavy_failure_handling():
    """
    Test that a failure in an async compute-heavy function is handled correctly.
    """
    executor = FunctionExecutor(timeout=5.0, num_processes=2)
    calls = [
        FunctionCall(
            function=async_compute_heavy_func1,
            arguments={},
            function_name="async_compute_heavy_func1",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
        FunctionCall(
            function=async_compute_heavy_fail,
            arguments={},
            function_name="async_compute_heavy_fail",
            call_id=str(uuid.uuid4()),
            compute_heavy=True,
        ),
    ]
    with pytest.raises(SkyAgentDetrimentalError) as exc_info:
        await executor.execute_all(calls)
    error_msg = str(exc_info.value)
    assert (
        "async_compute_heavy_fail" in error_msg
    ), "Error message should mention async_compute_heavy_fail"


# Additional tests can target individual categories if desired.


@pytest.mark.asyncio
async def test_inline_functions_mapping():
    """
    Test that the inline functions return the expected fixed values.
    """
    executor = FunctionExecutor(timeout=5.0, num_processes=2)
    calls = [
        FunctionCall(
            function=inline_func1,
            arguments={},
            function_name="inline_func1",
            call_id=str(uuid.uuid4()),
        ),
        FunctionCall(
            function=inline_func2,
            arguments={},
            function_name="inline_func2",
            call_id=str(uuid.uuid4()),
        ),
    ]
    results = await executor.execute_all(calls)
    mapping = {r.function_name: r.result for r in results}
    assert mapping["inline_func1"] == "inline1"
    assert mapping["inline_func2"] == "inline2"


@pytest.mark.asyncio
async def test_async_functions_mapping():
    """
    Test that the async functions return the expected fixed values.
    """
    executor = FunctionExecutor(timeout=5.0, num_processes=2)
    calls = [
        FunctionCall(
            function=async_func1,
            arguments={},
            function_name="async_func1",
            call_id=str(uuid.uuid4()),
        ),
        FunctionCall(
            function=async_func2,
            arguments={},
            function_name="async_func2",
            call_id=str(uuid.uuid4()),
        ),
    ]
    results = await executor.execute_all(calls)
    mapping = {r.function_name: r.result for r in results}
    assert mapping["async_func1"] == "async1"
    assert mapping["async_func2"] == "async2"
