from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import Any

from skyagent.exceptions import SkyAgentDetrimentalError


def run_async_function_sync(func, **kwargs):
    return asyncio.run(func(**kwargs))


@dataclass
class FunctionCall:
    """Represents a function call with its arguments and execution preferences."""

    function: callable
    arguments: dict[str, Any]
    function_name: str
    call_id: str
    compute_heavy: bool = False


@dataclass
class FunctionResult:
    """Represents the result of a function execution."""

    function_name: str
    call_id: str
    arguments: dict[str, Any]
    result: Any
    error: Exception | None = None


class FunctionExecutor:
    """
    Executes a list of functions with different execution strategies based on their type.
    - Regular (non-async, non-compute-heavy) functions are executed inline.
    - Async functions are executed concurrently.
    - Compute-heavy functions are executed in a process pool (even if they are async).

    If any function fails, all pending tasks are cancelled and a SkyAgentDetrimentalError is raised
    with the failing function's name, call ID, and parameters.
    """

    def __init__(self, timeout: float = 10.0, num_processes: int = 4) -> None:
        self._timeout = timeout
        self._num_processes = num_processes

    async def execute_all(
        self, function_calls: list[FunctionCall]
    ) -> list[FunctionResult]:
        if not function_calls:
            return []

        # Categorize function calls.
        inline_calls: list[FunctionCall] = []
        async_calls: list[FunctionCall] = []
        compute_heavy_calls: list[FunctionCall] = []

        for call in function_calls:
            # If compute_heavy, always use the process pool even if the function is async.
            if call.compute_heavy:
                if inspect.iscoroutinefunction(call.function):
                    # Wrap the async function in a top-level callable.
                    call.function = functools.partial(
                        run_async_function_sync, call.function
                    )
                compute_heavy_calls.append(call)
            elif inspect.iscoroutinefunction(call.function):
                async_calls.append(call)
            else:
                inline_calls.append(call)

        results = []
        # Execute inline functions sequentially.
        try:
            inline_results = self._execute_inline_calls(inline_calls)
            results.extend(inline_results)
        except Exception as e:
            raise e

        async_results = []
        comp_results = []

        try:
            async with asyncio.timeout(self._timeout):
                async with asyncio.TaskGroup() as tg:
                    task_async = None
                    task_comp = None
                    if async_calls:
                        task_async = tg.create_task(
                            self._execute_async_calls(async_calls)
                        )
                    if compute_heavy_calls:
                        task_comp = tg.create_task(
                            asyncio.to_thread(
                                self._execute_compute_heavy_calls, compute_heavy_calls
                            )
                        )
            if task_async is not None:
                async_results = task_async.result()
            if task_comp is not None:
                comp_results = task_comp.result()
        except Exception as e:

            if hasattr(e, "exceptions"):
                sub_errors = [str(sub) for sub in e.exceptions]
                detailed_error = " | ".join(sub_errors)
                raise SkyAgentDetrimentalError(
                    f"Function executions failed: {detailed_error}"
                ) from e
            else:
                raise SkyAgentDetrimentalError(f"Function execution failed: {e}") from e

        results.extend(async_results)
        results.extend(comp_results)
        return results

    def _execute_inline_calls(
        self, function_calls: list[FunctionCall]
    ) -> list[FunctionResult]:
        results = []
        for call in function_calls:
            print(
                f"Starting inline function: {call.function_name} (ID: {call.call_id})"
            )
            try:
                result_value = call.function(**call.arguments)
                result = FunctionResult(
                    function_name=call.function_name,
                    call_id=call.call_id,
                    arguments=call.arguments,
                    result=result_value,
                )
                print(
                    f"Completed inline function: {call.function_name} (ID: {call.call_id})"
                )
                results.append(result)
            except Exception as e:
                print(
                    f"Error executing function {call.function_name} (ID: {call.call_id}): {e}"
                )
                raise SkyAgentDetrimentalError(
                    f"Function '{call.function_name}' (ID: {call.call_id}) execution failed. Parameters: {call.arguments}"
                ) from e
        return results

    async def _execute_async_calls(
        self, function_calls: list[FunctionCall]
    ) -> list[FunctionResult]:
        if not function_calls:
            return []
        tasks = []
        try:
            async with asyncio.TaskGroup() as tg:
                for call in function_calls:
                    task = tg.create_task(self._execute_single_async_call(call))
                    # Attach the call info to each task for later error reporting.
                    task.call_info = call
                    tasks.append(task)
            results = [t.result() for t in tasks]
        except* asyncio.TimeoutError as eg:
            failed_functions = [
                f"{t.call_info.function_name} (ID: {t.call_info.call_id}, Parameters: {t.call_info.arguments})"
                for t in tasks
                if t.done()
                and t.exception()
                and isinstance(t.exception(), asyncio.TimeoutError)
            ]
            raise SkyAgentDetrimentalError(
                f"Async functions timed out: {', '.join(failed_functions)}"
            ) from eg
        except* Exception as eg:
            failed_functions = [
                f"{t.call_info.function_name} (ID: {t.call_info.call_id}, Parameters: {t.call_info.arguments})"
                for t in tasks
                if t.done() and t.exception() is not None
            ]
            raise SkyAgentDetrimentalError(
                f"Async function executions failed: {', '.join(failed_functions)}"
            ) from eg
        return results

    async def _execute_single_async_call(self, call: FunctionCall) -> FunctionResult:
        print(f"Starting async function: {call.function_name} (ID: {call.call_id})")
        try:
            async with asyncio.timeout(self._timeout):
                result_value = await call.function(**call.arguments)
            print(
                f"Completed async function: {call.function_name} (ID: {call.call_id})"
            )
            return FunctionResult(
                function_name=call.function_name,
                call_id=call.call_id,
                arguments=call.arguments,
                result=result_value,
            )
        except asyncio.TimeoutError as e:
            print(f"Async function {call.function_name} (ID: {call.call_id}) timed out")
            raise SkyAgentDetrimentalError(
                f"Function '{call.function_name}' (ID: {call.call_id}) timed out after {self._timeout}s. Parameters: {call.arguments}"
            ) from e
        except Exception as e:
            print(
                f"Error executing async function {call.function_name} (ID: {call.call_id}): {e}"
            )
            raise SkyAgentDetrimentalError(
                f"Function '{call.function_name}' (ID: {call.call_id}) execution failed. Parameters: {call.arguments}"
            ) from e

    def _execute_compute_heavy_calls(
        self, function_calls: list[FunctionCall]
    ) -> list[FunctionResult]:
        if not function_calls:
            return []
        results = []
        with ProcessPoolExecutor(max_workers=self._num_processes) as executor:
            futures = {}
            try:
                for call in function_calls:
                    print(
                        f"Starting compute-heavy function: {call.function_name} (ID: {call.call_id})"
                    )
                    future = executor.submit(call.function, **call.arguments)
                    futures[future] = call

                for future in as_completed(futures):
                    call = futures[future]
                    try:
                        result_value = future.result(timeout=self._timeout)
                        result = FunctionResult(
                            function_name=call.function_name,
                            call_id=call.call_id,
                            arguments=call.arguments,
                            result=result_value,
                        )
                        print(
                            f"Completed compute-heavy function: {call.function_name} (ID: {call.call_id})"
                        )
                        results.append(result)
                    except concurrent.futures.TimeoutError as e:
                        for f in futures:
                            f.cancel()
                        print(
                            f"Compute-heavy function {call.function_name} (ID: {call.call_id}) timed out"
                        )
                        raise SkyAgentDetrimentalError(
                            f"Function '{call.function_name}' (ID: {call.call_id}) timed out after {self._timeout}s. Parameters: {call.arguments}"
                        ) from e
                    except Exception as e:
                        print(
                            f"Error executing compute-heavy function {call.function_name} (ID: {call.call_id}): {e}"
                        )
                        raise SkyAgentDetrimentalError(
                            f"Function '{call.function_name}' (ID: {call.call_id}) execution failed. Parameters: {call.arguments}"
                        ) from e
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        return results
