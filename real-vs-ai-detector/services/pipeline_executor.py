"""
services/pipeline_executor.py — Parallel pipeline execution engine.

Runs independent analysis tasks concurrently using ThreadPoolExecutor.
Implements conditional execution to skip unused analyses.
Caches intermediate results for reuse.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logger import setup_logger

logger = setup_logger("services.pipeline_executor")

# Shared thread pool (reused across requests)
_executor = ThreadPoolExecutor(max_workers=4)


def execute_parallel(*tasks) -> dict:
    """
    Execute multiple tasks in parallel and collect results.

    Args:
        *tasks: Each task is a tuple of (name, callable, args_tuple).
                Example: ("fft", extract_fft_features, (image_path,))

    Returns:
        Dict mapping task_name -> result (or error dict).
    """
    start = time.time()
    results = {}
    futures = {}

    for task in tasks:
        name, func, args = task
        future = _executor.submit(func, *args)
        futures[future] = name

    for future in as_completed(futures):
        name = futures[future]
        try:
            results[name] = future.result(timeout=60)
        except Exception as e:
            logger.error(f"Pipeline task '{name}' failed: {e}")
            results[name] = {"error": str(e)}

    elapsed = round((time.time() - start) * 1000, 1)
    logger.info(f"Parallel pipeline: {len(tasks)} tasks in {elapsed}ms")

    return results


def time_execution(func, *args, **kwargs):
    """
    Execute a function and return (result, elapsed_ms).

    Args:
        func:     Function to call.
        *args:    Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tuple of (result, elapsed_ms).
    """
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = round((time.time() - start) * 1000, 1)
    return result, elapsed
