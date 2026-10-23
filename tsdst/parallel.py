from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from timeit import default_timer as dt
from .utils import updateProgBar


def p_prog_simp(args, loop_args, function, n_jobs=2, verbose=True, use_threads=True):
    """Run keyword tasks and return results in input order.

    Parallel task failures remain exception objects in the result list.
    The function keeps the input argument dictionaries unchanged.
    """
    t0 = dt()
    tasks = [{**args, **task} for task in loop_args]
    if n_jobs == 1:
        results = []
        for i, task in enumerate(tasks):
            results.append(function(**task))
            if verbose:
                updateProgBar(i + 1, len(tasks), t0)
        return results
    executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with executor(max_workers=n_jobs) as pool:
        futures = [pool.submit(function, **task) for task in tasks]
        if verbose:
            for i, future in enumerate(as_completed(futures)):
                updateProgBar(i + 1, len(tasks), t0)
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                results.append(error)
        return results


def p_prog_simp_memorySafe(args, loop_args, function, n_jobs=2):
    """Run process tasks with at most n_jobs submitted tasks at a time."""
    if n_jobs == 1:
        return p_prog_simp(args, loop_args, function, n_jobs=1)
    results = []
    t0 = dt()
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for start in range(0, len(loop_args), n_jobs):
            futures = [pool.submit(function, **{**args, **task})
                       for task in loop_args[start:start + n_jobs]]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append(error)
                updateProgBar(len(results), len(loop_args), t0)
    return results
