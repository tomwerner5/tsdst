from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from timeit import default_timer as dt
from .utils import updateProgBar


def p_prog_simp(args, loop_args, function, n_jobs=2, verbose=True, use_threads=True):
    r"""Run keyword-argument tasks with an optional progress display.

    Parameters
    ----------
    args : dict
        Common keyword arguments supplied to every task.
    loop_args : list of dict
        Per-task arguments. A task's values override duplicate common keys.
    function : callable
        Function called with the merged keyword arguments.
    n_jobs : int, optional
        Number of workers. One runs serially. The default is 2.
    verbose : bool, optional
        Show progress when True. The default is True.
    use_threads : bool, optional
        Use threads when True, or processes when False. Process tasks require
        pickleable callables and arguments. The default is True.

    Returns
    -------
    results : list
        Outputs in input task order. Parallel exceptions occupy the corresponding
        output position as exception objects. Serial exceptions propagate.

    Notes
    -----
    The helper leaves the common and per-task dictionaries unchanged. It submits
    all parallel tasks before collecting results. For a bounded submission queue,
    use p_prog_simp_memorySafe. Process workers need an importable execution
    context, which can matter in interactive frontends.
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
    r"""Run process tasks in bounded batches to limit queued work.

    This variant was introduced to reduce memory used by submitting all tasks
    at once. It keeps at most n_jobs tasks submitted in each batch. A new batch
    is submitted after the preceding batch has been collected.

    Parameters
    ----------
    args : dict
        Common keyword arguments supplied to every task.
    loop_args : list of dict
        Per-task arguments, overriding duplicate common keys.
    function : callable
        Function called with each merged argument mapping.
    n_jobs : int, optional
        Number of process workers and maximum batch size. The default is 2.
        One runs serially through p_prog_simp.

    Returns
    -------
    results : list
        Outputs in input task order. With multiple workers, an exception object
        occupies the failed task's position. Serial exceptions propagate.

    Notes
    -----
    Bounding queued futures does not place a fixed limit on the memory retained
    inside each worker. Inputs are not mutated. Progress is displayed while
    results are collected. Process callables and arguments must be pickleable.
    """
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
