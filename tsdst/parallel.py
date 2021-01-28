from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from timeit import default_timer as dt
from .utils import updateProgBar


def p_prog_simp(args, loop_args, function, n_jobs=2, verbose=True,
                use_threads=True):
    # TODO: provide a fix that will work more consistently
    '''
    Parallel Progress bar.

    Note: it is a known bug for this function to fail while using sypder.
    I have not been able to discover why, or provide a fix. If you run a script
    with this function from the console (or possibly jupyter), it should run
    just fine. 
    
    It is also known that using threads may be buggy for this function. It
    was intended to be used only on processes, but I included the thread option
    as a supplement to experiment with later. It is not guarenteed to work.
    
    Parameters
    ----------
    args : dict
        Dictionary of stationary arguments (i.e. the arguments that are
        constant throughout the loop).
    loop_args : list
        A list of dictionaries that contain the arguments that may change at 
        any point during the loop (this is the iterator, and any additional
        arguments that may have different values at different stages of the
        loop).
    function : function
        The function being being processed in the loop (args and loop args will
        be passed to this function).
    n_jobs : int, optional
        The number of parallel processes to run. The default is 2.

    Returns
    -------
    list
        Returns a list of the function outputs for each iteration.

    '''
    t0 = dt()
    niter = len(loop_args)
    
    if n_jobs == 1:
        return_list = []
        for i, l_arg in enumerate(loop_args):
            return_list.append(function(**args, **l_arg))
            updateProgBar(i+1, niter, t0)
        return return_list
    
    if use_threads:
        pool = ThreadPoolExecutor(max_workers=n_jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=n_jobs)
    with pool:
        futures = []
        for l_arg in loop_args:
            args.update(l_arg)
            futures.append(pool.submit(function,  **args))
        if verbose:
            for k, f in enumerate(as_completed(futures)):
                updateProgBar(k+1, niter, t0)
    out = []
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return out


def p_prog_simp_memorySafe(args, loop_args, function, n_jobs=2):
    '''
    Parallel Progress bar. The same as p_prog_simp, except it is more explicit
    in it's definition. This function was developed to try and limit
    memory consumption for parallel operations. I'm not 100% sure if it
    accomplishes that... More testing is needed.

    Parameters
    ----------
    args : dict
        Dictionary of stationary arguments (i.e. the arguments that are
        constant throughout the loop).
    loop_args : list
        A list of dictionaries that contain the arguments that may change at 
        any point during the loop (this is the iterator, and any additional
        arguments that may have different values at different stages of the
        loop).
    function : function
        The function being being processed in the loop (args and loop args will
        be passed to this function).
    n_jobs : int, optional
        The number of parallel processes to run. The default is 2.

    Returns
    -------
    list
        Returns a list of the function outputs for each iteration.

    '''
    t0 = dt()
    niter = len(loop_args)
    
    if n_jobs == 1:
        return_list = []
        for i, l_arg in enumerate(loop_args):
            return_list.append(function(**args, **l_arg))
            updateProgBar(i+1, niter, t0)
        return return_list
        
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        current_futures = []
        out = []
        jobs_complete = 0
        jobs_assigned = 0
        next_job = 0
        while (len(out) < niter):
            if len(current_futures) < n_jobs:
                if jobs_assigned < niter:
                    args.update(loop_args[next_job])
                    current_futures.append(pool.submit(function,  **args))
                    next_job += 1
                    jobs_assigned += 1
            for spot, fut in enumerate(current_futures):
                if fut.done():
                    try:
                        out.append(fut.result())
                    except Exception as e:
                        out.append(e)
                    del current_futures[spot]
                    jobs_complete += 1
                    updateProgBar(jobs_complete, niter, t0)
    return out
