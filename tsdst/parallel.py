from concurrent.futures import ProcessPoolExecutor, as_completed
from timeit import default_timer as dt
from .utils import updateProgBar


def parallel_progress(array, function, n_jobs=5,
                      use_kwargs=False, front_num=1):
    t0 = dt()
    niter = len(array)
    if front_num > 0:
        front = [(function(**a) if use_kwargs else function(a), updateProgBar(i+1, niter, t0))[0]
                 for i, a in enumerate(array[:front_num])]
    
    if n_jobs == 1:
        return front + [(function(**a) if use_kwargs else function(a), updateProgBar(j+2, niter, t0))[0]
                        for j, a in enumerate(array[front_num:])]
    
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        
        for k, f in enumerate(as_completed(futures)):
            updateProgBar(k+2, niter, t0)

    out = front + []

    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return out


def p_prog_simp(args, loop_args, function, n_jobs=5):
    '''args is a dictionary, loop_args is a list of dictionaries'''
    t0 = dt()
    #tma.start()
    #snap_1 = tma.take_snapshot()
    niter = len(loop_args)
    
    if n_jobs == 1:
        return_list = []
        for i, l_arg in enumerate(loop_args):
            return_list.append(function(**args, **l_arg))
            updateProgBar(i+1, niter, t0)
        return return_list
        
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = []
        for l_arg in loop_args:
            args.update(l_arg)
            futures.append(pool.submit(function,  **args))
        for k, f in enumerate(as_completed(futures)):
            updateProgBar(k+1, niter, t0)
        #snap_2 = tma.take_snapshot()
    out = []
    #top_stats = snap_2.compare_to(snap_1, 'lineno')
    #for stat in top_stats[:3]:
    #    print(stat)
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return out


def p_prog_simp_explicit(args, loop_args, function, n_jobs=5):
    t0 = dt()
    #tma.start()
    #snap_1 = tma.take_snapshot()
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
        #snap_2 = tma.take_snapshot()

    #top_stats = snap_2.compare_to(snap_1, 'lineno')
    #for stat in top_stats[:3]:
    #    print(stat)
    return out