import numdifftools as nd
import numpy as np
import warnings

from scipy.stats import norm


def signIntervals(fun, lower=-50, upper=50, step=0.01, fun_args=None):
    '''
    Returns the boundaries of all possible roots in a function. Works well
    with Brent root method.

    Parameters
    ----------
    fun : function
        The function to find the roots of.
    lower : float, optional
        The minimum value to investigate. The default is -50.
    upper : float, optional
        The maximum value to investigate. The default is 50.
    step : float, optional
        The interval between values in (lower, upper) to investigate.
        The default is 0.01.
    fun_args : dict
        Optional arguments to pass to fun.

    Returns
    -------
    bounds : list
        A list of tuples containing the upper and lower bounds of possible
        roots.

    '''
    if fun_args is None:
        fun_args = {}
    x = np.arange(lower, upper, step)
    steps = len(x)
    fx = np.zeros(steps)
    for i in range(steps):
        fx[i] = fun(x[i], **fun_args)

    curr = np.sign(fx[0])
    changeList = np.zeros(1)

    for j in range(steps):
        if np.sign(fx[j]) != curr:
            changeList = np.concatenate((changeList, np.array(j).reshape(1, )))
            curr = np.sign(fx[j])

    bounds = []
    for k in range(len(changeList[:-1])):
        bounds.append((x[int(changeList[k])], x[int(changeList[k + 1])]))

    return bounds


def Brent_1DRoot(fun, opt=None, print_=True, fun_args=None):
    '''
    Brent Method for Roots

    This function works well with signIntervals. For example, if the function
    "func" is x^2 - 5, the following code snippet will return both roots:
        bds = signIntervals(easy)
        for i in bds:
            Brent_1DRoot(easy, opt={'bounds': i})

    Parameters
    ----------
    fun : function
        The function to find the roots of.
    opt : dict, optional
        Optional arguments for the optimization. The default is None.
    print_ : bool, optional
        Print Results. The default is True.
    fun_args : dict
        Optional arguments to pass to fun.

    Raises
    ------
    ValueError
        Raised if there are invalid bounds, or no valid root.
    TypeError
        Raised if bounds is not a valid type.

    Returns
    -------
    final_res : dict
        A dictionary containing the results.

    '''
    if fun_args is None:
        fun_args = {}
    
    dom_tol = 1e-8
    maxiter = 10000
    bounds = [-10000.0, 10000.0]
    fun_tol = 1e-8
    if opt is not None:
        opt_keys = list(opt.keys())
        if 'dom_tol' in opt_keys:
            dom_tol = opt['dom_tol']
        if 'maxiter' in opt_keys:
            maxiter = opt['maxiter']
        if 'bounds' in opt_keys:
            bounds = opt['bounds']
        if 'fun_tol' in opt_keys:
            fun_tol = opt['fun_tol']

    try:
        if len(bounds) != 2:
            raise ValueError("Bounds option must be of length 2")
        if not isinstance(bounds, (tuple, list, np.ndarray)):
            raise TypeError("""must be array or list-like
                                (np.ndarray, tuple, or list""")
    except KeyError:
        bounds = [-10000.0, 10000.0]

    a = min(bounds)
    b = max(bounds)
    fa = fun(a, **fun_args)
    fb = fun(b, **fun_args)

    if fa*fb >= 0:
        raise ValueError("""Root is not bracketed, or root does not exist.
                          (try changing the interval)""")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = (float(a) + float(b))/2
    fc = float(fa)
    mflag = True
    cur_iter = 1
    d = float(c)

    while (abs(b - a) > dom_tol) and cur_iter < maxiter:
        if abs(fa - fc) > dom_tol and abs(fb - fc) > dom_tol:
            # inverse quadratic interpolation
            q1 = a*fb*fc / ((fa - fb)*(fa - fc))
            q2 = b*fa*fc / ((fb - fa)*(fb - fc))
            q3 = c*fa*fb / ((fc - fa)*(fc - fb))
            s = q1 + q2 + q3
        else:
            # secant method
            s = b - fb*((b - a)/(fb - fa))

        delta = abs(2*fun_tol*abs(b))
        if ((s < ((3*a + b)/4) and s > b) or
           (mflag and abs(s - b) >= abs(b - c)/2) or
           (not mflag and abs(s - b) >= abs(c - d)/2) or
           (mflag and abs(b - c) < abs(delta)) or
           (not mflag and abs(c - d) < abs(delta))):
            # bisection method
            s = (a + b)/2
            mflag = True
        else:
            mflag = False
        fs = fun(s, **fun_args)
        d = float(c)
        c = float(b)

        if fa*fs < 0:
            b = float(s)
            fb = float(fs)
        else:
            a = float(s)
            fa = float(fs)

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        cur_iter += 1

    if cur_iter >= maxiter:
            warnings.warn('Warning: May not have found roots. Number of' +
                          'iterations exceeded maximum', Warning)

    final_res = {'root': a,
                 'Final Function Value': fa,
                 'Domain Tolerances': abs(b - a),
                 'Iterations': cur_iter,
                 'a:b:s': (a, b, s)}
    if print_:
        print("Inputs: \n", opt, "\n\nFinal Results\n", final_res)

    return final_res


def AdaptiveNMorBrent(fun, start=None, opt=None, print_=True, fun_args=None):
    '''
    Nelder-Mead Method and Brent method (if one-dimensional) for finding the
    minimum of a function.

    Parameters
    ----------
    fun : function
        The function to find the roots of.
    start : numpy array
        The starting values for the optimization. Must be equal in length to
        the number of parameters being optimized.
    opt : dict, optional
        Optional arguments for the optimization. The default is None.
    print_ : bool, optional
        Print Results. The default is True.
    fun_args : dict
        Optional arguments to pass to fun.

    Raises
    ------
    ValueError
        Raised if there are invalid bounds, or no valid root.
    TypeError
        Raised if bounds is not a valid type.

    Returns
    -------
    final_res : dict
        A dictionary containing the results.

    '''
    dom_tol = 1e-8
    maxiter = 10000
    bounds = [-10000.0, 10000.0]
    fun_tol = 1e-8
    if opt is not None:
        opt_keys = list(opt.keys())
        if 'dom_tol' in opt_keys:
            dom_tol = opt['dom_tol']
        if 'maxiter' in opt_keys:
            maxiter = opt['maxiter']
        if 'bounds' in opt_keys:
            bounds = opt['bounds']
        if 'fun_tol' in opt_keys:
            fun_tol = opt['fun_tol']
    

    if start is None or len(start) == 1:
        # Continue with Brent Method
        try:
            bounds = opt['bounds']
            if len(bounds) != 2:
                raise ValueError("Bounds option must be of length 2")
            if not isinstance(bounds, (tuple, list, np.ndarray)):
                raise TypeError("""must be array or list-like
                                (np.ndarray, tuple, or list)""")
        except KeyError:
            bounds = [-10000.0, 10000.0]
        # 1/(phi^2) where phi is the golden ratio
        golden = (1/((1 + 5**0.5) / 2))**2
        max_x = max(bounds)
        min_x = min(bounds)
        x = w = v = max_x
        fw = fv = fx = fun(x, **fun_args)
        delta = delta2 = 0
        cur_iter = 1

        while(cur_iter <= maxiter):
            mid = (min_x + max_x)/2
            f1 = (dom_tol * abs(x) + dom_tol)/4
            f2 = 2 * f1
            if abs(x - mid) <= (f2 - (max_x - min_x)/2):
                break

            if abs(delta2) > f1:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2 * (q - r)
                if q > 0:
                    p = -p
                q = abs(q)
                td = delta2
                delta2 = delta
                if ((abs(p) >= abs(q * td / 2)) or
                   (p <= q * (min_x - x)) or
                   (p >= q * (max_x - x))):
                    delta2 = min_x - x if (x >= mid) else max_x - x
                    delta = golden * delta2
                else:
                    delta = p / q
                    u = x + delta
                    if (u - min_x) < f2 or (max_x - u) < f2:
                        delta = -abs(f1) if (mid - x) < 0 else abs(f1)
            else:
                delta2 = min_x - x if x >= mid else max_x - x
                delta = golden * delta2

            u = (x + delta if abs(delta) >= f1
                 else (x + abs(f1) if delta > 0 else x - abs(f1)))
            fu = fun(u, **fun_args)

            if fu <= fx:
                if u >= x:
                    min_x = x
                else:
                    max_x = x
                v = w
                w = x
                x = u
                fv = fw
                fw = fx
                fx = fu
            else:
                if u < x:
                    min_x = u
                else:
                    max_x = u

                if (fu <= fw) or (w == x):
                    v = w
                    w = u
                    fv = fw
                    fw = fu
                elif (fu <= fv) or (v == x) or (v == w):
                    v = u
                    fv = fu
            cur_iter += 1

        if cur_iter >= maxiter:
            warnings.warn('Warning: May not have converged. Number of' +
                          'iterations exceeded maximum', Warning)

        final_res = {'Minimum': x,
                     'Final Function Value': fx,
                     'Domain Tolerances': abs(x - min_x),
                     'Iterations': cur_iter,
                     'Method': "Brent"}

    else:

        start = np.array(start).reshape(1, -1)
        num_parms = start.size

        alpha = 1
        beta = 1 + (2 / num_parms)
        gamma = 0.75 - (1 / (2 * num_parms))
        delta = 1 - (1 / num_parms)

        # intialize simplex, function values, and centroid

        h_j = np.where(start != 0, 0.05, 0.00025).reshape(1, -1)
        e_j = np.identity(num_parms)

        simplex = np.concatenate((start, start + (h_j * e_j)),
                                 axis=0)
        sx_w_fval = np.concatenate((simplex, np.empty((num_parms + 1, 1))),
                                   axis=1)
        # sx_w_fval[:, -1] = np.nan
        
        for i in range(num_parms + 1):
            sx_w_fval[i, -1] = fun(simplex[i, :], **fun_args)

        # initial sort
        sx_w_fval = sx_w_fval[sx_w_fval[:, -1].argsort()]

        # when stdev between vertices is small
        domain_conv = False
        # when the function evaluations between vertices is small
        funval_conv = False
        # when the loop reaches a max number of iterations
        cur_iter = 0

        while cur_iter < maxiter and not (funval_conv and domain_conv):
            shrink = False
            centroid = np.mean(sx_w_fval[:-1, :-1], axis=0)
            # Reflection
            x_r = centroid + alpha*(centroid - sx_w_fval[-1, :-1])
            f_x_r = fun(x_r, **fun_args)
            if f_x_r < sx_w_fval[-2, -1] and f_x_r >= sx_w_fval[0, -1]:
                # if reflected is better than worst, replace
                sx_w_fval[-1, :] = np.hstack((x_r, f_x_r))
            elif f_x_r < sx_w_fval[0, -1]:
                # Expansion
                # if reflected point is best yet, try expanding
                x_e = centroid + beta*(x_r - centroid)
                f_x_e = fun(x_e, **fun_args)
                if f_x_e < f_x_r:
                    # if expanded is better, replace with expanded
                    sx_w_fval[-1, :] = np.hstack((x_e, f_x_e))
                else:
                    # replace with reflected
                    sx_w_fval[-1, :] = np.hstack((x_r, f_x_r))
            else:
                # Contraction
                # First, check outside contraction
                # if reflected value is between 2nd worst and worst
                if sx_w_fval[-2, -1] <= f_x_r and f_x_r < sx_w_fval[-1, -1]:
                    x_c_o = centroid + gamma*(x_r - centroid)
                    f_x_c_o = fun(x_c_o, **fun_args)
                    if f_x_c_o <= f_x_r:
                        sx_w_fval[-1, :] = np.hstack((x_c_o, f_x_c_o))
                    else:
                        shrink = True
                else:
                    # Check inside contraction
                    x_c_i = centroid - gamma*(x_r - centroid)
                    f_x_c_i = fun(x_c_i, **fun_args)
                    if f_x_c_i < sx_w_fval[0, -1]:
                        sx_w_fval[-1, :] = np.hstack((x_c_i, f_x_c_i))
                    else:
                        shrink = True

                if shrink:
                    shrunk_vals = (sx_w_fval[0, :-1] +
                                   delta*(sx_w_fval[1:, :-1] -
                                          sx_w_fval[0, :-1]))
                    for i in range(num_parms):
                        sx_w_fval[i + 1, :] = np.hstack((shrunk_vals[i, :],
                                                         fun(shrunk_vals[i, :],
                                                         **fun_args)))

            sx_w_fval = sx_w_fval[sx_w_fval[:, -1].argsort()]

            std_dom = np.std(sx_w_fval[:, :num_parms], axis=0, ddof=1)

            if all(np.array(std_dom)*2 < dom_tol):
                domain_conv = True

            if np.std(sx_w_fval[:, -1], ddof=1)*2 < fun_tol:
                funval_conv = True

            cur_iter += 1

        msg = "Successful"
        if cur_iter >= maxiter:
            msg = 'Warning: May not have converged. Number of' + \
                          'iterations exceeded maximum'
            warnings.warn(msg, Warning)

        final_params = sx_w_fval[0, :]

        final_res = {'Parameters': final_params[:-1],
                     'Final Function Value': final_params[-1],
                     'Domain Tolerances': std_dom,
                     'Function Value Tolerance': np.std(sx_w_fval[:, -1],
                                                        ddof=1),
                     'Iterations': cur_iter,
                     'Message': msg,
                     'Method': "NM"}
    if print_:
        print("Final Results\n", final_res, '\n')

    return final_res


def max_like(start, lf, opt=None, print_=True, like_args=None):
    '''
    Compute maximum likelihood estimation of a function and it's parameters.

    Parameters
    ----------
    start : numpy array
        The starting values for the optimization.
    lf : function
        The likelihood function. An array of the parameters for the function
        must be the first argument.
    opt : dict, optional
        Optional arguments for the optimization. The default is None.
    print_ : bool, optional
        Print the results. The default is True.
    fun_args : dict
        Optional arguments to pass to fun.

    Returns
    -------
    params : list
        A list containing the parameter MLEs, the standard error, and 
        95% CI for the parameters (each row is a parameter).

    '''
    if opt is None:
        opt = {'dom_tol': 1e-8, 'maxiter': 500}
    # initialize variables
    param_res = se = params_ci = hess = None

    # log(start) may be used here aswell because optimization
    # will perform better with log
    
    if like_args is None:
        like_args = {}
    mle_res = AdaptiveNMorBrent(lf, start, opt, print_, fun_args=like_args)
    # note: Hessian needs to be given the parameters
    # in the same scale as the optimization
    # to be able to find the correct derivative
    param_res = 0
    if mle_res['Method'] == "Brent":
        param_res = mle_res["Minimum"]
    else:
        param_res = mle_res["Parameters"]

    try:
        hess = nd.Hessian(lf)(param_res, **like_args)
        np.linalg.inv(hess)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            h = False
        else:
            raise

    if np.isnan(hess).any() or not np.isfinite(hess).any():
        h = False
    else:
        if not ((np.linalg.eigvals(hess))).all() > 0:
            h = False
        else:
            h = True

    # se = sqrt(diagonal(covariance matrix)), or,
    # se = sqrt(diagonal(inv(hessian)))
    #
    # ci = estimate +- N(confidence limit,0,1)*se
    #
    # note: when minimizing negative log liklihood in MLE,
    # use the sqrt of the diagonal of the inverse of the Hessian matrix.
    # If maximizing log-likelihood, use negative hessian instead to find
    # standard errors
    #
    # other note: if you perform a log-transformation on the parameters
    # during MLE optimization, you must multiply standard error
    # by exp(paramaters), as well as exponentiate the CI limits.

    num_params = len(np.array(param_res).reshape(-1))
    if h:
        se = np.sqrt(np.diagonal(np.linalg.inv(hess)))
        params_ci = (param_res + np.array([-1, 1]).reshape(2, 1).dot(
                (norm.ppf(1-(1-0.95)/2)*se).reshape(1, num_params))).T

    else:
        se = None
        params_ci = None

    params = [param_res, se, params_ci]

    return params
