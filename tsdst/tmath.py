from __future__ import (division, generators, absolute_import,
                        print_function, with_statement, nested_scopes,
                        unicode_literals)

import numpy as np
import pandas as pd
import warnings

from itertools import chain, combinations
from numpy.linalg import inv
from scipy.stats import rankdata, kendalltau, gaussian_kde


def custom_abs(x):
    '''
    Innards of the absolute value function. (This is just here for fun).
    
    Parameters
    ----------
    x : signed float or int
        The number to be evaluated.

    Returns
    -------
    int or float
        The absolute value of x.

    '''
    m = x % (x**2 - x + 2)
    p = m % (x**2 + 2)
    return 2*p - x


def powerset(iterable):
    '''
    Create the powerset (all possible combinations) of a list.

    Parameters
    ----------
    iterable : list or list-like
        The collection of items to create a powerset from.

    Returns
    -------
    list of lists
        list containing all possible combinations.

    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
  

def percentIncrease(old, new):
    '''
    Calculates the percent increase between two numbers.

    Parameters
    ----------
    old : numeric
        The old number.
    new : numeric
        The new number.

    Returns
    -------
    numeric
        The percent increase (as a decimal).

    '''
    return (new - old)/old


# This will do a 'first' ranking of the data and return the ranks.
# For R-like rank functionality, import scipy's rankdata from 
# this module
def rank(x, small_rank_is_high_num=True, rank_from_1=True):
    '''
    Rank items in an array. Using the 'first' method, which ranks ties using
    the order of appearance. For rank functionality similar to R, see scipy's
    rankdata function (which is imported from this module for convenience).'

    Parameters
    ----------
    x : numpy array or array-like
        The array of values to be sorted.
    small_rank_is_high_num : bool, optional
        Smallest rank value is the highest/largest number.
        The default is True.
    rank_from_1 : bool, optional
        Use 1 as the top rank rather than 0. The default is True.

    Returns
    -------
    rk : numpy array
        An array containing the ranks.

    '''
    rk = np.argsort(np.argsort(x))
    if small_rank_is_high_num:
        rk = (len(x) - 1) - rk
    if rank_from_1:
        rk += 1
    return rk


def cov2cor(V):
    '''
    Translation of R's cov2cor function

    Parameters
    ----------
    V : numpy array
        Covariance matrix.

    Returns
    -------
    R : numpy array
        Correlation matrix.

    '''
    p, n = V.shape
    if len(V.shape) != 2 or p != n:
        raise ValueError('V is not a square matrix')
    Is = np.sqrt(1/np.diag(V))
    if np.any(~np.isfinite(Is)):
        warnings.warn("""Diagonal had 0, NA, or infinite entries. Non-finite
                      result is doubtful""")
    r = Is.reshape(-1, 1) * V * np.repeat(Is, p).reshape(p, p).T
    return r
    

def mann_whitney(x, y, y_is='groups'):
    '''
    Mann-Whitney sum-rank test.

    Parameters
    ----------
    x : numpy array
        Values to be evaluated. Either a single group, or an array containing
        both group values if y_is == 'groups'
    y : numpy array
        Values to be evaluated against x, or, if y_is == 'groups',
        then y is a boolean value indicating which group the values
        of x belong to.
    y_is : bool, optional
        Whether or not y is itself a group of values, or an indicator variable
        for identifying the groups within x. The default is 'groups'.

    Returns
    -------
    res : dict
        The results of the Mann-Whitney test.

    '''
    if isinstance(y, (list, tuple)):
        y = np.array(y)
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    if y_is == 'groups':
        X = x
        Y = y
        overall_rank = rankdata(X, 'average')
    else:
        if len(x.shape) >= 2:
            x = x.ravel()
        if len(y.shape) >= 2:
            y = y.ravel()
        X = np.concatenate((x, y))
        Y = np.concatenate((np.repeat(0, x.size), np.repeat(1, y.size)))
        overall_rank = rankdata(X, 'average')
    uni_groups = np.unique(Y)
    rank_group1 = overall_rank[Y == uni_groups[0]]
    rank_group2 = overall_rank[Y == uni_groups[1]]
    sum_ranks_g1 = rank_group1.sum()
    sum_ranks_g2 = rank_group2.sum()
    
    n1 = rank_group1.shape[0]
    n2 = rank_group2.shape[0]
    
    U1 = sum_ranks_g1 - (n1*(n1 + 1))/2
    U2 = sum_ranks_g2 - (n2*(n2 + 1))/2
    
    res = {'U1': U1,
           'U2': U2,
           'n1': n1,
           'n2': n2,
           'rank_group1': rank_group1,
           'rank_group2': rank_group2,
           'sum_ranks_g1': sum_ranks_g1,
           'sum_ranks_g2': sum_ranks_g2}
    
    res['max_U'] = 'U1' if U1 >= U2 else 'U2'
    return res


def biserial_rank_corr(x, y, y_is='groups'):
    '''
    Biserial Rank Correlation.

    Parameters
    ----------
    x : numpy array
        Values to be evaluated. Either a single group, or an array containing
        both group values if y_is == 'groups'
    y : numpy array
        Values to be evaluated against x, or, if y_is == 'groups',
        then y is a boolean value indicating which group the values
        of x belong to.
    y_is : bool, optional
        Whether or not y is itself a group of values, or an indicator variable
        for identifying the groups within x. The default is 'groups'.

    Returns
    -------
    r : float
        Biserial rank correlation.

    '''
    mw = mann_whitney(x, y, y_is=y_is)
    max_u = mw[mw['max_U']]
    r = (2*max_u)/(mw['n1']*mw['n2']) - 1
    return r


# mimics R cor function. Columns are correlation variables.
# X and Y must at least have the same length
def corr(x, y=None, method='pearson'):
    '''
    Mimics the R cor function to calculate correlations.

    Parameters
    ----------
    x : numpy array or pandas dataframe
        Either a matrix (to calculate correlations across columns) or a 1D
        array to compare with y.
    y : numpy array or pandas series, optional
        A 1D array to compare with a 1D x, or None if x is a matrix.
        The default is None.
    method : str, optional
        Type of correlation to compute. The default is 'pearson'.

    Raises
    ------
    ValueError
        Raised if method provided is invalid.

    Returns
    -------
    numpy array
        Array containing the correlation.

    '''
    cor_mat = None
    X = np.array(x, ndmin=2)
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if X.shape[0] == 1:
        X = X.T
    if y is not None:
        Y = np.array(y, ndmin=2)
        if Y.shape[0] == 1:
            Y = Y.T
        X = np.concatenate((X, Y), axis=1)
    if method == 'pearson':
        cor_mat = np.corrcoef(X, rowvar=False)
    elif method == 'spearman':
        M = np.apply_along_axis(rankdata, 0, X)
        D = np.eye(X.shape[1])*M.std(ddof=1, axis=0)
        Dinv = inv(D)
        cor_mat = Dinv.dot(np.cov(M, rowvar=False)).dot(Dinv)
    elif method == 'kendall':
        cor_mat = np.zeros((X.shape[1], X.shape[1]))
        corrs = []
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                c, _ = kendalltau(X[:, i], X[:, j])
                corrs.append(c)
        corrs = np.array(corrs)
        row, col = np.triu_indices(X.shape[1])
        cor_mat[row, col] = corrs
        cor_mat[col, row] = corrs
    else:
        raise ValueError('Not a valid correlation method. (%s) ' % method)
    
    if isinstance(x, pd.DataFrame):
        cor_mat = pd.DataFrame(cor_mat, index=x.columns, columns=x.columns)
    return cor_mat


def mode_histogram(data, bin_max='auto', edge_check=None, approx=False,
                   surround=True, surround_step=1):
    # TODO: make this work in a multivariate setting
    '''
    Calculates the mode (from a continuous distribution) using data provided
    from a histogram. Created because calculating mode from KDE estimate
    can be slow.

    Parameters
    ----------
    data : numpy array
        Array of data to calculate the mode from.
    bin_max : int or str, optional
        Number of histogram bins to pass to np.histogram. Can be string or
        integer. The default is 'auto'.
    edge_check : str or None, optional
        The edge of the largest histogram rectangle (mode) to use for
        comparing to the original data. For example, if the mode
        rectangle of the histogram created is between 1 and 3 on the x-axis,
        then 'left' uses 1 for comparison, 'right' uses 3, and 'center' uses
        the midpoint 2. If None, the optimal metod is used based on the number
        of bins. Default is None.
    approx : bool, optional
        If approx is True, use the histogram output as the mode. If False, find
        the closest value in data to the histogram value and use that as the
        mode.

    Returns
    -------
    float
        The mode of the array.

    '''
    if isinstance(bin_max, str):
        bins = bin_max
    else:
        if len(data) < bin_max:
            bins = len(data)
        else:
            bins = bin_max
    hist, edges = np.histogram(data, bins=bins)
    hist_max_idx = np.argmax(hist)
    if edge_check is None:
        if hist.shape[0] % 2 == 0:
            edge_check = "left"
        else:
            edge_check = "center"
    
    if surround:
        hist_max_indices = np.arange(hist_max_idx - surround_step, hist_max_idx + surround_step + 1, 1)
        if hist_max_indices[0] < 0:
            hist_max_indices = hist_max_idx
        hist_max_values = hist[hist_max_indices]
    else:
        hist_max_indices = hist_max_idx
    
    if edge_check == "center" or edge_check == "centre":
        max_edge = edges[hist_max_indices] + (edges[hist_max_idx + 1] - edges[hist_max_idx])/2
    elif edge_check == "left":
        max_edge = edges[hist_max_indices]
    elif edge_check == "right":
        max_edge = edges[hist_max_indices + 1]
    else:
        raise ValueError("Not a valid edge_check. See help(mode_histogram)")
    
    if surround:
        max_edge = np.sum(max_edge*(hist_max_values/hist_max_values.sum()))
    
    if approx:
        mode = max_edge
    else:
        mode_idx = np.abs(np.asarray(data) - max_edge).argmin()
        mode = data[mode_idx]
    return mode
    

def mode_kde(data, kde_args=None):
    '''
    Calculate mode from gaussian kernel density estimate (using scipy
    gaussian_pde).

    Parameters
    ----------
    data : numpy array
        Array to calculate mode from.
    kde_args : dict
        A dictionary containing arguments for gaussian_kde (scipy)

    Returns
    -------
    mode : 
        A numeric value, which is the mode.

    '''
    if kde_args is None:
        kde_args = {}
    kernel = gaussian_kde(data, **kde_args)
    height = kernel.pdf(data)
    mode = data[np.argmax(height)]
    return mode


def norm(x, p, use_expo=False):
    '''
    Calculate LP norm of a vector.

    Parameters
    ----------
    data : numpy array
        A 1D numpy array of values.
    p : int
        The LP norm specification (1,2,3, ...).

    Returns
    -------
    numeric
        Value of LP norm on the vector.

    '''
    expo = 1/p
    if p == 1:
        return np.sum(np.abs(x))
    elif p >= 2:
        if p % 2 == 0:
            norm = np.sum(x**p)
        else:
            norm = np.sum(np.abs(x)**p)
        # Regression applications with L2 penalty often dismiss the full norm,
        # particularily in L2 regularization. I'll note here that there is no added benefit
        # to using norms higher than 2 in those cases. Odd norms (L3 and higher) are non-differntiable at the origin
        # and they behave the same as L1. L4 and higher norms have the same 
        # local meaning are differentiable, thus providing no additional useful
        # information in regularization.
        if use_expo:
            return norm**expo
        else:
            return norm
    elif p == 0:
        # Note: Hastie defines the L_0 norm as the number of elements
        # though this is not derived mathmatically
        return len(x)
    else:
        raise('Not a valid norm')
        
        
def norm_der(x, p, use_expo=False):
    norm_der = (p*np.abs(x)**(p-1))*np.sign(x)
    if use_expo:
        norm_der = norm_der*((1/p)*np.sum(np.abs(x)**p)**(1/p - 1))
    return norm_der

        
def mahalanobis(data, produce=None):
    '''
    Calculate mahalanobis distance on a matrix of column vectors. 
    Assumes that rows are observations and columns are features.
    
    Parameters
    ----------
    data : numpy array or pandas dataframe
        The data to calculate distances on (columns are variables, rows are
        observations).
    produce : str, optional
        Variation of the output to produce, either `squared`, `leverage',
        or `sqrt` (None). The default is None.

    Returns
    -------
    numpy array
        Array containing the distances.

    '''
    arr = np.array(data).reshape(data.shape[0], -1)
    cent = arr - arr.mean(axis=0)
    covmat = np.cov(cent, rowvar=False)
    invcov = None
    if arr.shape[1] == 1:
        invcov = 1/covmat
    else:
        try:
            invcov = np.linalg.inv(covmat)
        except np.linalg.LinAlgError:
            invcov = np.linalg.pinv(covmat)
    md2 = np.sum(cent.dot(invcov) * cent, axis=1)
    if produce == "squared":
        return md2
    elif produce == "leverage":
        n = data.shape[0]
        return ((md2/(n - 1)) + (1/n))
    else:
        return np.sqrt(md2)


def scaling(x, a=0, b=1, percentile=(0, 100),
            center=None, scale=None, ddof=0):
    '''
    Common scaling algorithm. Either min-max (with any desired bound),
    standardizing, max absolute scaling, or robust scaling.

    Parameters
    ----------
    x : numpy array
        The data.
    a : float, optional
        The minimum value desired. The default is 0.
    b : float, optional
        The maximum value desired. The default is 1.
    percentile : list or array-like, optional
        Provide percentile bounds on values. The default is (0, 100).
    center : float or str, optional
        Either a value representing the centrality measure, or a string
        for the measure to be computed, either mean or median.
        If None, use the minimum of the percentile bounds.
        The default is None.
    scale : float or str, optional
        Either a value representing the dispersion measure, or a string
        for the measure to be computed, either std (standard deviation),
        var (variance), or maxAbs.
        If None, use the absolute difference of the percentile bounds.
        The default is None.
    ddof : int, optional
        Degrees of freedom for the variance/std. dev measure. The default is 0.

    Raises
    ------
    TypeError
        Raised if center is an invalid type.

    Returns
    -------
    scaled : numpy array
        The scaled values.

    '''
    lower = upper = centVal = scaleVal = None
    if center is not None:
        try:
            centVal = x - center
        except TypeError:
            if center == "mean":
                centVal = x - np.nanmean(x)
            elif center == "median":
                centVal = x - np.nanmedian(x)
            else:
                raise TypeError("""center must be a string, either 'mean' or
                                'median', else a number (central tendency)""")
    else:
        (lower, upper) = np.nanpercentile(x, percentile)
        centVal = x - lower

    if scale is not None:
        try:
            scaleVal = scale/1
        except TypeError:
            if scale == "maxAbs":
                scaleVal = np.nanmax(np.abs(x))
            elif scale == "std":
                scaleVal = np.nanstd(x, ddof=ddof)
            elif scale == "var":
                scaleVal = np.nanvar(x, ddof=ddof)
    else:
        if lower is None:
            (lower, upper) = np.nanpercentile(x, percentile)
        scaleVal = upper - lower

    scaled = (b - a)*centVal/scaleVal + a

    return scaled
        