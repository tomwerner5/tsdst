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


def mode_histogram(data, delta=75, dataMax_thres=1e9, median_thres=None):
    # TODO: make this work in a multivariate setting
    # TODO: Go through the function again to provide more useful documentation
    '''
    Calculates the mode (from a continuous distribution) using data provided
    from a histogram. Created because calculating mode from KDE estimate
    can be slow.

    Parameters
    ----------
    data : numpy array
        Array of data to calculate the mode from.
    delta : int, optional
        Used to calculate histogram bin width. The default is 75.
    dataMax_thres : float, optional
        Max value to consider in hisogram calculation (for the right edge).
        The default is 1e9.
    median_thres : float, optional
        Used to limit the center of the distribution, if median is a really 
        high number. The default is None.

    Returns
    -------
    float
        The mode of the array.
    hist : numpy array
        The histogram used to calculate mode.

    '''
    bin_min = np.round(np.floor(np.min(data/delta))*delta, 3)
    bin_max = np.round(np.ceil(np.max(data) / delta + 1)*delta, 3)
    data_max = np.max(data)
    if median_thres is None:
        if data_max > dataMax_thres:
            bin_max = dataMax_thres
    else:
        data_median = np.median(data)
        if data_max < dataMax_thres:
            pass
        elif data_max >= dataMax_thres and data_median < median_thres:
            bin_max = np.round(np.ceil((2*data_median) / delta + 1) * delta, 3)
    bin_array = np.arange(bin_min, bin_max, delta)
    hist = np.histogram(data, bin_array)
    return hist[1][np.where(hist[0] == np.max(hist[0]))][0], hist


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


def norm(data, p):
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
    if p == 1:
        return np.sum(np.abs(data))
    elif p == 2:
        return np.sum(data*data)**0.5
    elif p >= 3:
        if p % 2 == 0:
            return np.sum(data**p)**(1/p)
        else:
            return np.sum(np.abs(data)**p)**(1/p)
    elif p == 0:
        return 0
    else:
        raise('Not a valid norm')

        
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
        