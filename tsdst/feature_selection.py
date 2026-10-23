import numpy as np
import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from .distributions import (likelihood_bernoulli, likelihood_poisson,
                            likelihood_gaussian)
from .metrics import aicCalc
from .parallel import p_prog_simp
#from .tmath import percentIncrease, powerset


def naiveVarDrop(X, searchCols, tol=0.0001, standardize=False, asList=False,
                 print_=False):
    """
    Drop columns based on which columns have variance below the threshold.

    Parameters
    ----------
    X : pandas dataframe
        Feature (or design) matrix.
    searchCols : list or list-like (str)
        The columns to search. If None, use all columns. Default is None.
    tol : float, optional
        The threshold for variance to decide which columns to keep or drop.
        The default is 0.0001.
    asList : bool, optional
        Return only the list of columns to be dropped. The default is False.
    print_ : bool, optional
        Print the columns to be dropped. The default is False.

    Returns
    -------
    list or dataframe
        Either list of columns to be dropped or dataframe with columns removed.

    """
    cols_to_drop = []
    if searchCols is None:
        searchCols = list(X.columns)
    for i in searchCols:
        var = X.loc[:, i].var(ddof=1)
        if var < tol:
            cols_to_drop.append(i)
    if print_:
        print("Dropped " + str(len(cols_to_drop)) + " low-var Columns")
    if asList:
        return cols_to_drop
    else:
        return X.drop(cols_to_drop, axis=1)


def naiveScoreDrop(X, scores, tol=0.001, asList=False):
    """
    Drop columns based on which columns have scores below the threshold. This 
    could be used with any arbitrary score function, where scores is a 
    1-column dataframe, in which the index are column names, and the values
    are the scores.

    Parameters
    ----------
    X : pandas dataframe
        Feature (or design) matrix.
    scores : pandas dataframe
        The score results for each column.
    tol : float, optional
        The threshold for variance to decide which columns to keep or drop.
        The default is 0.0001.
    asList : bool, optional
        Return only the list of columns to be dropped. The default is False.

    Returns
    -------
    list or dataframe
        Either list of columns to be dropped or dataframe with columns removed.
    """
    cols_to_drop = []
    for i in scores.index:
        score = scores.loc[i].values
        if score < tol:
            cols_to_drop.append(i)
    if asList:
        return cols_to_drop
    else:
        return X.drop(cols_to_drop, axis=1)
    
    
def getHighCorrs(corr_mat, corr_thres, split_key="..&.."):
    """
    Given a correlation matrix, return the
    correlations that are above the threshold.
    
    To be used before dropHighCorrs.

    Parameters
    ----------
    corr_mat : pandas dataframe
        The correlation matrix.
    corr_thres : float
        The threshold for correlation to decide which columns to keep or drop.
        Between 0 and 1.
    split_key : str, optional
        The unique key to use to join the column names together, for example,
        if split_key was '..&..', then the index for that correlation would be
        'col1..&..col2'. The default is "..&..".

    Returns
    -------
    top_corr : pandas dataframe
        A dataframe containing a list of the top correlations from the
        correlation matrix.

    """
    top_corr = {}
    columns = corr_mat.columns
    for i in corr_mat.columns:
        for j in columns:
            if i == j:
                continue
            if np.isnan(corr_mat.loc[i, j]):
                continue
            cur_corr = corr_mat.loc[i, j]
            if abs(cur_corr) >= corr_thres:
                top_corr[split_key.join([i, j])] = cur_corr
        columns = columns.drop(i)
    return top_corr


def dropHighCorrs(X, top_corr, split_key="..&..", asList=False, print_=False):
    """
    Remove high inter-correlations from the dataset. When dropping a column,
    the column with the lowest variance is selected.
    
    To be used after getHighCorrs.

    Parameters
    ----------
    X : pandas dataframe
        The data in tabular form (the feature or design matrix of which the 
        correlation is being evaluated).
    top_corr : dataframe
        The output from the getHighCorrs function, or, a dataframe containing a
        list of the high correlations, where the index is a list of the column
        names concatenated together by split_key, for example, 'col1..&..col2'.
    split_key : str, optional
        The unique key to use to join the column names together, for example,
        if split_key was '..&..', then the index for that correlation would be
        'col1..&..col2'. The default is "..&..".
    asList : bool, optional
        Return only the list of columns to be dropped. The default is False.
    print_ : bool, optional
        Print the columns to be dropped. The default is False.

    Returns
    -------
    list or dataframe
        Either list of columns to be dropped or dataframe with columns removed.

    """
    cols_to_drop = []
    for i in top_corr.keys():
        cols = i.split(split_key)
        var1 = np.nanvar(X.loc[:, cols[0]])
        var2 = np.nanvar(X.loc[:, cols[1]])
        if var1 < var2:
            cols_to_drop.append(cols[0])
        else:
            cols_to_drop.append(cols[1])
    if print_:
        print("Dropped " + str(len(cols_to_drop)) + " high-corr Columns")
    if asList:
        return cols_to_drop
    else:
        return X.drop(cols_to_drop, axis=1)


def dropCorrProcedure(X, corr_thres, split_key="..&..", 
                      asList=False, print_=False):
    # TODO: add an option for dropping columns that have a low correlation with
    # the target
    """
    Calculates the inter-correlation of the columns in a matrix/dataframe, and then
    drops columns that are above a given threshold.

    Parameters
    ----------
    X : numpy array or pandas dataframe
        The data in tabular form (the feature or design matrix). If not a 
        dataframe, it will be converted.
    corr_thres : float
        The threshold for correlation to decide which columns to keep or drop.
        Between 0 and 1.
    split_key : str, optional
        The unique key to use to join the column names together, for example,
        if split_key was '..&..', then the index for that correlation would be
        'col1..&..col2'. The default is "..&..".
    asList : bool, optional
        Return only the list of columns to be dropped. The default is False.
    print_ : bool, optional
        Print the columns to be dropped. The default is False.

    Returns
    -------
    dropped_cor : dataframe
        Dataframe with High correlation columns dropped.

    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X.copy())
    corr_mat = X.corr()
    top_corr = getHighCorrs(corr_mat, corr_thres, split_key=split_key)
    dropped_cor = dropHighCorrs(X, top_corr, split_key=split_key,
                                asList=asList, print_=print_)
    return dropped_cor


def permutation_importance(fit_model, Xtest, Ytest,
                           metric_func, seed=None, sort="dsc"):
    """
    Performs permutation importance on a fitted model.
    
    Assumes that your metric function takes inputs in this order: ytrue, yperd.
    If it doesn't, write a simple wrapper that will. Also assumes a pretrained
    model with a predict method.
    
    Credit: https://explained.ai/rf-importance/
    
    This function was adapted from the above link.

    Parameters
    ----------
    fit_model : sklearn, (or similar)
        Can be any model that has a 'predict' method.
    Xtest : pandas Dataframe
        The observations in the feature or design matrix to perform the
        permutation test on.
    Ytest : pandas Series
        The observations in the response variable to perform the
        permutation test on.
    metric_func : function
        The function to evaluate your metric of interest.
    seed : int, optional
        Set Random Seed for reporducibility. The default is None.
    sort : str, optional
        Either 'asc' for ascending or 'dsc' for descending.
        The default is "dsc".

    Returns
    -------
    res : dataframe
        A dataframe containing the results of the permutation test.

    """
    Ypred = fit_model.predict(Xtest)
    baseline = metric_func(Ytest, Ypred)
    imps = []
    if seed is not None:
        np.random.seed(seed)
    for col in Xtest.columns:
        col_copy = Xtest.loc[:, col].copy()
        Xtest.loc[:, col] = np.random.permutation(Xtest.loc[:, col])
        Ypred = fit_model.predict(Xtest)
        score = metric_func(Ytest, Ypred)
        Xtest.loc[:, col] = col_copy
        imps.append(baseline - score)
    res = pd.DataFrame(imps, index=Xtest.columns, columns=["Importance"])
    # Could reduce the sorting to a one-liner, but this way someone
    # could choose not to sort. Though, probably not a useful feature.
    if sort == "asc":
        res = res.sort_values('Importance', ascending=True)
    elif sort == "dsc":
        res = res.sort_values('Importance', ascending=False)
    return res


def permutation_importance_CV(model, X, Y, metric_func, num_splits=8, seed=None, sort="dsc"):
    """
    Performs a cross-validated permutation importance on a non-fitted model.
    
    Assumes that your metric function takes inputs in this order: ytrue, yperd.
    If it doesn't, write a simple wrapper that will. Also assumes a pretrained
    model with a predict method.
    
    Credit: https://explained.ai/rf-importance/
    
    This function was adapted from the above link.

    Parameters
    ----------
    model : sklearn, (or similar)
        Can be any model that has a 'predict' method. Send to the function 
        unfitted, only instantiated
    X : pandas Dataframe
        The observations in the feature or design matrix to perform the
        permutation test on.
    Y : pandas Series
        The observations in the response variable to perform the
        permutation test on.
    metric_func : function
        The function to evaluate your metric of interest.
    num_splits : int
        Number of CV folds to use (using stratified k-fold)
    seed : int, optional
        Set Random Seed for reporducibility. The default is None.
    sort : str, optional
        Either 'asc' for ascending or 'dsc' for descending.
        The default is "dsc".

    Returns
    -------
    res : dataframe
        A dataframe containing the results of the permutation test.

    """
    splits = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    pi_total = np.zeros((X.shape[1], 1))
    i = 0
    for train_idx, test_idx in splits.split(X, Y):
        Xtrain, Xtest = X.iloc[train_idx], X.iloc[test_idx]
        Ytrain, Ytest = Y.iloc[train_idx], Y.iloc[test_idx]
        fi_model = model.fit(Xtrain, Ytrain)
        pi = permutation_importance(fi_model, Xtest, Ytest,
                                metric_func, seed=seed, sort="None")
        pi_total += pi.values
        i += 1
        print("Completed " + str(i) + " of " + str(num_splits))
    pi_total /= num_splits
    pi_tot_df = pd.DataFrame(pi_total, index=X.columns, columns=["Importance"])
    # Could reduce the sorting to a one-liner, but this way someone
    # could choose not to sort. Though, probably not a useful feature.
    if sort == "asc":
        pi_tot_df = pi_tot_df.sort_values('Importance', ascending=True)
    elif sort == "dsc":
        pi_tot_df = pi_tot_df.sort_values('Importance', ascending=False)
    return pi_tot_df


def _calcForwardAICs(X, Y, model, metric, family):
    """
    A utility function for the forwardSelection algorithm. Calculates AIC
    values for a two-class classification model.

    Parameters
    ----------
    X : pandas Dataframe
        The feature or design matrix.
    Y : pandas Series
        The response variable.
    model : sklearn, or similar
        Any model that has a fit and predict (or predict_proba) method.
    metric : str
        The AIC metric to be used, for example, aic, aicc, bic, ebic, hastie, 
        or kwano.
    family : str
        The family of distributions to calculate log-likelihood from.

    Returns
    -------
    score : float
        The AIC (or variant) score.

    """
    mod = clone(model).fit(X, np.asarray(Y).ravel())

    ss = X.shape[0]
    ncoefs = X.shape[1] + int(getattr(mod, 'fit_intercept', not getattr(mod, 'has_constant', False)))
    if family == 'gaussian':
        ncoefs += 1  # Gaussian likelihood also estimates the residual variance.

    if family == "binomial":
        nllf = likelihood_bernoulli
        # Assumes the probability of positive class
        y_pred = mod.predict_proba(X)[:, 1]
    elif family == "poisson":
        nllf = likelihood_poisson
        y_pred = mod.predict(X)
    elif family == "gaussian":
        nllf = likelihood_gaussian
        y_pred = mod.predict(X)
    else:
        raise ValueError("Not a valid family")

    loglike = nllf(np.asarray(Y).ravel(), y_pred, neg=False)

    score = aicCalc(loglike, ncoefs, sample_size=ss, c=2, metric=metric)
    return score


def _doParallelForward(save_path, save_name, save_ext, metric, model, target_var,
                       new_predictor, use_probabilities, mod_cols, family,
                       Yprob=None, serialize_flavor='feather'):
    """Read a temporary dataframe and score one forward-selection candidate."""
    if serialize_flavor not in ('feather', 'parquet'):
        raise ValueError('Serialization must use feather or parquet')
    data = getattr(pd, 'read_' + serialize_flavor)(save_path + save_name + save_ext)
    columns = [new_predictor] if isinstance(new_predictor, str) else list(new_predictor)
    if use_probabilities and Yprob is not None:
        predictors = data.loc[:, columns].copy()
        predictors.insert(0, 'Yprob', np.asarray(Yprob))
    else:
        predictors = data.loc[:, list(mod_cols) + columns]
    return _calcForwardAICs(predictors, data[target_var], model, metric, family)

def _prepare_for_parallel(XY, serialize_flavor='feather'):
    """Write a temporary dataframe for process workers.

    Feather and Parquet require the optional PyArrow dependency.
    The caller must remove the returned file and directory after use.
    """
    import tempfile
    if serialize_flavor not in ('feather', 'parquet'):
        raise ValueError('Serialization must use feather or parquet')
    path = tempfile.mkdtemp(prefix='tsdst-forward-')
    name, extension = '/data', '.feather' if serialize_flavor == 'feather' else '.parquet'
    getattr(XY, 'to_' + serialize_flavor)(path + name + extension)
    return name, path, extension, None


def forwardSelection(XY, target_var, model, metric='bic', family='gaussian',
                     verbose=True, n_jobs=1, early_stop=False, perc_min=0.01,
                     stop_at_p=1000, stop_when=5, use_probabilities=False,
                     return_type='all', serialize_flavor='feather', use_threads=True):
    """Add the feature that gives the lowest information criterion at each step.

    Stop when no candidate improves the criterion. With early_stop=True,
    also apply the feature limit and the consecutive small-improvement limit.
    Each candidate gets a separate estimator. Parallel workers use a temporary
    Feather or Parquet file. The function removes that file after the search.
    """
    if target_var not in XY or n_jobs < 1:
        raise ValueError('Supply a target column and a positive job count')
    if return_type not in ('all', 'list', 'data', 'model'):
        raise ValueError('return_type must be all, list, data, or model')
    if use_probabilities and family != 'binomial':
        raise ValueError('Probability compression requires the binomial family')
    if len(XY.columns) < 2:
        raise ValueError('Forward selection needs at least one candidate predictor')
    if early_stop and (stop_at_p < 1 or stop_when < 1 or not 0 <= perc_min < 1):
        raise ValueError('Use positive stopping counts and perc_min between zero and one')
    selected, remaining = [], [column for column in XY if column != target_var]
    best, small_steps = np.inf, 0
    arguments = None
    if n_jobs > 1:
        name, path, extension, _ = _prepare_for_parallel(XY, serialize_flavor)
        arguments = dict(save_path=path, save_name=name, save_ext=extension,
                         metric=metric, model=model, target_var=target_var, family=family,
                         use_probabilities=use_probabilities, serialize_flavor=serialize_flavor)
    try:
        limit = min(len(remaining), stop_at_p) if early_stop else len(remaining)
        for step in range(limit):
            probability = None
            if use_probabilities and len(selected) > 1:
                previous = clone(model).fit(XY[selected], XY[target_var])
                probability = previous.predict_proba(XY[selected])[:, 1]
            if n_jobs > 1:
                scores = p_prog_simp({**arguments, 'mod_cols': selected.copy(), 'Yprob': probability},
                         [{'new_predictor': [column]} for column in remaining],
                         _doParallelForward, n_jobs=n_jobs, verbose=verbose, use_threads=use_threads)
                for score in scores:
                    if isinstance(score, Exception):
                        raise score
            else:
                scores = []
                for column in remaining:
                    predictors = XY[selected + [column]]
                    if probability is not None:
                        predictors = pd.DataFrame({'Yprob': probability, column: XY[column]}, index=XY.index)
                    scores.append(_calcForwardAICs(predictors, XY[target_var], model, metric, family))
            winner = int(np.argmin(scores))
            current = scores[winner]
            if selected and current >= best:
                break
            improvement = (best-current)/max(abs(best), 1) if np.isfinite(best) else np.inf
            small_steps = small_steps + 1 if improvement < perc_min else 0
            selected.append(remaining.pop(winner))
            best = current
            if verbose:
                print('Selected {}: {} = {:.6g}'.format(selected[-1], metric, best))
            if early_stop and small_steps >= stop_when:
                break
    finally:
        if arguments is not None:
            os.remove(path + name + extension)
            os.rmdir(path)
    if return_type == 'list':
        return selected
    if return_type == 'data':
        return XY[selected].copy()
    fitted = clone(model).fit(XY[selected], XY[target_var])
    if return_type == 'model':
        return fitted
    return selected, fitted, XY[selected].copy()

## TODO: import vif from modeling and incorporate a vif drop  
def vifDrop():
    """Report that VIF-based feature removal is not implemented."""
    raise NotImplementedError('Use modeling.vif to inspect variance inflation factors')
