import numpy as np
import pandas as pd
import warnings

from copy import deepcopy as copy
from scipy.stats import t, norm as normal
from sklearn.metrics import (f1_score, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.model_selection import (StratifiedKFold, KFold,
                                     ShuffleSplit, StratifiedShuffleSplit)
from timeit import default_timer as dt

from .metrics import (cox_snell_r2, nagelkerke_r2, tjur_r2, mcfadden_r2,
                      conf_mat_metrics, bias, rpmse, r2, adj_r2, top_20p,
                      number_of_nonzero_coef)
from .nn.activations import sigmoid
from .utils import reshape_to_vector


DEFAULT_CLASSIFICATION_METRICS = ['Accuracy',
                                  'F1',
                                  'Sens/Recall',
                                  'Specificity',
                                  'PPV',
                                  'NPV',
                                  'AUC'
                                ]

DEFAULT_REGRESSION_METRICS = ['Bias',
                              'RPMSE',
                              'R2',
                              'Adj. R2']


def scoreModel(X, Y, model, metrics=None,
               thres=None, mtype='classification',
               print_=True, avg="weighted"):
    """Score a model using the given metrics.
    
    Parameters
    ----------
    X : pandas dataframe
        Feature or design matrix.
    Y : pandas series
        Response or Target variable.
    model : sklearn, or similar
        model to fit to the data. Must have fit, predict, and/or predict_proba
        methods, depending on the metrics.
    metrics : list, optional
        The metrics to include in the analysis.
        The default is ['Accuracy', 'F1', 'Sens/Recall', 'Specificity', 'ppv', 'npv', 'AUC'].
    thres : float, optional
        The threshold to use for classification metrics where it might apply.
        If threshold is None, the default predict method is used (which
        for most models uses a boundary at or equivalent to 0.5 probability).
        The default is None.
    mtype : str, optional
        The type of model, currently either classification or regression.
        The default is 'classification'.
    print_ : bool, optional
        Print progress as applicable. The default is True.
    avg : str, optional
        for metrics where it applies, such as AUC, how to calculate the
        metric (see sklearn docs for more details, particularly with AUC).
        The default is 'weighted'.

    Returns
    -------
    res : dict
        A dictionary containing the results for each metric.
    """
    if metrics is None:
        if mtype == 'classification':
            metrics = DEFAULT_CLASSIFICATION_METRICS
        elif mtype == 'regression':
            metrics = DEFAULT_REGRESSION_METRICS
        else:
            raise ValueError('Metrics must be of mtype `classification` or '
                             '`regression`')

    res = {}
    args = {'average': avg,
            'conf_metric': None,
            'X': X,
            'y_true': Y,
            'y_pred': None,
            'y_score': None,
            'll_est': None,
            'll_null': None, 
            'y_prob_nullmod': None,
            'n_obs': None,
            'rsquared': None,
            'cs_r2': None,
            'model': model}
    All_Metrics = {
            'Accuracy': {
                         'Function': accuracy_score,
                         'arguments': ['y_true', 'y_pred']
                        },
            'F1': {
                   'Function': f1_score,
                   'arguments': ['y_true', 'y_pred']
                  },
            'Sens/Recall': {
                            'Function': conf_mat_metrics,
                            'arguments': ['y_true', 'y_pred', 'conf_metric']
                           },
            'Specificity': {
                            'Function': conf_mat_metrics,
                            'arguments': ['y_true', 'y_pred', 'conf_metric']
                           },
            'PPV': {
                    'Function': conf_mat_metrics,
                    'arguments': ['y_true', 'y_pred', 'conf_metric']
                   },
            'NPV': {
                    'Function': conf_mat_metrics,
                    'arguments': ['y_true', 'y_pred', 'conf_metric']
                   },
            'Confusion Matrix': {
                         'Function': conf_mat_metrics,
                         'arguments': ['y_true', 'y_pred', 'conf_metric']
                        },
            'AUC': {
                    'Function': roc_auc_score,
                    'arguments': ['y_true', 'y_score', 'average']
                        }, 
            'Cox/Snell R2': {
                            'Function': cox_snell_r2,
                            'arguments': ['ll_est', 'll_null', 'n_obs']
                        },
            'McFadden R2': {
                            'Function': mcfadden_r2,
                            'arguments': ['ll_est', 'll_null']
                        },
            'Tjur R2': {
                        'Function': tjur_r2,
                        'arguments': ['y_true', 'y_score']
                       },
            'Nagelkerke R2': {
                              'Function': nagelkerke_r2,
                              'arguments': ['ll_null', 'n_obs', 'cs_r2', 'll_est']
                             },
            'Bias': {
                             'Function': bias,
                             'arguments': ['y_true', 'y_pred']
                            },
            'RPMSE': {
                             'Function': rpmse,
                             'arguments': ['y_true', 'y_pred']
                            },
            'R2': {
                             'Function': r2,
                             'arguments': ['y_true', 'y_pred']
                            },
            'Adj. R2': {
                             'Function': adj_r2,
                             'arguments': ['y_true', 'y_pred', 'X', 'rsquared']
                            },
            'Top 20%': {
                        'Function': top_20p,
                        'arguments': ['y_true', 'y_score']
            },
            'Avg Number of Used Features': {
                        'Function': number_of_nonzero_coef,
                        'arguments': ['X', 'model']
            }
        }
    
    y_temp = np.asarray(Y).ravel()
    args['y_true'] = y_temp
    if mtype == "classification":
        
        args['n_obs'] = Y.shape[0]
        args['y_score'] = model.predict_proba(X)[:, 1]
        
        if thres is None:
            args['y_pred'] = model.predict(X)
        else:
            args['y_pred'] = np.sum([args['y_score'] >= thres], axis=0)
        
        if any((met in ['McFadden R2',
                        'Nagelkerke R2',
                        'Cox/Snell R2'] for met in metrics)):
            
            args['y_prob_nullmod'] = np.repeat(np.mean(y_temp), args['n_obs'])
            args['ll_est'] = np.sum(y_temp*np.log(args['y_score']) + (1 - y_temp)*np.log(1 - args['y_score']))
            args['ll_null'] = np.sum(y_temp*np.log(args['y_prob_nullmod']) +
                                     (1 - y_temp)*np.log(1 - args['y_prob_nullmod']))
                  
    else:
        if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.Series):
            args['y_true'] = y_temp
        args['y_pred'] = model.predict(X)
    
    for met in metrics:
        args['conf_metric'] = met
        temp_args = {k: args[k] for k in All_Metrics[met]['arguments']}
        res[met] = All_Metrics[met]['Function'](**temp_args) 
    
    if print_:
        print(res)
    
    return res


# Currently, this takes just as long as scoreModel and the logic is simpler... keeping for legacy reasons
def scoreModel_legacy(X, Y, model, thres=None, mtype='classification', print_=True, avg='weighted'):
    """Return the legacy score fields with corrected classification counts."""
    if mtype == 'classification':
        # Keep the historical output fields, including four likelihood-based pseudo-R2 values.
        metrics = ['Accuracy', 'F1', 'Sens/Recall', 'Specificity', 'PPV', 'NPV',
                   'Confusion Matrix', 'AUC', 'McFadden R2', 'Tjur R2',
                   'Nagelkerke R2', 'Cox/Snell R2']
        result = scoreModel(X, Y, model, metrics=metrics, thres=thres,
                            mtype=mtype, print_=False, avg=avg)
        result['ppv'] = result.pop('PPV')
        result['npv'] = result.pop('NPV')
        result['Conf_Mat'] = result.pop('Confusion Matrix')
    else:
        truth = np.asarray(Y).ravel()
        predicted = np.asarray(model.predict(X)).ravel()
        centered = truth - truth.mean()
        total = np.sum(centered**2)
        explained = np.sum((predicted - truth.mean())**2)
        # This legacy ratio is SSR/SST. Equality with 1-SSE/SST and squared
        # correlation requires the usual fitted OLS conditions and need not hold out of sample.
        rsquared = explained / total if total else np.nan
        result = {'Bias': bias(truth, predicted), 'RPMSE': rpmse(truth, predicted),
                  'R2': rsquared,
                  'Adj. R2': 1 - (1 - rsquared) * (len(truth) - 1) / (len(truth) - X.shape[1] - 1)}
    if print_:
        print(result)
    return result


def runScorers(X, Y, splits, model, mtype, metrics=None,
               avg='weighted', calculate=None,
               method_on_X=None, mox_args=None, Y_for_test_only=None,
               sample_limit=20):
    """Performs the cross-validation.

    Parameters
    ----------
    X : pandas dataframe
        Feature or design matrix.
    Y : pandas series
        Response or Target variable.
    splits : sklearn.model_selection object
        Contains the split type and information about the split.
    model : sklearn, or similar
        model to fit to the data. Must have fit, predict, and/or predict_proba
        methods, depending on the metrics.
    mtype : str, optional
        The type of model, currently either classification or regressor.
        The default is 'classification'.
    metrics : list, optional
        The metrics to include in the analysis.
        The default is ['Accuracy', 'F1', 'Sens/Recall', \
                        'Specificity', 'ppv', 'npv', 'AUC'].
    avg : str, optional
        for metrics where it applies, such as AUC, how to calculate the
        metric (see sklearn docs for more details, particularly with AUC).
        The default is "weighted".
    calculate : list or str, optional
        What to calculate the metrics on, either in-sample or out-of-sample,
        or both. In-sample measures the metrics on the train set while
        out-of-sample measures on the test set.
        The default is ['Out of Sample'].
    method_on_X : function, optional
        An optional function to pass that performs an operation on X.
        For example, you could pass a function to perform PCA before
        performing the fits. This would enable the function to be applied to
        each level of split. The method used to transform/modify X can be
        of any class, the only requirement is that it has both fit and
        transform methods, and that the arguments for fit method accept both
        an X and Y argument, even if the Y argument does nothing. You may need
        to create a simple wrapper for this, if you have a known method you 
        want to use, but doesn't quite fit what you need. The default is None.
    mox_args : dict, optional
        Any optional arguments that get passed to the constructor of the
        method_on_X. The default is None.
    Y_for_test_only : pandas series or numpy array, optional
        An alternate target variable to test on, mainly for the use case of
        training on one response (perhaps one that is more informative or
        restrictive), but then predicting on a separate test set.
        The default is None.
    sample_limit : int, optional
        The minimum acceptable samples in a split for the response
        variable. Only applies to Y_for_test_only currently.
        The default is 20.

    Returns
    -------
    keys : dict
        A dictionary containing the results.
    """
    if metrics is None:
        metrics = DEFAULT_CLASSIFICATION_METRICS if mtype == 'classification' else DEFAULT_REGRESSION_METRICS
    if calculate is None:
        calculate = ['Out of Sample']
    if mox_args is None:
        mox_args = {}
    keys = {}
 
    for train_idx, test_idx in splits.split(X, Y):
        Xtrain, Xtest = X.iloc[train_idx], X.iloc[test_idx]
        Ytrain, Ytest = Y.iloc[train_idx], Y.iloc[test_idx]
        # Adding a note so I don't forget what this is for... 
        # This is for the situation where you want to train on one definition of truth
        # but test on another, for example, train on the definition where a positive case
        # is either a revoke or a deny, but test on the definition where a positive case
        # is only revoked 
        if Y_for_test_only is not None:
            new_Ytest = Y_for_test_only.iloc[test_idx]
            if mtype == 'classification':
                classes = np.unique(new_Ytest)
                counts = np.array([new_Ytest[new_Ytest.values == i].shape[0] for i in classes])
                if np.all(counts >= sample_limit):
                    Ytest = new_Ytest
                else:
                    warnings.warn("Not all classes have at least %d samples. Using default test set" % (sample_limit))
            else:
                Ytest = new_Ytest
        
        # Method on X must return a dataframe in your method
        # on X function, set Y=None as a default argument
        # and let it pass through if not needed
        if method_on_X is not None:
            method_fit = method_on_X(**mox_args).fit(Xtrain, Ytrain)
            Xtrain = method_fit.transform(Xtrain)
            Xtest = method_fit.transform(Xtest)
        
        mod = copy(model).fit(Xtrain, Ytrain.values.ravel())

        score = None
        if len(keys) == 0:
            for sk in calculate:
                keys[sk] = {}
                if sk == 'Out of Sample':
                    score = scoreModel(Xtest, Ytest, mod, metrics=metrics,
                                       mtype=mtype, print_=False, avg=avg)
                elif sk == 'In Sample':
                    score = scoreModel(Xtrain, Ytrain, mod, metrics=metrics,
                                       mtype=mtype, print_=False, avg=avg)
                    
                for key in score.keys():
                    keys[sk][key] = [score[key]]
        else:
            for sk in calculate:
                if sk == 'Out of Sample':
                    score = scoreModel(Xtest, Ytest, mod, metrics=metrics,
                                       mtype=mtype, print_=False, avg=avg)
                elif sk == 'In Sample':
                    score = scoreModel(Xtrain, Ytrain, mod, metrics=metrics,
                                       mtype=mtype, print_=False, avg=avg)
                    
                for key in score.keys():
                    keys[sk][key].append(score[key])
    return keys


def printScores(scores):
    """
    A helper function to print the outputs of the crossVal function.

    Currently, it does not print confusion matrices.    

    Parameters
    ----------
    scores : dict
        The dictionary of scores.

    Returns
    -------
    None.

    """
    longest_key = 0
    for sample in scores.keys():
        for score_type in scores[sample].keys():
            if len(score_type) > longest_key:
                longest_key = len(score_type)
    for sample in scores.keys():
        print(sample, ": ", sep='')
        for score_type in scores[sample].keys():
            values = scores[sample][score_type]
            if score_type == 'Conf_Mat':
                continue
            else:
                spacer = longest_key - len(score_type)
                print('\t', score_type, ' '*spacer, ': ', '\u03BC: ', '{:.4f}'.format(np.mean(values)), ', ',
                      '\u03C3: ', '{:.4f}'.format(np.std(values, ddof=1)), ', ',
                      '2\u03C3 Interval (', '{:.4f}'.format(np.mean(values)-(2*np.std(values, ddof=1))), ', ',
                      '{:.4f}'.format(np.mean(values)+(2*np.std(values, ddof=1))), ")",
                      sep='')
        print('')
    return None

        
def crossVal(X, Y, cv_iterations, model, method='k-fold',
             mtype='classification', stratified=True, print_=True,
             random_state=None, method_on_X=None, mox_args=None,
             avg='weighted', shuffle=True, test_size=0.1,
             calculate=None, metrics=None,
             Y_for_test_only=None, sample_limit=20):
    # TODO: add parallel support
    """A custom cross-validation strategy. While sklearn has a lot of beefed up
    functionality regarding cross-validation strategies, I like this one
    because it puts most of the strategies I like to use in one place. Also,
    it allows for you to output and calculate several cross-validation metrics
    at once.
    
    One thing that still needs to be added here is parallelization, since each
    of the folds can be calculated simultaneously.

    Parameters
    ----------
    X : pandas dataframe
        Feature or design marix.
    Y : pandas series
        Response or Target variable.
    cv_iterations : int
        The number of cross validation iterations. For k-fold, it is the
        number of folds, and for shuffle, it is the number of iterations.
    model : sklearn, or similar
        model to fit to the data. Must have fit, predict, and/or predict_proba
        methods, depending on the metrics.
    method : str, optional
        The method of cross-validation, either k-fold or shuffle.
        The default is 'k-fold'.
    mtype : str, optional
        The type of model, currently either classification or regressor.
        The default is 'classification'.
    stratified : bool, optional
        Use a stratified sample instead of a random sample.
        The default is True.
    print_ : bool, optional
        Print the results along the way. The default is True.
    random_state : int, optional
        Set the random seed for reproducibility. The default is None.
    method_on_X : function, optional
        An optional function to pass that performs an operation on X.
        For example, you could pass a function to perform PCA before
        performing the fits. This would enable the function to be applied to
        each level of split. The method used to transform/modify X can be
        of any class, the only requirement is that it has both fit and
        transform methods, and that the arguments for fit method accept both
        an X and Y argument, even if the Y argument does nothing. You may need
        to create a simple wrapper for this, if you have a known method you 
        want to use, but doesn't quite fit what you need. The default is None.
    mox_args : dict, optional
        Any optional arguments that get passed to the constructor of the
        method_on_X. The default is {}.
    avg : str, optional
        for metrics where it applies, such as AUC, how to calculate the
        metric (see sklearn docs for more details, particularily with AUC).
        The default is "weighted".
    shuffle : bool, optional
        Shuffle the dataset in the k-fold operations. The default is True.
    test_size : float, optional
        A percent representing the size of the test set. The default is 0.1.
    calculate : list or str, optional
        What to calculate the metrics on, either in-sample or out-of-sample,
        or both. In-sample measures the metrics on the train set while
        out-of-sample measures on the test set.
        The default is ['Out of Sample'].
    metrics : list, optional
        The metrics to include in the analysis.
        The default is ['Accuracy', 'F1', 'Sens/Recall', 'Specificity', 'ppv', 'npv', 'AUC'].
    Y_for_test_only : pandas series or numpy array, optional
        An alternate target variable to test on, mainly for the use case of
        training on one response (perhaps one that is more informative or
        restrictive), but then predicting on a separate test set.
        The default is None.
    sample_limit : int, optional
        The minimum acceptable samples in a split for the response
        variable. Only applies to Y_for_test_only currently.
        The default is 20.

    Returns
    -------
    scores : dict
        Dictionary of performance metrics.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> from matplotlib import pyplot as plt
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from timeit import default_timer as dt
    >>> 
    >>> from tsdst.modeling import crossVal
    >>> 
    >>> X, y = make_classification(n_samples=10000, n_features=50, flip_y=0.2, random_state=42)
    >>> X = pd.DataFrame(X, columns=['V' + str(i) for i in range(X.shape[1])])
    >>> y = pd.DataFrame(y.reshape(-1, 1), columns=['y'])
    >>> 
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    >>> 
    >>> sk_mod = LogisticRegression()
    >>> 
    >>> results = crossVal(X_train, y_train, 5, sk_mod)
    Out of Sample:
        Accuracy   : μ: 0.8155, σ: 0.0052, 2σ Interval (0.8051, 0.8259)
        F1         : μ: 0.8151, σ: 0.0048, 2σ Interval (0.8056, 0.8246)
        Sens/Recall: μ: 0.8184, σ: 0.0105, 2σ Interval (0.7973, 0.8394)
        Specificity: μ: 0.8129, σ: 0.0063, 2σ Interval (0.8002, 0.8256)
        ppv        : μ: 0.8121, σ: 0.0095, 2σ Interval (0.7930, 0.8312)
        npv        : μ: 0.8189, σ: 0.0138, 2σ Interval (0.7914, 0.8464)
        AUC        : μ: 0.8637, σ: 0.0066, 2σ Interval (0.8505, 0.8769)
    """
    if mox_args is None:
        mox_args = {}
    if metrics is None:
        metrics = DEFAULT_CLASSIFICATION_METRICS if mtype == 'classification' else DEFAULT_REGRESSION_METRICS
    if calculate is None:
        calculate = ['Out of Sample']

    splits = None
    scores = None
    if mtype != 'classification':
        stratified = False
        
    if method == 'k-fold':
        if stratified:
            splits = StratifiedKFold(cv_iterations, shuffle=shuffle,
                                     random_state=random_state)
        else:
            splits = KFold(cv_iterations, shuffle=shuffle,
                           random_state=random_state)
    else:
        if stratified:
            splits = StratifiedShuffleSplit(cv_iterations, test_size=test_size,
                                            random_state=random_state)
        else:
            splits = ShuffleSplit(cv_iterations, test_size=test_size,
                                  random_state=random_state)

    scores = runScorers(X, Y, splits, model, mtype, metrics=metrics, avg=avg,
                        calculate=calculate, method_on_X=method_on_X,
                        mox_args=mox_args, Y_for_test_only=Y_for_test_only,
                        sample_limit=sample_limit)

    if print_:
        printScores(scores)
            
    return scores


# labels is a dictionary identifying what the label values should be,
# where the dictionary key is the new label, and the dictionary value
# is the current class value
def prettyConfMat(Ytrue, Ypred, print_=True, margins=True, labels=None):
    r"""Return and optionally print a labeled confusion matrix with margins.

    Parameters
    ----------
    Ytrue : array-like
        Actual class labels.
    Ypred : array-like
        Predicted class labels, in the same observation order as Ytrue.
    print_ : bool, optional
        Print the result table. The default is True.
    margins : bool, optional
        Include row totals, column totals, and a grand total. The default is True.
    labels : dict, optional
        Map displayed names to observed class values. Dictionary insertion order
        controls both axes. If None, sorted observed class values are used.

    Returns
    -------
    confmat : pandas.DataFrame
        Actual classes are rows and predicted classes are columns. Each axis
        has a corresponding Actual or Predicted group label. Optional margins
        use the label Total.

    Notes
    -----
    For binary class values ordered [0, 1], the cells are [[TN, FP], [FN, TP]].
    Custom display names and a different requested class order change the
    labels and ordering together; they do not transpose the matrix.
    """
    truth, prediction = np.asarray(Ytrue).ravel(), np.asarray(Ypred).ravel()
    if labels is None:
        values = list(np.unique(np.r_[truth, prediction]))
        names = values
    else:
        names, values = list(labels), list(labels.values())
        if not set(np.unique(np.r_[truth, prediction])).issubset(values):
            raise ValueError('Labels must include every observed class')
    matrix = confusion_matrix(truth, prediction, labels=values)
    if margins:
        matrix = np.column_stack([matrix, matrix.sum(axis=1)])
        matrix = np.vstack([matrix, matrix.sum(axis=0)])
        names = names + ['Total']
    result = pd.DataFrame(matrix,
              index=pd.MultiIndex.from_product([['Actual'], names]),
              columns=pd.MultiIndex.from_product([['Predicted'], names]))
    if print_:
        print(result)
    return result

  
def GenerateRegCov(X, Y, fit_mod, logit=True, low_memory=False, lamda=0,
                   *, covariance='sampling', penalize_intercept=False):
    r"""Calculate coefficient covariance for linear or binary logistic regression.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training feature matrix. A separately fitted intercept is added to the
        internal design matrix.
    Y : array-like
        Training response values.
    fit_mod : fitted estimator
        A linear or binary logistic model with ``coef_`` and ``intercept_``. Models
        with fit_intercept=False use their supplied design without another
        intercept column.
    logit : bool, optional
        Use logistic information when True, or linear information when False.
        The default is True.
    low_memory : bool, optional
        Retained argument. Both paths use weighted matrix products and avoid
        allocating an n by n diagonal weight matrix. The default is False.
    lamda : float, optional
        Nonnegative effective L2 Hessian penalty, matching the fitted model.
        The default is zero. This is a ridge/Gaussian-prior penalty, not an L1
        or Laplace-prior penalty.
    covariance : {'sampling', 'curvature'}, optional
        Sampling uses A^-1 S A^-1, where S is the unpenalized information and
        A=S+lamda*P. Curvature uses A^-1. The default is 'sampling'.
    penalize_intercept : bool, optional
        Include a separately fitted intercept in P when True. The default is
        False. Match this setting to the actual fit.

    Returns
    -------
    cov : numpy.ndarray
        Coefficient covariance. Its diagonal contains coefficient variances.
    base : numpy.ndarray or None
        The selected unscaled linear covariance matrix. For ordinary least
        squares this is (X.T @ X)^-1 for the internal design. Logistic returns
        None in this slot.
    sigma2 : float or None
        OLS noise-variance estimate RSS/(n-p), where p includes a fitted
        intercept. Logistic regression returns None and uses unit dispersion.
    betas : numpy.ndarray
        Coefficients as a column vector, with a separately fitted intercept first.

    Notes
    -----
    Logistic information uses Bernoulli variance p*(1-p). Linear covariance
    multiplies its base matrix by sigma2. The linear noise estimate comes from
    OLS so that ridge shrinkage bias does not inflate residual variance.
    RSS/n is the corresponding maximum-likelihood variance estimate.

    With no penalty, both covariance choices reduce to inverse information.
    With a penalty, sampling covariance describes variation around the
    shrunken estimator's sampling mean. Curvature describes a local Gaussian
    approximation with the matching Gaussian-prior scale convention. These
    are model-based calculations; a misspecification-robust covariance would
    require a different variance estimate. Standard errors are sqrt(diag(cov)).
    """
    original_X = X
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float).ravel()
    if X.ndim != 2 or len(Y) != len(X) or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError('Supply finite features and one finite response per observation')
    if covariance not in ('sampling', 'curvature'):
        raise ValueError('covariance must be sampling or curvature')
    if not np.isfinite(lamda) or lamda < 0:
        raise ValueError('lamda must be finite and nonnegative')
    intercept = bool(getattr(fit_mod, 'fit_intercept', True))
    design = np.column_stack([np.ones(len(X)), X]) if intercept else X
    beta = np.asarray(fit_mod.coef_).ravel()
    if intercept:
        beta = np.r_[np.asarray(fit_mod.intercept_).ravel(), beta]
    if beta.size != design.shape[1]:
        raise ValueError('The fitted coefficient count does not match the design')
    penalty = np.eye(design.shape[1])
    if intercept and not penalize_intercept:
        penalty[0, 0] = 0
    if logit:
        probabilities = np.asarray(fit_mod.predict_proba(original_X))
        if probabilities.shape != (len(X), 2):
            raise ValueError('Logistic covariance requires a binary probability model')
        # For binary probabilities the product is p*(1-p), the Bernoulli variance function.
        weights = np.prod(probabilities, axis=1)
        information = design.T @ (weights[:, None]*design)
        inverse = np.linalg.inv(information+lamda*penalty)
        base = inverse @ information @ inverse if covariance == 'sampling' else inverse
        return base, None, None, beta[:, None]
    degrees = len(X)-design.shape[1]
    if degrees <= 0:
        raise ValueError('Noise-variance estimation needs positive residual degrees of freedom')
    information = design.T @ design
    # Estimate noise from OLS so shrinkage bias does not inflate residual variance.
    ordinary = np.linalg.solve(information, design.T @ Y)
    # RSS/(n-p) estimates noise variance; p includes an estimated intercept.
    variance = np.sum((Y-design @ ordinary)**2)/degrees
    inverse = np.linalg.inv(information+lamda*penalty)
    # The sampling choice is the ridge sandwich covariance before multiplying by sigma2.
    base = inverse @ information @ inverse if covariance == 'sampling' else inverse
    return variance*base, base, variance, beta[:, None]
  

def RegressionSE(X, Y, fit_mod, logit=True, low_memory=False, lamda=0,
                 *, covariance='sampling', penalize_intercept=False):
    r"""Calculate regression coefficient standard errors and covariance quantities.

    Parameters
    ----------
    X : array-like
        Training feature matrix.
    Y : array-like
        Training response values.
    fit_mod : fitted estimator
        Linear or binary logistic model.
    logit : bool, optional
        Calculate logistic standard errors when True. The default is True.
    low_memory : bool, optional
        Retained argument passed to GenerateRegCov. The default is False.
    lamda : float, optional
        Effective L2 penalty matching the fit. The default is zero.
    covariance : {'sampling', 'curvature'}, optional
        Covariance interpretation passed to GenerateRegCov. Sampling is the default.
    penalize_intercept : bool, optional
        Match the fit's intercept penalty. The default is False.

    Returns
    -------
    se : numpy.ndarray
        Standard errors, sqrt(diag(cov)).
    cov : numpy.ndarray
        Selected coefficient covariance matrix.
    base : numpy.ndarray or None
        Unscaled linear covariance matrix; None for logistic regression.
    sigma2 : float or None
        OLS error-variance estimate; None for logistic regression.
    betas : numpy.ndarray
        Coefficient column vector, including a separately fitted intercept first.

    See Also
    --------
    GenerateRegCov : Covariance definitions and assumptions.
    """
    cov, base, sigma2, betas = GenerateRegCov(X, Y, fit_mod, logit, low_memory, lamda,
        covariance=covariance, penalize_intercept=penalize_intercept)
    return np.sqrt(np.diag(cov)), cov, base, sigma2, betas
  

def predict(X_test, X_train, Y_train, fit_mod, logit=True, interval='prediction',
            low_memory=False, lamda=0, level=0.95, logit_transform=False,
            logit_se_method='wald', *, covariance='sampling', penalize_intercept=False):
    r"""Calculate regression predictions with confidence or prediction intervals.

    Parameters
    ----------
    X_test : array-like
        Feature rows where predictions are required.
    X_train : array-like
        Training feature matrix used to estimate coefficient covariance.
    Y_train : array-like
        Training response values.
    fit_mod : fitted estimator
        Linear or binary logistic model.
    logit : bool, optional
        Use logistic calculations when True. The default is True.
    interval : {'prediction', 'confidence'}, optional
        Linear prediction intervals include error variance for a new response;
        confidence intervals describe the fitted mean. The default is 'prediction'.
        For logistic regression, 'prediction' emits a warning and returns a
        confidence interval for the response probability.
    low_memory : bool, optional
        Retained argument passed to GenerateRegCov. The default is False.
    lamda : float, optional
        Effective L2 penalty matching the fit. The default is zero.
    level : float, optional
        Interval coverage level between zero and one. The default is 0.95.
    logit_transform : bool, optional
        Transform logistic predictions and both endpoints to probabilities.
        Otherwise return log-odds. The default is False.
    logit_se_method : {'wald'}, optional
        Logistic interval method. The default is 'wald'. A delta-method
        interval is not implemented.
    covariance : {'sampling', 'curvature'}, optional
        Covariance interpretation. The default is 'sampling'.
    penalize_intercept : bool, optional
        Match the fitted model's intercept penalty. The default is False.

    Returns
    -------
    result : pandas.DataFrame
        Columns yhat, lower_int, and upper_int contain predictions and bounds.

    Notes
    -----
    OLS intervals use Student t critical values with n-p residual degrees of
    freedom. Logistic Wald intervals use normal critical values on the linear
    predictor. Transforming both endpoints through the inverse logit gives
    probability confidence bounds.

    Intervals for penalized fits are approximations. Sampling covariance
    describes variation of the shrunken estimator and does not remove its bias
    for unpenalized population parameters. Curvature intervals use a local
    Gaussian-prior interpretation with an estimated noise scale when needed.
    """
    if interval not in ('prediction', 'confidence') or not 0 < level < 1:
        raise ValueError('Supply a valid interval type and a level between zero and one')
    cov, _, variance, beta = GenerateRegCov(X_train, Y_train, fit_mod, logit, low_memory, lamda,
        covariance=covariance, penalize_intercept=penalize_intercept)
    X_test = np.asarray(X_test)
    design = np.column_stack([np.ones(len(X_test)), X_test]) if beta.size == X_test.shape[1]+1 else X_test
    estimate = (design @ beta).ravel()
    prediction_variance = np.einsum('ij,jk,ik->i', design, cov, design)
    if logit:
        if logit_se_method != 'wald':
            raise NotImplementedError('Only the Wald method is available')
        if interval == 'prediction':
            warnings.warn('Using a confidence interval for a binary response', UserWarning)
        critical = normal.ppf((1+level)/2)
    else:
        if interval == 'prediction':
            prediction_variance += variance
        critical = t.ppf((1+level)/2, df=len(X_train)-beta.size)
    margin = critical*np.sqrt(np.maximum(prediction_variance, 0))
    result = np.column_stack([estimate, estimate-margin, estimate+margin])
    if logit and logit_transform:
        # The inverse logit is monotone, so transform the Wald endpoints with the prediction.
        result = sigmoid(result)
    return pd.DataFrame(result, columns=['yhat', 'lower_int', 'upper_int'])


def corrmat_validity_check(corr_mat, corr_tol=1e-8):
    """Check for perfect correlation and/or a singular covariance matrix. Either
    of these conditions could indicate a matrix unsuitable for Linear
    Regression.

    Parameters
    ----------
    corr_mat : numpy array
        The correlation matrix.
    corr_tol : float
        The floating point error allowed to check for perfect correlation.

    Returns
    -------
    has_perfect_corr : numpy array
        An array of boolean values indicating perfect correlation.
    """
    det = np.linalg.det(corr_mat)
    ncols = corr_mat.shape[0]
    corr_mat_flat = corr_mat.reshape(-1, )
    has_perfect_corr = (np.abs(corr_mat_flat) >= 1-corr_tol).reshape(ncols, ncols)
    np.fill_diagonal(has_perfect_corr, 0)
    return has_perfect_corr, det


def vif(data, root=False, corr_tol=1e-8, sing_tol=1e-15):
    """
    Calculate Variance Inflation Factors. The sqrt of vif indicates how many
    times larger the standard error is than it would be if that variable had
    no correlation with the other variables 
    
    Assumes rows are observations and columns are variables.
    
    Expects pandas.DataFrame or numpy.array

    Parameters
    ----------
    data : pandas dataframe or numpy array
        The feture or design matrix.
    root : bool, optional
        Return the sqrt of the variance inflation factors. The default is False.

    Returns
    -------
    Either pandas Dataframe or numpy array
        The VIFs.

    """
    corr = np.corrcoef(data, rowvar=False)
    perfect_corr, det = corrmat_validity_check(corr, corr_tol=corr_tol)
    if perfect_corr.sum() > 0:
        warnings.warn("There are perfectly correlated variables. VIF "
                      "may not be reliable.")
    if det < sing_tol:
        warnings.warn("The corr_mat determinant is essentially zero. VIF "
                      "may not be reliable.")
    
    vifs = np.diag(np.linalg.inv(corr))
    
    if isinstance(data, pd.DataFrame):
        vifs = pd.DataFrame(vifs,
                            index=data.columns,
                            columns=['Variance Inflation Factors'])
    if np.any(vifs < 1):
        warnings.warn("Some vifs are less than one. Model is not correctly "
                      "specified or is not a suitable linear model")
                      
    if root:
        return np.sqrt(vifs)
    else:
        return vifs


def beta_trans(coef, type_="percent"):
    """
    Transform the model coefficients in a logistic regression model.

    Parameters
    ----------
    coef : numpy array
        The model coefficients.
    type_ : str, optional
        The type of transform, either percent, odds, or decimal.
        
        Decimal and percent are the same, except that percent = decimal*100
        
        The default is "percent".

    Returns
    -------
    numpy array
        Transformed Coefficients.

    """
    val = np.exp(coef)
    if type_ == "odds":
        return val
    elif type_ == "decimal":
        return val - 1
    elif type_ == "percent":
        return 100.0*(val - 1)
    else:
        return coef
    

def getPriors(data, Ycol):
    """Calculate the class distributions across each column of the dataset.
    
    For classification problems, but could easily be used with regression
    problems if an indicator variable was made that indicated some split in the
    data.

    Parameters
    ----------
    data : pandas dataframe
        The feature or design matrix, along with the response variable.
    Ycol : str
        The column representing the target or response variable.

    Returns
    -------
    alltab : pandas dataframe
        A dataframe containing all of the results.
    """
    tab = None
    alltab = None
    #numeric = False
    sub = data
    for k, i in enumerate([column for column in sub.columns if column != Ycol]):
        if len(sub.loc[:, i].unique()) > 7:
            arr = np.array([[sub[sub[Ycol] == 0][i].mean(),
                             sub[sub[Ycol] == 1][i].mean(),
                             sub.loc[:,i].mean()],
                            [sub[sub[Ycol] == 0][i].std(),
                             sub[sub[Ycol] == 1][i].std(),
                             sub.loc[:,i].std()]])
            tab = pd.DataFrame(arr, index=["Mean", "Std. Dev"], columns=[0, 1, u'All'])
            tab.columns.name = Ycol
            tab.index.name = i
            tab.index = pd.MultiIndex.from_arrays([[tab.index.name]*len(tab.index), tab.index], names=(None, None))
        else:
            tab = pd.crosstab(sub.loc[:, i], sub.loc[:, Ycol], margins=True, normalize=True, dropna=False)
            tab.index = pd.MultiIndex.from_arrays([[tab.index.name]*len(tab.index), tab.index], names=(None, None))
        if k == 0:
            alltab = tab
        else:
            alltab = pd.concat((alltab, tab))
    return alltab
    

class EstimatorSelectionHelper(object):
    """A wrapper for easily grid-searching multiple models in the
    same run.

    This is adapted from David Batista, so credit goes to him.

    see here: https://davidsbatista.net/blog/2018/02/23/model_optimization

    I added some functionality that will extend to the sklearn-deap package
    as well. see here: https://github.com/rsteca/sklearn-deap
    """

    def __init__(self, models, params, searchCV, searchCVparams):
        """
        Constructor for the EstimatorSelectionHelper class.

        Parameters
        ----------
        models : dict
            A dictionary of model types.
        params : dict
            A dictionary of parameters for each model.
        searchCV : sklearn object
            The type of search being performed.
        searchCVparams : dict
            A dictionary of optional arguments to send to searchCV.

        Raises
        ------
        ValueError
            Raised when a model is missing parameters, or there is a mismatch.

        Returns
        -------
        None.

        """
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.searchCV = searchCV
        self.searchCVparams = searchCVparams
        self.compute_times = {}
        self.gs_name = ""
        
    def fit(self, X, y):
        """
        perform the search.

        Parameters
        ----------
        X : pandas dataframe or numpy array
            The feature or design matrix.
        y : pandas series or numpy array
            The response or target variable.

        Returns
        -------
        self
            Adds fitted parameters/models to the object.

        """
        for key in self.keys:
            t0 = dt()
            print("Running GridSearch for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = self.searchCV(model, params, **self.searchCVparams)
            gs.fit(X, y)
            self.grid_searches[key] = gs
            self.compute_times[key] = dt() - t0
            self.gs_name = gs.__class__.__name__
        return self
    
    def _score_summary_ea(self, sort_by):
        """Create a summary of the evolutionary algorithm results.

        Parameters
        ----------
        sort_by : list or list-like
            Columns to sort by, for example, 'mean_score'.

        Returns
        -------
        pandas dataframe
            A dataframe containing the results.
        """
        rows = []
        for k in self.grid_searches:
            params = self.grid_searches[k].cv_results_['params']
            min_test_score = self.grid_searches[k].cv_results_["min_test_score"]
            max_test_score = self.grid_searches[k].cv_results_["max_test_score"]
            mean_test_score = self.grid_searches[k].cv_results_["mean_test_score"]
            std_test_score = self.grid_searches[k].cv_results_["std_test_score"]

            for i in range(len(min_test_score)):
                d = {
                    'estimator': k,
                    'min_score': min_test_score[i],
                    'max_score': max_test_score[i],
                    'mean_score': mean_test_score[i],
                    'std_score': std_test_score[i],
                    'all_params': params,
                }
                new_row = pd.Series({**params[i], **d})
                rows.append((new_row))
        
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
        
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]
    
    def _score_summary_skgs(self, sort_by):
        """Create a summary of the Grid Search results.

        Parameters
        ----------
        sort_by : list or list-like
            Columns to sort by, for example, 'mean_score'.

        Returns
        -------
        pandas dataframe
            A dataframe containing the results.
        """
        def row(key, scores, params, score_type):
            d = {
                'estimator': key,
                'min_score': min(scores),
                'max_score': max(scores),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'score_type': score_type
            }
            return pd.Series({**params, **d})
        
        rows = []
        for k in self.grid_searches:
            params = self.grid_searches[k].cv_results_['params']
            for score_method_key in self.searchCVparams['scoring'].keys():
                scores = []
                for i in range(self.grid_searches[k].cv):
                    key = "split{}_test_{}".format(i, score_method_key)
                    r = self.grid_searches[k].cv_results_[key]
                    scores.append(r.reshape(len(params), 1))
            
                all_scores = np.hstack(scores)
                for p, s in zip(params, all_scores):
                    rows.append((row(k, s ,p, score_method_key)))
        
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
        
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]
    
    def score_summary(self, sort_by='mean_score'):
        """Create a summary of the scores and results.

        Parameters
        ----------
        sort_by : list or list-like
            Columns to sort by, for example, 'mean_score'.

        Returns
        -------
        scores : pandas dataframe
            A dataframe containing the results.
        """
        scores = None
        if self.gs_name == 'EvolutionaryAlgorithmSearchCV':
            scores = self._score_summary_ea(sort_by=sort_by)
        else:
            scores = self._score_summary_skgs(sort_by=sort_by)
        return scores


def BPCA_initmodel(y, q):
    """Initialize the bPCA model.

    Parameters
    ----------
    y : numpy array
        The data to be nan-filled.
    q : int
        The number of dimensions to consider in the PCA.

    Returns
    -------
    M : dict
        A dictionary of the initialized values.
    """
    M = {}
    M['N'] = y.shape[0]
    M['d'] = y.shape[1]
    M['q'] = q
    M['yest'] = y.copy()
    M['missidx'] = []
    M['nomissidx'] = []
    M['gnomiss'] = []
    M['gmiss'] = []
    
    for i in range(y.shape[0]):
        M['missidx'].append(np.where(np.isnan(y[i, :]))[0])
        M['nomissidx'].append(np.where(~np.isnan(y[i, :]))[0])
        if M['missidx'][i].shape[0] == 0:
            M['gnomiss'].append(i)
        else:
            M['gmiss'].append(i)
            M['yest'][i, M['missidx'][i]] = 0
    
    # ynomiss = y[M['gnomiss'], :]
    covy = np.cov(M['yest'], rowvar=False)
    U, S, _ = np.linalg.svd(covy, full_matrices=True)
    U = U[:, :q]
    S = S[:q]
    #V = V.T[:, :q]

    #M['mu'] = np.nansum(y, axis=0)/np.array([np.sum(~np.isnan(y[:, col])) for col in range(y.shape[1])])
    M['mu'] = np.nanmean(y, axis=0)
    
    # W are the Principal axis vectors
    M['W'] = U * np.sqrt(S);
    M['tau'] = 1/(np.sum(np.diag(covy)) - np.sum(np.diag(S)))
    taumax = 1e10
    taumin = 1e-10
    M['tau'] = max(min(M['tau'], taumax), taumin)
    
    M['galpha0'] = 1e-10
    M['balpha0'] = 1
    M['alpha'] = (2*M['galpha0'] + M['d'])/(M['tau']*np.diag(M['W'].T.dot(M['W'])) + 2*M['galpha0']/M['balpha0'])
    
    M['gmu0'] = 0.001
    
    M['btau0'] = 1
    M['gtau0'] = 1e-10
    M['SigW'] = np.eye(q)
    
    return M


def BPCA_dostep(M, y):
    """The workhorse of the bPCA algorithm, the EM-like (or Variational Bayes)
    step.

    Parameters
    ----------
    M : dict
        A dictionary containing preliminary results for the algorithm.
    y : numpy array
        The original data.

    Returns
    -------
    M : dict
        The updated results.
    """
    N = M['N']
    d = M['d']
    
    Rx = np.eye(M['q']) + M['tau']*M['W'].T.dot(M['W']) + M['SigW']
    Rxinv_o = np.linalg.inv(Rx)

    idx = M['gnomiss']
    n = len(idx)
    
    dy = y[idx, :] - np.tile(M['mu'], (n, 1))
    
    # x is the factor scores
    x = M['tau'] * Rxinv_o.dot(M['W'].T).dot(dy.T)

    Tt = dy.T.dot(x.T)
    trS = np.sum(dy * dy)
    for i in M['gmiss']:
        dyo = y[i, M['nomissidx'][i]] - M['mu'][M['nomissidx'][i]]

        Wm = M['W'][M['missidx'][i], :]
        Wo = M['W'][M['nomissidx'][i], :]
        Rxinv = np.linalg.inv(Rx - M['tau']*Wm.T.dot(Wm))
        ex = M['tau']*Wo.T.dot(dyo.T)
        x = Rxinv.dot(ex)
        
        # estimating data missing values
        dym = Wm.dot(x)
        
        dy = y[i, :].copy()
        dy[M['nomissidx'][i]] = dyo.T
        dy[M['missidx'][i]] = dym.T
        M['yest'][i, :] = dy + M['mu']

        Tt = Tt + reshape_to_vector(dy).dot(reshape_to_vector(x).T)
        Tt[M['missidx'][i], :] = Tt[M['missidx'][i], :] + Wm.dot(Rxinv)
        trS = trS + dy.dot(dy.T) + len(M['missidx'][i])/M['tau'] + np.trace(Wm.dot(Rxinv).dot(Wm.T))

    Tt = Tt/N
    trS = trS/N

    Dw = Rxinv_o + M['tau']*Tt.T.dot(M['W']).dot(Rxinv_o) + np.diag(M['alpha'])/N
    Dwinv = np.linalg.inv(Dw)
    M['W'] = Tt.dot(Dwinv)
    
    M['tau'] = (d + 2*M['gtau0']/N)/(trS - np.trace(Tt.T.dot(M['W'])) + (M['mu'].dot(M['mu'].T)*M['gmu0'] + 2*M['gtau0']/M['btau0'])/N)
    M['SigW'] = Dwinv*(d/N)
    
    # hyperparameter alpha update
    M['alpha'] = (2*M['galpha0'] + d)/(M['tau']*np.diag(M['W'].T.dot(M['W'])) + np.diag(M['SigW']) + 2*M['galpha0']/M['balpha0'])

    return M
        


# assumes no rows of all NaNs
def bPCA(data, k=None, maxepoch=200, stepsize=10, dtau_tol=1e-8):
    r"""An algorithm to compute missing values using Bayesian PCA. Translated from
    MATLAB code by Shigeyuki OBA, 2002 May. 5th.

    Parameters
    ----------
    data : numpy array
        The data with missing values to be estimated.
    k : int, optional
        The number of components to consider. Must be less than the number of
        columns. If None, use num_cols - 1. The default is None.
    maxepoch : int, optional
        Number of iterations. The default is 200.
    stepsize : int, optional
        The number of iterations to compute before printing results.
        The default is 10.
    dtau_tol : float, optional
        The precision tolerance. Breaks if precision is lower than the
        tolerance. The default is 1e-8.

    Returns
    -------
    M : dict
        A dictionary of the results, containing:

            - yest : The original data matrix with missing values imputed
            - mu : The estimated mean row vector
            - W : The estimated principal axis matrix
            - tau : The estimated precision (inverse variance) of the residual error

    Examples
    --------
    >>> import numpy as np
    >>> from tsdst.modeling import bPCA
    >>> X = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, np.nan, 9],
    ...               [10, 11, 12]])
    >>> M = bPCA(X, maxepoch=5, stepsize=10)  # doctest: +ELLIPSIS
    epoch=0, dtau=...
    >>> bool(np.isfinite(M['yest']).all())
    True
    >>> bool(np.allclose(M['yest'][~np.isnan(X)], X[~np.isnan(X)]))
    True
    """
    N, d = data.shape
    if k is None:
        k = d - 1
    
    M = BPCA_initmodel(data, k)
    tauold = 1000
    
    for epoch in range(maxepoch):
        M = BPCA_dostep(M, data)
        if epoch % stepsize == 0:
            tau = M['tau'];
            dtau = np.abs(np.log10(tau)-np.log10(tauold));
            print('epoch=%d, dtau=%g' % (epoch, dtau));
            if dtau < dtau_tol:
                break
            tauold = tau    
    return M
        
