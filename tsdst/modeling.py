import numpy as np
import pandas as pd
import warnings

from copy import deepcopy as copy
from sklearn.metrics import (f1_score, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.model_selection import (StratifiedKFold, KFold,
                                     ShuffleSplit, StratifiedShuffleSplit)
from timeit import default_timer as dt

from .metrics import (cox_snell_r2, nagelkerke_r2, tjur_r2, mcfadden_r2,
                      conf_mat_metrics, bias, rpmse, r2, adj_r2, top_20p,
                      number_of_nonzero_coef)
from .utils import reshape_to_vect


def scoreModel(X, Y, model, metrics=['Accuracy',
                                    'F1',
                                    'Sens/Recall',
                                    'Specificity',
                                    'ppv',
                                    'npv',
                                    'AUC'
                                    ],
               thres=None, mtype='classification',
               print_=True, avg="weighted"):
    '''
    Score a model using the given metrics.

    Parameters
    ----------
    X : pandas dataframe
        Feature or design marix.
    Y : pandas series
        Response or Target variable.
    model : sklearn, or similar
        model to fit to the data. Must have fit, predict, and/or predict_proba
        methods, depending on the metrics.
    metrics : list, optional
        The metrics to include in the analysis.
        The default is ['Accuracy', 'F1', 'Sens/Recall',
                        'Specificity', 'ppv', 'npv', 'AUC'].
    thres : float, optional
        The threshold to use for classification metrics where it might apply.
        If threshold is None, the default predict mehtod is used (which
        for most models uses a boundary at or equivalent to 0.5 probability).
        The default is None.
    mtype : str, optional
        The type of model, currently either classification or regressor.
        The default is 'classification'.
    print_ : bool, optional
        Print progress as applicable. The default is True.
    avg : str, optional
        for metrics where it applies, such as AUC, how to calculate the
        metric (see sklearn docs for more details, particularily with AUC).
        The default is "weighted".

    Returns
    -------
    res : dict
        A dictionary containing the results for each metric.

    '''
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
            'ppv': {
                    'Function': conf_mat_metrics,
                    'arguments': ['y_true', 'y_pred', 'conf_metric']
                   },
            'npv': {
                    'Function': conf_mat_metrics,
                    'arguments': ['y_true', 'y_pred', 'conf_metric']
                   },
            'Conf_Mat': {
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
    
    y_temp = Y.values.ravel()
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
            args['ll_null'] = np.sum(y_temp*np.log(args['y_prob_nullmod']) + (1 - y_temp)*np.log(1 - args['y_prob_nullmod']))
                  
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
def scoreModel_legacy(X, Y, model, thres=None, mtype='classification', print_=True, avg="weighted"):
    res = None
    if mtype == "classification":
        nobs = Y.shape[0]
        Yprob = model.predict_proba(X)[:, 1]
        Yprob_nullmod = np.repeat(np.mean(Y.values), nobs)
        
        if thres is None:
            Ypred = model.predict(X)
        else:
            Ypred = np.sum([Yprob >= thres], axis=0)
        
        accs = accuracy_score(Y, Ypred)
        f1s = f1_score(Y, Ypred)
        aucs = roc_auc_score(Y, Yprob, average=avg)
        
        ll_est = np.sum(Y.values*np.log(Yprob) + (1 - Y.values)*np.log(1 - Yprob))
        ll_null = np.sum(Y.values*np.log(Yprob_nullmod) + (1 - Y.values)*np.log(1 - Yprob_nullmod))
        
        # Cox Snell R2
        ratio = 2/nobs
        cs_r2 = 1 - np.exp(ratio*(ll_null - ll_est))
        r2_max = 1 - np.exp(ratio*ll_null)

        # Correction to CS R2 to bound between 0, 1
        n_r2 = cs_r2/r2_max
        
        # When the saturated model is not available (common), then
        # Mcfaddens R2 is equivalent to the likelihood ratio r2
        m_r2 = 1 - (ll_est/ll_null)
        
        #Tjur R2
        y_mu1 = Yprob[Y == 1].mean()
        y_mu0 = Yprob[Y == 0].mean()
        t_r2 = np.abs(y_mu1 - y_mu0)

        # Assumes 1 as the positive class
        confMat = confusion_matrix(Y, Ypred)
        tn = confMat[0, 0]
        tp = confMat[1, 1]
        fn = confMat[0, 1]
        fp = confMat[1, 0]
        tnfp = tn + fp
        tpfn = tp + fn
        tpfp = tp + fp
        tnfn = tn + fn
        if tnfp != 0:
            spec = tn/tnfp
        else:
            spec = 0
        if tpfn != 0:
            sens = tp/tpfn
        else:
            sens = 0
        if tpfp != 0:
            ppv = tp/tpfp
        else:
            ppv = 0
        if tnfn != 0:
            npv = tn/tnfn
        else:
            npv = 0
            
        if print_:
            print("Accuracy       : ", accs)
            print("F1             : ", f1s)
            print("Sens/Recall    : ", sens)
            print("Specificty     : ", spec)
            print("Pos Prev Val   : ", ppv)
            print("Neg Prev Val   : ", npv)
            print("AUC            : ", aucs)
            print("R2 (McFadden)  : ", m_r2)
            print("R2 (Tjur)      : ", t_r2)
            print("R2 (Nagelkerke): ", n_r2)
            print("R2 (Cox/Snell) : ", cs_r2)
    
        res = {
            'Accuracy': accs,
            'F1': f1s,
            'Sens/Recall': sens,
            'Specificity': spec,
            'ppv': ppv,
            'npv': npv,
            'Conf_Mat': confMat,
            'AUC': aucs,
            'McFadden R2': m_r2,
            'Tjur R2': t_r2,
            'Nagelkerke R2': n_r2,
            'Cox/Snell R2': cs_r2
        }
                  
    else:
        Ypred = model.predict(X)
        Ydiff = Ypred - Y.values
        Ybar = np.mean(Y.values)
        
        bias = np.mean(Ydiff)
        rpmse = np.sqrt(np.mean(Ydiff**2))
        SST = np.sum((Y.values - Ybar)**2)
        #SSE = np.sum((Y.values - Ypred)**2)
        SSR = np.sum((Ypred - Ybar)**2)
        # also equal to 1 - SSE/SST
        # alternative R2: np.corrcoef((ypred, ytrue))**2
        r2 = SSR/SST
        adj_r2 = 1 - (1 - r2)*((X.shape[0] - 1)/(X.shape[0] - X.shape[1] - 1))
        
        if print_:
            print("Bias       : ", bias)
            print("RPMSE      : ", rpmse)
            print("R2         : ", r2)
            print("Adj. R2    : ", adj_r2)
    
        res = {
            'Bias': bias,
            'RPMSE': rpmse,
            'R2': r2,
            'Adj. R2': adj_r2
        }
        
    return res


def runScorers(X, Y, splits, model, mtype, metrics=['Accuracy',
                                                    'F1',
                                                    'Sens/Recall',
                                                    'Specificity',
                                                    'ppv',
                                                    'npv',
                                                    'AUC'],
               avg='weighted', calculate=['Out of Sample'],
               method_on_X=None, mox_args={}, Y_for_test_only=None,
               sample_limit=20):
    '''
    Performs the cross-validation.

    Parameters
    ----------
    X : pandas dataframe
        Feature or design marix.
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
        The default is ['Accuracy', 'F1', 'Sens/Recall',
                        'Specificity', 'ppv', 'npv', 'AUC'].
    avg : str, optional
        for metrics where it applies, such as AUC, how to calculate the
        metric (see sklearn docs for more details, particularily with AUC).
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
        method_on_X. The default is {}.
    Y_for_test_only : pandas series or numpy array, optional
        An alternate target variable to test on, mainly for the use case of
        training on one response (perhaps one that is more informative or
        restrictive), but then predicting on a seperate test set.
        The default is None.
    sample_limit : int, optional
        The minimum acceptable samples in a split for the response
        variable. Only applies to Y_for_test_only currently.
        The default is 20.

    Returns
    -------
    keys : dict
        A dictionary containing the results.

    '''
    #is_score = None
    #oos_score = None
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
    '''
    A helper function to print the outputs of the crossVal function.

    Currently, it does not print confusion matrices.    

    Parameters
    ----------
    scores : dict
        The dictionary of scores.

    Returns
    -------
    None.

    '''
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
             random_state=None, method_on_X=None, mox_args={},
             avg='weighted', shuffle=True, test_size=0.1,
             calculate=['Out of Sample'],
             metrics=['Accuracy', 'F1', 'Sens/Recall', 'Specificity',
                      'ppv', 'npv', 'AUC'],
             Y_for_test_only=None, sample_limit=20):
    # TODO: add parallel support
    '''
    A custom cross-validation strategy. While sklearn has a lot of beefed up
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
        The default is ['Accuracy', 'F1', 'Sens/Recall',
                        'Specificity', 'ppv', 'npv', 'AUC'].
    Y_for_test_only : pandas series or numpy array, optional
        An alternate target variable to test on, mainly for the use case of
        training on one response (perhaps one that is more informative or
        restrictive), but then predicting on a seperate test set.
        The default is None.
    sample_limit : int, optional
        The minimum acceptable samples in a split for the response
        variable. Only applies to Y_for_test_only currently.
        The default is 20.

    Returns
    -------
    scores : dict
        Dictionary of performance metrics.

    '''
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
def prettyConfMat(Ytrue, Ypred,
                  print_=True, margins=True, labels=None):
    '''
    Print a pretty confusion matrix.

    Parameters
    ----------
    Ytrue : numpy array or pandas series
        The true class values.
    Ypred : numpy array or pandas series
        The predicted class values.
    print_ : bool, optional
        Print the confusion matrix at the end. The default is True.
    margins : bool, optional
        Include the summed margins in the matrix. The default is True.
    labels : dict, optional
        A dictionary of key-value pairs, where the keys are the labels,
        and the values are the class values in Ytrue. class labels for the
        confusion matrix. The default is None.

    Raises
    ------
    KeyError
        If labels are provided, an error is raised if the size of the array
        containing the labels does not match the number of possible classes.

    Returns
    -------
    confmat : pandas dataframe
        A dataframe containing the confusion matrix.

    '''
    if labels is not None:
        if isinstance(Ytrue, pd.DataFrame) or isinstance(Ytrue, pd.Series):
            Ytrue = Ytrue.copy().values
        if isinstance(Ypred, pd.DataFrame) or isinstance(Ypred, pd.Series):
            Ypred = Ypred.copy().values
        maxlen = np.unique(Ytrue).shape[0]
        if len(labels) != maxlen:
            raise KeyError('Incorrect number of labels compared to truth label values')
        else:
            idx = np.arange(0, Ytrue.shape[0])
            Ytrue_label = pd.Series(index=idx, dtype=object)
            Ypred_label = pd.Series(index=idx, dtype=object)
            for lab in labels.keys():
                Ytrue_label.loc[Ytrue == labels[lab]] = lab
                Ypred_label.loc[Ypred == labels[lab]] = lab
    else:
        Ytrue_label = Ytrue.copy()
        Ypred_label = Ypred.copy()
    
    cm_vals = pd.crosstab(Ytrue_label, Ypred_label, margins=margins, margins_name='Total')

    rowcolkeys = [lab for lab in labels.keys()] + ['Total']
    cm_vals = cm_vals.reindex(rowcolkeys, axis=0)
    cm_vals = cm_vals.reindex(rowcolkeys, axis=1)
    
    rownames = pd.MultiIndex.from_product([['Actual'], rowcolkeys])
    colnames = pd.MultiIndex.from_product([['Predicted'], rowcolkeys])
    #confmat = cm_vals.reindex(pd.MultiIndex.from_product([['Actual'], [lab for lab in labels.keys()] + ['All']]))
    confmat = pd.DataFrame(cm_vals.values, index=rownames, columns=colnames)

    if print_:
        print(confmat)
    return confmat


def RegressionSE(X, Y, fit_mod, logit=True, low_memory=False):
    '''
    Since sklearn doesn't generate the standard errors by default...

    Parameters
    ----------
    X : pandas dataframe
        The feature or design matrix.
    Y : pandas series or numpy array
        The response or target variable.
    fit_mod : sklearn model, or similar
        The fitted model (Logistic or linear Regression).
    logit : bool, optional
        If True, calculate Logistic Regression SE, otherwise, Linear. Default is True
    low_memory : bool, optional
        Sometimes this process uses a lot of memory. If True, calculate using less memory.
        The default is False.

    Returns
    -------
    se : numpy array
        The standard error of the coefficients.

    '''
    se = np.nan
    Xt = np.hstack([np.ones((X.shape[0], 1)), X])
    if logit:
        #Ypred = fit_mod.predict(X)
        Yprob = fit_mod.predict_proba(X)
        Yprob_prod = np.product(Yprob, axis=1)
        if low_memory:
            XV = np.zeros(shape=Xt.T.shape)
            for j in range(X.T.shape[1]):
                XV[:, j] = Xt.T[:, j] * Yprob_prod[j]
            XVXt = np.zeros((XV.shape[0], XV.shape[0]))
            try:
                XVXt = XV.dot(Xt)
            except MemoryError:
                for i in range(XV.shape[0]):
                    for k in range(XV.shape[1]):
                        sumprod = np.sum(np.product(XV[i, :], Xt[:, k]))
                        XVXt[i, k] = sumprod
        else:
            # p*(1-p), or pq
            Yprob_prod = np.diagflat(np.product(Yprob, axis=1))
            # cov of logit
            XVXt = np.dot(np.dot(Xt.T, Yprob_prod), Xt)
        cov = np.linalg.inv(XVXt)
    else:
        betas = np.concatenate((fit_mod.intercept_, fit_mod.coef_)).reshape(-1, 1)
        sigma2 = np.sum((Y - Xt.dot(betas))**2)/(Xt.shape[0] - Xt.shape[1])
        cov = sigma2*np.linalg.inv(np.dot(Xt.T, Xt))
    se = np.sqrt(np.diag(cov))  
    return se


def corrmat_validity_check(corr_mat, corr_tol=1e-8):
    '''
    Check for perfect correlation and/or a singular covariance matrix. Either
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

    '''
    det = np.linalg.det(corr_mat)
    ncols = corr_mat.shape[0]
    corr_mat_flat = corr_mat.reshape(-1, )
    has_perfect_corr = (np.abs(corr_mat_flat) >= 1-corr_tol).reshape(ncols, ncols)
    np.fill_diagonal(has_perfect_corr, 0)
    return has_perfect_corr, det


def vif(data, root=False, corr_tol=1e-8, sing_tol=1e-15):
    '''
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

    '''
    corr = np.corrcoef(data, rowvar=False)
    perfect_corr, det = corrmat_validity_check(corr, corr_tol=corr_tol)
    if perfect_corr.sum() > 0:
        warnings.warn("""There are perfectly correlated variables. VIF
                         may not be reliable.""")
    if det < sing_tol:
        warnings.warn("""The corr_mat determinant is essentially zero. VIF
                         may not be reliable.""")
    
    vifs = np.diag(np.linalg.inv(corr))
    
    if isinstance(data, pd.DataFrame):
        vifs = pd.DataFrame(vifs,
                            index=data.columns,
                            columns=['Variance Inflation Factors'])
    if np.any(vifs < 1):
        warnings.warn("""Some vifs are less than one. Model is not correctly
                      specified or is not a suitable linear model""")
                      
    if root:
        return np.sqrt(vifs)
    else:
        return vifs


def beta_trans(coef, type_="percent"):
    '''
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

    '''
    val = np.exp(coef)
    if type_ == "odds":
        return val
    elif type == "decimal":
        return val - 1
    elif type_ == "percent":
        return 100.0*(val - 1)
    else:
        return coef
    

def getPriors(data, Ycol):
    '''
    Calculate the class distributions across each column of the dataset.
    
    For classification problems, but could easily be used with regression
    problems if an idicator variable was made that indicated some split in the
    data.

    Parameters
    ----------
    data : pandas dataframe
        The feature or design matrix, along with the reponse variable.
    Ycol : str
        The column representing the target or response variable.

    Returns
    -------
    alltab : pandas dataframe
        A dataframe containing all of the results.

    '''
    tab = None
    alltab = None
    #numeric = False
    sub = data
    for k, i in enumerate(sub.columns[:-1]):
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
    '''
    This is adapted from David Batista, so credit goes to him. 
    
    see here: davidsbatista.net/blog/2018/02/23/model_optimization
    
    I added some functionality that will extend to the sklearn-deap package
    as well. see here: https://github.com/rsteca/sklearn-deap
    '''
    def __init__(self, models, params, searchCV, searchCVparams):
        '''
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

        '''
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
        '''
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

        '''
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
        '''
        Create a summary of the evolutionary algorithm results.

        Parameters
        ----------
        sort_by : list or list-like
            Columns to sort by, for example, 'mean_score'.

        Returns
        -------
        pandas dataframe
            A dataframe containing the results.

        '''
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
        '''
        Create a summary of the Grid Search results.

        Parameters
        ----------
        sort_by : list or list-like
            Columns to sort by, for example, 'mean_score'.

        Returns
        -------
        pandas dataframe
            A dataframe containing the results.

        '''
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
        '''
        Create a summary of the scores and results.

        Parameters
        ----------
        sort_by : list or list-like
            Columns to sort by, for example, 'mean_score'.

        Returns
        -------
        scores : pandas dataframe
            A dataframe containing the results.

        '''
        scores = None
        if self.gs_name == 'EvolutionaryAlgorithmSearchCV':
            scores = self._score_summary_ea(sort_by=sort_by)
        else:
            scores = self._score_summary_skgs(sort_by=sort_by)
        return scores


def BPCA_initmodel(y, q):
    '''
    Initialize the bPCA model.

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

    '''
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

    M['mu'] = np.nansum(y, axis=0)/np.array([np.sum(~np.isnan(y[:, col])) for col in range(y.shape[1])])

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
    '''
    The workhorse of the bPCA algorithm, the Expectation-Maximization
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

    '''
    N = M['N']
    d = M['d']
    
    Rx = np.eye(M['q']) + M['tau']*M['W'].T.dot(M['W']) + M['SigW']
    Rxinv_o = np.linalg.inv(Rx)

    idx = M['gnomiss']
    n = len(idx)
    
    dy = y[idx, :] - np.tile(M['mu'], (n, 1))
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
        
        dym = Wm.dot(x)
        dy = y[i, :].copy()
        dy[M['nomissidx'][i]] = dyo.T
        dy[M['missidx'][i]] = dym.T
        M['yest'][i, :] = dy + M['mu']

        Tt = Tt + reshape_to_vect(dy).dot(reshape_to_vect(x).T)
        Tt[M['missidx'][i], :] = Tt[M['missidx'][i], :] + Wm.dot(Rxinv)
        trS = trS + dy.dot(dy.T) + len(M['missidx'][i])/M['tau'] + np.trace(Wm.dot(Rxinv).dot(Wm.T))

    Tt = Tt/N
    trS = trS/N

    Dw = Rxinv_o + M['tau']*Tt.T.dot(M['W']).dot(Rxinv_o) + np.diag(M['alpha'])/N
    Dwinv = np.linalg.inv(Dw)
    M['W'] = Tt.dot(Dwinv)
    
    M['tau'] = (d + 2*M['gtau0']/N)/(trS - np.trace(Tt.T.dot(M['W'])) + (M['mu'].dot(M['mu'].T)*M['gmu0'] + 2*M['gtau0']/M['btau0'])/N)
    M['SigW'] = Dwinv*(d/N)
    M['alpha'] = (2*M['galpha0'] + d)/(M['tau']*np.diag(M['W'].T.dot(M['W'])) + np.diag(M['SigW']) + 2*M['galpha0']/M['balpha0'])

    return M
        


# assumes no rows of all NaNs
def bPCA(data, k=None, maxepoch=200, stepsize=10, dtau_tol=1e-8):
    '''
    An algorithm to compute missing values using Bayesian PCA. Translated from
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
            mu: The estimated mean row vector
            W: The estimated principal axis matrix
            tau: The estimated precision (inverse variance) of the residual 
                 error

    '''
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
        