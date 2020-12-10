import numpy as np
import pandas as pd
import warnings

from copy import deepcopy as copy
from sklearn.metrics import (f1_score, confusion_matrix, recall_score, roc_curve,
                             roc_auc_score, auc, accuracy_score, r2_score)
from sklearn.model_selection import (StratifiedKFold, KFold, train_test_split,
                                     ShuffleSplit, StratifiedShuffleSplit)
from timeit import default_timer as dt

from .metrics import (cox_snell_r2, nagelkerke_r2, tjur_r2, mcfadden_r2,
                     conf_mat_metrics, bias, rpmse, r2, adj_r2, top_20p,
                     number_of_nonzero_coef)


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
            'r2': None,
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
                             'arguments': ['y_true', 'y_pred', 'X', 'r2']
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
            
            args['y_prob_nullmod'] = np.repeat(np.mean(Y.values), args['n_obs'])
            args['ll_est'] = np.sum(Y.values*np.log(args['y_score']) + (1 - Y.values)*np.log(1 - args['y_score']))
            args['ll_null'] = np.sum(Y.values*np.log(args['y_prob_nullmod']) + (1 - Y.values)*np.log(1 - args['y_prob_nullmod']))
                  
    else:
        if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.Series):
            args['y_true'] = Y.values
        args['y_pred'] = model.predict(X)
    
    for met in metrics:
        args['conf_metric'] = met
        temp_args = {k: args[k] for k in All_Metrics[met]['arguments']}
        res[met] = All_Metrics[met]['Function'](**temp_args) 
    
    if print_:
        print(res)
    
    return res


# Currently, this takes just as long and the logic is simpler... keeping for legacy reasons
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
        
        bias = np.mean(ydiff)
        rpmse = np.sqrt(np.mean(Ydiff**2))
        SST = np.sum((Y.values - Ybar)**2)
        SSE = np.sum((Y.values - Ypred)**2)
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
    
    is_score = None
    oos_score = None
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
        
        mod = copy(model).fit(Xtrain, Ytrain)
        
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

        
def crossVal(X, Y, cv_iterations, model, method='k-fold', mtype='classification',
             stratified=True, print_=True, random_state=123, method_on_X=None, mox_args={},
             avg='weighted', shuffle=True, test_size=0.1, calculate=['Out of Sample'],
             metrics=['Accuracy', 'F1', 'Sens/Recall', 'Specificity', 'ppv', 'npv', 'AUC'],
             Y_for_test_only=None, sample_limit=20):
  
    splits = None
    scores = None
    if mtype != 'classification':
        stratified = False
        
    if method == 'k-fold':
        if stratified:
            splits = StratifiedKFold(cv_iterations, shuffle=shuffle, random_state=random_state)
        else:
            splits = KFold(cv_iterations, shuffle=shuffle, random_state=random_state)
    else:
        if stratified:
            splits = StratifiedShuffleSplit(cv_iterations, test_size=test_size, random_state=random_state)
        else:
            splits = ShuffleSplit(cv_iterations, test_size=test_size, random_state=random_state)

    scores = runScorers(X, Y, splits, model, mtype, metrics=metrics, avg=avg, calculate=calculate,
                        method_on_X=method_on_X, mox_args=mox_args, Y_for_test_only=Y_for_test_only,
                        sample_limit=sample_limit)

    if print_:
        printScores(scores)
            
    return scores


# labels is a dictionary identifying what the label values should be,
# where the dictionary key is the new label, and the dictionary value
# is the current class value
def prettyConfMat(Ytrue, Ypred,
                  print_=True, margins=True, labels=None):
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


def RegressionSE(X, Y, fit_mod, low_memory=False):
    se = np.nan
    Ypred = fit_mod.predict(X)
    Yprob = fit_mod.predict_proba(X)
    Yprob_prod = np.product(Yprob, axis=1)
    Xt = np.hstack([np.ones((X.shape[0], 1)), X])
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
    se = np.sqrt(np.diag(cov))

    return se


def vif(data, root=False):
    '''
    Calculate Variance Inflation Factors. The sqrt of vif indicates how many times larger the standard error is than it would be if that variable had no correlation with the other variables 
    
    Expects pandas.DataFrame or numpy.array
    '''
    vifs = np.diag(np.linalg.inv(np.corrcoef(data.values, rowvar=False)))
    if isinstance(data, pd.DataFrame):
        vifs = pd.DataFrame(vifs, index=data.columns, columns=['Variance Inflation Factors'])
    if root:
        return np.sqrt(vifs)
    else:
        return vifs


def beta_trans(coef, type_="percent"):
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
    tab = None
    alltab = None
    numeric = False
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
    

# adapted from davidsbatista.net/blog/2018/02/23/model_optimization
class EstimatorSelectionHelper(object):
    def __init__(self, models, params, searchCV, searchCVparams):
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
        scores = None
        if self.gs_name == 'EvolutionaryAlgorithmSearchCV':
            scores = self._score_summary_ea(sort_by=sort_by)
        else:
            scores = self._score_summary_skgs(sort_by=sort_by)
        return scores
        