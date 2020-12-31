import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from time import mktime
from timeit import default_timer as dt

from .feature_selection import dropHighCorrs, getHighCorrs, naiveVarDrop
from .metrics import aicCalc, glm_regularized_AIC as glmAIC
from .modeling import crossVal, prettyConfMat
from .parallel import p_prog_simp
from .utils import print_time, updateProgBar


def calcAICsLasso(penalty, Xtrain, Ytrain, sample_size, unreg_full_mod=None,
                  random_state=123):
    '''
    Utility function for quick_analysis. Need to update this to include more
    GLM models.

    Parameters
    ----------
    penalty : flost
        The strength of the L1 penalty.
    Xtrain : numpy array or pandas dataframe
        The design or feature matrix.
    Ytrain : numpy array or pandas series
        The target or response variable.
    sample_size : int
        The number of observations.
    unreg_full_mod : sklearn object, or similar, optional
        The unregularized full model. The default is None.
    random_state : int, optional
        Random seed for the process. The default is 123.

    Returns
    -------
    nzc : list
        A list indicating the non-zero coefficients.
    aics : list
        A list containing the AIC values.

    '''
    cur_mod_reg = LogisticRegression(C=penalty, max_iter=10000, penalty="l1",
                                     solver='liblinear',
                                     random_state=random_state,
                                     n_jobs=1).fit(Xtrain, Ytrain)
    Yprob = cur_mod_reg.predict_proba(Xtrain)[:, 1]
    Ypred = cur_mod_reg.predict(Xtrain)
    mod_coef = np.concatenate((cur_mod_reg.intercept_,
                               np.squeeze(cur_mod_reg.coef_)))
    nzc = sum((1 if coef != 0 else 0 for coef in mod_coef))
    loglike = np.sum(Ytrain*np.log(Yprob) + (1 - Ytrain)*np.log(1 - Yprob))
    
    aics = []
    aics.append(aicCalc(loglike, nzc, sample_size, c=2, metric="aic"))
    aics.append(aicCalc(loglike, nzc, sample_size, c=2, metric="aicc"))
    aics.append(aicCalc(loglike, nzc, sample_size, c=2, metric="bic"))
    aics.append(aicCalc(loglike, nzc, sample_size, c=2, metric="ebic"))
    aics.append(np.mean(Ypred != Ytrain))
    aics.append(f1_score(Ytrain, Ypred))
    if unreg_full_mod is not None:
        aics.append(glmAIC(Xtrain, Ytrain, cur_mod_reg, unreg_full_mod,
                             tol=1e-8, method="kawano"))
        aics.append(glmAIC(Xtrain, Ytrain, cur_mod_reg, unreg_full_mod,
                             tol=1e-8, method="hastie"))
    
    return nzc, aics


def parallel_AIC_data_retriever(save_path, save_name, num_chunks, sample_size,
                                unreg_full_mod, random_state, target_var,
                                cur_pred_list, penalty):
    '''
    Utility function for QuickAnalysis. Not to be used externally.

    Parameters
    ----------
    save_path : str
        The directory path to save temporary files to.
    save_name : str
        The name of the temporary file to save.
    num_chunks : int
        The number of chunks for the pandas msgpacks.
    sample_size : int
        The number of observations.
    unreg_full_mod : sklearn object, or similar
        Unregularized full model.
    random_state : int
        Random seed for the process.
    target_var : str
        The name of the target variable.
    cur_pred_list : list
        The list of current predictors in the model.
    penalty : float
        The strength of the L1 penalty.

    Returns
    -------
    res : tuple
        Tuple of results (non-zero coefficients and aic values).

    '''
    # Note: Should probably update this to feather (as of 2019)
    # since it is faster and better on memory in general.
    # Since this is not long term storage, feather would be a 
    # good choice. However, for long-term storage, parquet is 
    # probably the best option
    
    #data = pd.read_msgpack(save_path + save_name + '.msg')
    #chunks = [int(x[6:]) for x in data.keys()]
    #chunks = sorted(chunks)
    #mergedDF = pd.DataFrame()
    #for chunk in chunks:
    #    mergedDF = mergedDF.append(data['chunk_'+str(chunk)])
    mergedDF = pd.read_feather(save_path + save_name + '.fth')
    
    Xtrain = mergedDF.loc[:, cur_pred_list]
    Ytrain = mergedDF.loc[:, target_var]
    
    res = calcAICsLasso(penalty=penalty,
                        Xtrain=Xtrain,
                        Ytrain=Ytrain,
                        sample_size=sample_size,
                        unreg_full_mod=unreg_full_mod,
                        random_state=random_state)
    
    # For debugging purposes, to attempt to force garbage collection in parallel
    mergedDF = []
    Xtrain = []
    Ytrain = []
    return res


class QuickAnalysis(object):
    '''
    A class for performing a quick sweep of the data. Helps establish a
    baseline model to get started with, as well as quickly identify potentially
    useful features.
    '''
    def __init__(self, train, holdout, target_var, low_memory=False,
                 name="QuickAnalysis"):
        '''
        Constructor for QuickAnalysis

        Parameters
        ----------
        train : pandas dataframe
            The training dataset.
        holdout : pandas dataframe
            The holdout set (for evaluation of the final model).
        target_var : str
            The target or response variable.
        low_memory : bool, optional
            For instances where it applies, attempts to use less memory.
            The default is False.
        name : str, optional
            A name for the modeling object. The default is "QuickAnalysis".

        Returns
        -------
        None.

        '''
        self._full_analysis_arglist = None
        self._bdp_analysis_arglist = None
        self._fsa_analysis_arglist = None
        self.name = name
        self.train_raw = train.copy()
        self.holdout_raw = holdout.copy()
        if low_memory:
            self.train_raw = None
            self.holdout_raw = None
        self.low_memory = low_memory
        self.train = train.copy()
        self.holdout = holdout.copy()
        self.target_var = target_var
        self.steps = {}
        self.step_num = 0
        self.predictors_raw = train.columns[np.where(target_var != train.columns)[0]]
        self.cur_pred_list = train.columns[np.where(target_var != train.columns)[0]]
        self.HCdropList = None
        self.y_corr = None
        self.corr_mat = None
        self.highCorr = None
        self.varDropList = None
        self.aics = None
        self.nonzero_coefs = None
        self.reduction = None
        self.opt_c = None
        self.optimal_num_params = None
        self.forwardElimMetrics = None
        self.num_params_forward = None
        self.coef_ord_red_final = None
        self.num_dropped_cols = {}
        
    
    def GenCorrStats(self, is_raw=False):
        '''
        Generate Correlation Statistics.

        Parameters
        ----------
        is_raw : bool, optional
            Is the raw data (vs. scaled data). The default is False.

        Returns
        -------
        None.

        '''
        corr_mat = None
        if is_raw:
            corr_mat = pd.DataFrame(np.corrcoef(self.train_raw.values,
                                                rowvar=False),
                                    index=self.train_raw.columns,
                                    columns=self.train_raw.columns)
        else:
            corr_mat = pd.DataFrame(np.corrcoef(self.train.values,
                                                rowvar=False),
                                    index=self.train.columns,
                                    columns=self.train.columns)
        self.y_corr = corr_mat.loc[self.cur_pred_list, self.target_var]
        self.corr_mat = corr_mat
    
    def sort_y_corr(self):
        '''
        Sort columns by their correlation with the target variable.

        Returns
        -------
        pandas series
            Sorted Correlations.

        '''
        if self.y_corr is None:
            print("No y_corr recorded. Run GenCorrMat")
        else:
            return self.y_corr.abs().sort_values(ascending=False)
    
    def plot_y_corr(self):
        '''
        Plot variable correlations with the target variable.

        Returns
        -------
        None.

        '''
        sorted_corrs_with_y = self.y_corr.sort_values()

        plt.figure(figsize=(10, 5))
        ax = plt.subplot(1, 1, 1)

        ax.plot(sorted_corrs_with_y.values)
        plt.ylabel("Correlation")
        plt.xlabel("Sorted Features")
        plt.title("Correlation of Each Feature with " + self.target_var)
        plt.show()
        sorted_corrs_with_y = []
    
    def genHighCorrs(self, corr_cutoff):
        '''
        Generate a list of variables that are highly correlated with one
        another.

        Parameters
        ----------
        corr_cutoff : float
            Cutoff value to determine high correlations.

        Returns
        -------
        None.

        '''
        self.highCorr = getHighCorrs(self.corr_mat, corr_cutoff)
    
    def calcAICs(self, num_cs, try_parallel, n_jobs, random_state, remove_msg,
                 chunk):
        '''
        Calculate AIC values for the models to prepare for comparison.

        Parameters
        ----------
        num_cs : int, optional
            number of penalty values to test (i.e. number of models to evaluate
            during AIC calculations). 
        try_parallel : bool
            Whether to try running the operation in parallel.
        n_jobs : int
            Number of parallel processes to run.
        random_state : int
            Random seed for the process.
        remove_msg : bool
            Remove the temporary pandas msgpacks once calculations are
            complete.
        chunk : bool
            Break the operation into chunks.

        Returns
        -------
        None.

        '''
        nonzero_coefs = []
        aics = np.zeros(shape=(num_cs, 6))
        
        res = None
        
        if try_parallel:
            num_chunks = int(((self.train.memory_usage(index=True).sum()/(1024**3))/2) + 1)
            
            save_path = 'qr_tmp'
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            save_name = '/qr_run_' + str(int(mktime(datetime.now().timetuple())))
            
            #pd.to_msgpack(save_path + save_name + '.msg', {
            #        'chunk_{0}'.format(i):chunk for i, chunk in enumerate(np.array_split(self.train, num_chunks))
            #                                              })
            temp_train = self.train.copy(deep=True)
            temp_train.to_feather(save_path + save_name + '.fth')
            
            size = self.train.shape[0]
            arr = {'save_path': save_path,
                   'save_name': save_name,
                   'num_chunks': num_chunks,
                   'sample_size': size,
                   'unreg_full_mod': None,
                   'random_state': random_state,
                   'target_var': self.target_var,
                   'cur_pred_list': list(self.cur_pred_list),
                  }
            
            if chunk:
                res = []
                num_chunks = int((num_cs/n_jobs) + 1)
                
                for chunk in range(num_chunks):
                    loop_arr = [{'penalty': red} for red in self.reduction[chunk*n_jobs:((chunk+1)*n_jobs)]]
                    #temp_jobs = len(loop_arr)
                    new_res = p_prog_simp(args=arr, loop_args=loop_arr,
                                  function=parallel_AIC_data_retriever,
                                  n_jobs=n_jobs)
                res = res + new_res
            else:
                
                loop_arr = [{'penalty': red} for red in self.reduction]

                res = p_prog_simp(args=arr, loop_args=loop_arr,
                                  function=parallel_AIC_data_retriever,
                                  n_jobs=n_jobs)

            if remove_msg:
                #file_ = save_path + save_name + '.msg'
                file_ = save_path + save_name + '.fth'
                try:
                    os.remove(file_)
                except FileNotFoundError:
                    print('Could Not find file ', file_, 'Continuing...')
            
            arr = []
            loop_arr = []
        
        else: 
            res = [calcAICsLasso(penalty=red,
                                 Xtrain=self.train.loc[:, self.cur_pred_list],
                                 Ytrain=self.train.loc[:, self.target_var],
                                 sample_size=self.train.shape[0],
                                 unreg_full_mod=None,
                                 random_state=random_state) for red in self.reduction]
        
        for i, re in enumerate(res):
            nonzero_coefs.append(re[0])
            aics[i, :] = re[1]

        self.aics = aics
        self.nonzero_coefs = list(nonzero_coefs)
    
    def plotAICs(self):
        '''
        Plot the AIC calculations.

        Returns
        -------
        None.

        '''
        fig, ax = plt.subplots(figsize=(16,10))
        ax.plot(self.nonzero_coefs, self.aics[:, 1])
        ax.plot(self.nonzero_coefs, self.aics[:, 2])
        ax.set_ylabel("AIC/BIC Value")
        ax.set_xlabel("Number of Model Parameters")
        ax2 = ax.twinx()
        ax2.plot(self.nonzero_coefs, self.aics[:, 4], color="purple")
        ax2.plot(self.nonzero_coefs, self.aics[:, 5], color="green")
        ax2.set_ylabel("F1 Score and Misclassification Rate")
        ax.legend(["aicc", "bic"])
        ax2.legend(["misclass", "f1"])
        plt.title("Number of Non-zero Parameters by Change in L1 Penalty")
        plt.show()
        
    def forwardElimMetricCalc(self, coef_ord_red, stride, random_state):
        '''
        Run a very basic forward elimination procedure, and calculate model 
        metrics.

        Parameters
        ----------
        coef_ord_red : list
            A list of model coefficients, ordered by importance.
        stride : int
            Window size of the coefficients to test in the forward selection
            procedure. For example, stride=1 is 1,2,3,4,..., stride=2 is
            1,3,5,7,...
        random_state : int
            Random seed for procedure.

        Returns
        -------
        None.

        '''
        num_params = np.arange(1, self.train.shape[1]-1, stride)
        # f1, acc, sens, spec, auc
        metrics = np.zeros(shape=(len(num_params), 10))
        
        t0 = dt()

        for i, parm in enumerate(num_params):
            cur_mod_reg = LogisticRegression(C=self.opt_c, max_iter=10000,
                                             penalty="l1", solver='liblinear',
                                             random_state=random_state)
            # Cross-Val fully reduced model
            cols_to_include = list(coef_ord_red.index[0:parm])
            cur_mod_cv_results = crossVal(self.train.loc[:, cols_to_include],
                                          self.train.loc[:, self.target_var],
                                          5, cur_mod_reg, print_=False)

            metrics[i, 0:2] = (np.mean(cur_mod_cv_results['Out of Sample']["Accuracy"]),
                               np.std(cur_mod_cv_results['Out of Sample']["Accuracy"], ddof=1))
            metrics[i, 2:4] = (np.mean(cur_mod_cv_results['Out of Sample']["F1"]),
                               np.std(cur_mod_cv_results['Out of Sample']["F1"], ddof=1))
            metrics[i, 4:6] = (np.mean(cur_mod_cv_results['Out of Sample']["Sens/Recall"]),
                               np.std(cur_mod_cv_results['Out of Sample']["Sens/Recall"], ddof=1))
            metrics[i, 6:8] = (np.mean(cur_mod_cv_results['Out of Sample']["Specificity"]),
                               np.std(cur_mod_cv_results['Out of Sample']["Specificity"], ddof=1))
            metrics[i, 8:10] = (np.mean(cur_mod_cv_results['Out of Sample']["AUC"]),
                               np.std(cur_mod_cv_results['Out of Sample']["AUC"], ddof=1))
            
            updateProgBar(i + 1, len(num_params), t0)
        self.forwardElimMetrics = metrics
        self.num_params_forward = num_params
        
    def plotForwardMetrics(self):
        '''
        Plot forward elimination metric results.

        Returns
        -------
        None.

        '''
        
        plt.figure(figsize=(10,7))
        
        acc_mean = self.forwardElimMetrics[:, 0].reshape(-1, 1)
        f1_mean = self.forwardElimMetrics[:, 2].reshape(-1, 1)
        sens_mean = self.forwardElimMetrics[:, 4].reshape(-1, 1)
        spec_mean = self.forwardElimMetrics[:, 6].reshape(-1, 1)
        auc_mean = self.forwardElimMetrics[:, 8].reshape(-1, 1)
        
        acc_bnds = self.forwardElimMetrics[:, 1].reshape(-1, 1).dot(np.array([-2, 2]).reshape(1, 2)) + acc_mean
        f1_bnds = self.forwardElimMetrics[:, 3].reshape(-1, 1).dot(np.array([-2, 2]).reshape(1, 2)) + f1_mean
        sens_bnds = self.forwardElimMetrics[:, 5].reshape(-1, 1).dot(np.array([-2, 2]).reshape(1, 2)) + sens_mean
        spec_bnds = self.forwardElimMetrics[:, 7].reshape(-1, 1).dot(np.array([-2, 2]).reshape(1, 2)) + spec_mean
        auc_bnds = self.forwardElimMetrics[:, 9].reshape(-1, 1).dot(np.array([-2, 2]).reshape(1, 2)) + auc_mean
       
        plt.plot(self.num_params_forward, acc_mean)
        plt.fill_between(self.num_params_forward,
                         acc_bnds[:, 0], acc_bnds[:, 1], alpha=0.5)
        plt.plot(self.num_params_forward, f1_mean)
        plt.fill_between(self.num_params_forward,
                         f1_bnds[:, 0], f1_bnds[:, 1], alpha=0.5)
        plt.plot(self.num_params_forward, sens_mean)
        plt.fill_between(self.num_params_forward,
                         sens_bnds[:, 0], sens_bnds[:, 1], alpha=0.5)
        plt.plot(self.num_params_forward, spec_mean)
        plt.fill_between(self.num_params_forward,
                         spec_bnds[:, 0], spec_bnds[:, 1], alpha=0.5)
        plt.plot(self.num_params_forward, auc_mean)
        plt.fill_between(self.num_params_forward,
                         auc_bnds[:, 0], auc_bnds[:, 1], alpha=0.5)
        
        plt.xlabel("Number of Model Features")
        plt.ylabel("Metric Values")
        plt.title("Change in Model Metrics as Features Increase")
        plt.legend(["acc", "f1", "sens", "spec", "auc"])
        plt.show()
    
    def confusion_matrix(self, print_=True):
        '''
        Generate a confusion matrix (for classification data).

        Parameters
        ----------
        print_ : bool, optional
            Print the confusion matrix to the console. The default is True.

        Returns
        -------
        confmat : pandas dataframe
            A dataframe containing the confusion matrix.

        '''
        confmat = None
        if self.Ytest is None:
            print("Run QuickAnalysis first")
        else:
            confmat = prettyConfMat(self.Ytest, self.Ytest_pred, print_=print_)
        if print_:
            print(confmat)
        return confmat
    
    def build_model(self, name, cv_iterations, penalty, random_state):
        '''
        A utility function for building the model at different steps of the 
        QuickAnalysis process.

        Parameters
        ----------
        name : str
            The step where the model is being built.
        cv_iterations : int
            Number of cross validation steps.
        penalty : float
            The strength of the L1 penalty.
        random_state : int
            Random seed of the process.

        Returns
        -------
        None.

        '''
        mod_crossval = crossVal(self.train.loc[:, self.cur_pred_list],
                                self.train.loc[:, self.target_var],
                                cv_iterations=cv_iterations,
                                model=LogisticRegression(C=penalty,
                                                         penalty="l1",
                                                         solver="liblinear",
                                                         max_iter=10000,
                                                         random_state=random_state),
                                print_=False)
        self.steps[name] = {
                'crossval': mod_crossval,
                'model': LogisticRegression(C=penalty,
                                            penalty="l1",
                                            solver="liblinear",
                                            max_iter=10000,
                                            random_state=random_state).fit(self.train.loc[:, self.cur_pred_list],
                                                                           self.train.loc[:, self.target_var]),
                'predictors': list(self.cur_pred_list)
            }
        
    
    def BaselineDropProcedures(self, freshStart=True, downsample=False,
                               dropCols=None, dropCor=True, corr_cutoff=0.9,
                               dropVar=True, dropVarTol=0.001,
                               cv_iterations=5,
                               random_state=123, default_penalty=0.01,
                               verbose=True, t0=None):
        '''
        Create Baseline model and begin dropping features from the model 
        based on given criteria.

        Parameters
        ----------
        freshStart : bool, optional
            Whether to use a previous result, or start from scrath.
            The default is True.
        downsample : bool, optional
            Whether or not to downsample the majority class.
            The default is False.
        dropCols : list, optional
            List of columns to initially drop. The default is None.
        dropCor : bool, optional
            Drop variables with high correlations. The default is True.
        corr_cutoff : float, optional
            Cutoff for removing high correlation variables. Anything below the
            cutoff is kept. The default is 0.9.
        dropVar : bool, optional
            Drop variables with very low variance.
            Does nothing if data is standardized. The default is True.
        dropVarTol : float, optional
            Cutoff value for low variance. The default is 0.001.
        cv_iterations : int, optional
            The number of cross validation steps. The default is 5.
        random_state : int, optional
            The random seed for the process. The default is 123.
        default_penalty : float, optional
            The penalty term to be applied to models where the penalty is
            otherwise unspecified. The default is 0.01.
        verbose : bool, optional
            Print steps as they complete. The default is True.
        t0 : float, optional
            The start time of the process. For internal use.
            The default is None.

        Returns
        -------
        None.

        '''
        self._bdp_analysis_arglist = {'freshStart': freshStart,
                                 'downsample': downsample,
                                 'dropCols': dropCols,
                                 'dropCor': dropCor,
                                 'corr_cutoff': corr_cutoff,
                                 'dropVar': dropVar,
                                 'dropVarTol': dropVarTol,
                                 'cv_iterations': cv_iterations,
                                 'random_state': random_state,
                                 'default_penalty': default_penalty,
                                 'verbose': verbose,
                                 't0': t0
                                 }
        
        if t0 is None:
            t0 = dt()
        
        if freshStart:
            if not self.low_memory:
                self.train = self.train_raw
                self.holdout = self.holdout_raw
            else:
                print("Can't run fresh start in low memory mode.")
        
        if verbose:
            print_time("\nCreating Baseline Model...", t0, te=dt())
        
        self.build_model('baseline_mod', cv_iterations, default_penalty,
                         random_state)
        
        if verbose:
            print_time("\nExecuting Column Drop Procedures...", t0, te=dt())

        if dropCols is not None:
            if verbose:
                print_time("\nDropping Pre-Determined Columns...", t0, te=dt())

            cols = list(self.train.columns)
            # what I used to do... 
            #idx = list(np.where([True if sum((True if reject in col else False for reject in dropCols))>0 else False for col in cols])[0])
            idx = [i for i, col in zip(range(len(cols)),
                                       cols) if col in dropCols]
            label_idx = list(self.train.columns[idx])
            self.train = self.train.drop(label_idx, axis=1)
            self.cur_pred_list = self.train.columns[np.where(self.target_var != self.train.columns)[0]]
            self.num_dropped_cols['dropCol_list'] = len(dropCols)
            if verbose:
                print_time("\nDropped " + str(len(dropCols)) + " Predefined Columns...",
                           t0, te=dt())
        
        if dropVar:
            if verbose:
                print_time("\nDropping Low Variance Columns...", t0, te=dt())

            varDropList = naiveVarDrop(self.train,
                                       list(self.train.columns[np.where(self.target_var != self.train.columns)[0]]),
                                       tol=dropVarTol,
                                       asList=True)
            
            self.train = self.train.drop(varDropList, axis=1)
            self.varDropList = varDropList
            self.cur_pred_list = self.train.columns[np.where(self.target_var != self.train.columns)[0]]
            self.num_dropped_cols['varDropList'] = len(varDropList)
            if verbose:
                print_time("\nDropped " + str(len(varDropList)) + " Low-Variance Columns...", t0, te=dt())
            varDropList = []
        
        if dropCor:
            if verbose:
                print_time("\nGenerating/Recovering Correlation Matrix...", t0,
                           te=dt())

            if self.corr_mat is None or self.y_corr is None:
                self.GenCorrStats(is_raw=False)
            if self.highCorr is None:
                self.genHighCorrs(corr_cutoff)
        
            if verbose:
                print_time("\nDropping High Correlations...", t0, te=dt())

            HCdropList = dropHighCorrs(self.train, self.highCorr, asList=True,
                                       print_=False)
            self.train = self.train.drop(HCdropList, axis=1)
            self.HCdropList = HCdropList
            self.num_dropped_cols['HCdropList'] = len(HCdropList)
            if verbose:
                print_time("\nDropped " + str(len(HCdropList)) + " Correlated Columns...", t0, te=dt())
            HCdropList = []
            self.cur_pred_list = self.train.columns[np.where(self.target_var != self.train.columns)[0]]
            self.plot_y_corr()
        
        if verbose:
            total_dropped = 0
            for key in self.num_dropped_cols.keys():
                total_dropped += self.num_dropped_cols[key]
            print_time("\nDropped " + str(total_dropped) + " Columns...", t0,
                       te=dt())
        
        if verbose:
            print_time("\nCreating Post-Drop Baseline Model...", t0, te=dt())
        
        self.build_model('postdrop_baseline_mod', cv_iterations,
                         default_penalty, random_state)
        
        if verbose:
            print_time("\nFinished Baseline Drop...", t0, te=dt())
    
    def ForwardSelectionAnalysis(self, cv_iterations=5, random_state=123,
                                 default_penalty=0.01, for_stride=2,
                                 reduce_features_by=30, red_metric='bic',
                                 verbose=True, t0=None):
        '''
        Run very basic forward selection procedure on the model.

        Parameters
        ----------
        default_penalty : float, optional
            The penalty term to be applied to models where the penalty is
            otherwise unspecified. The default is 0.01.
        for_stride : int
            Window size of the coefficients to test in the forward selection
            procedure. For example, stride=1 is 1,2,3,4,..., stride=2 is
            1,3,5,7,... The default is 2.
        reduce_features_by : int, optional
            Number of features to evaluate in the final model.
            The default is 30.
        red_metric : str, optional
            Metric to use for finding optimal penalty value.
            The default is 'bic'.
        verbose : bool, optional
            Print results of the process. The default is True.
        t0 : float, optional
            Initial start time for the process. The default is None.

        Returns
        -------
        None.

        '''
        # TODO: make final feature selection automatic
        self._fsa_analysis_arglist = {
                                      'cv_iterations': cv_iterations,
                                      'random_state': random_state,
                                      'default_penalty': default_penalty,
                                      'for_stride': for_stride,
                                      'reduce_features_by': reduce_features_by,
                                      'red_metric': red_metric,
                                      'verbose': verbose,
                                      't0': t0
        }
        
        if t0 is None:
            t0 = dt()
        
        self.plotAICs()
        aic_type = {
            'aic': 0,
            'aicc': 1,
            'bic': 2,
            'ebic': 3,
            'misclass': 4
        }
        self.opt_c = self.reduction[np.where(self.aics[:, aic_type[red_metric]] == np.min(self.aics[:, aic_type[red_metric]]))][0]
        self.optimal_num_params = np.array(self.nonzero_coefs)[np.where(self.aics[:, aic_type[red_metric]] == np.min(self.aics[:, aic_type[red_metric]]))][0]
        
        if verbose:
            print_time("\nCreating Model with Optimal Penalty Value...", t0,
                       te=dt())
        
        self.build_model('lasso_reduction_mod', cv_iterations, self.opt_c,
                         random_state)
        
        coef = self.steps['lasso_reduction_mod']['model'].coef_.reshape(-1,)
        coef_ord = pd.DataFrame(np.abs(coef), index=self.train.loc[:, self.cur_pred_list].columns,
                                columns=["Importance"])
        coef_ord = coef_ord.sort_values('Importance', ascending=False)
        
        keep_cols = list(coef_ord[coef_ord["Importance"] > 0].index)
        keep_cols.append(self.target_var)
        self.train = self.train.loc[:, keep_cols]
        self.cur_pred_list = self.train.columns[np.where(self.target_var != self.train.columns)[0]]
        self.steps['lasso_reduction_mod']['keep_cols'] = keep_cols
        keep_cols = []
        coef_ord = []
        coef = []
        
        if verbose:
            print_time("\nCreating Model with Optimal Penalty Value After Removing Zeroed Parameters...\n",
                       t0, te=dt())
        
        self.build_model('lasso_reduction_mod_reduced', cv_iterations, 1,
                         random_state)
        
        coef_red = self.steps['lasso_reduction_mod_reduced']['model'].coef_.reshape(-1,1)
        coef_ord_red = pd.DataFrame(np.concatenate((np.abs(coef_red), coef_red,
                                                    100.0*(np.exp(coef_red)-1)),
                                                   axis=1),
                                    index=self.train.loc[:, self.cur_pred_list].columns,
                                    columns=["Importance",
                                             "Coefficiants",
                                             "% increase in Prob"])
        coef_ord_red = coef_ord_red.sort_values('Importance', ascending=False)
        
        self.steps['lasso_reduction_mod_reduced']['coef_ord_red'] = coef_ord_red
        
        self.forwardElimMetricCalc(coef_ord_red, for_stride, random_state)
        self.plotForwardMetrics()
        
        cols_to_include = list(coef_ord_red.index[:reduce_features_by])
        self.cur_pred_list = cols_to_include
        
        if verbose:
            print_time("\nCreating Final Model...", t0, te=dt())
        
        self.build_model('final_mod', cv_iterations, 1, random_state)
        
        coef_red_final = self.steps['final_mod']['model'].coef_.reshape(-1,1)
        coef_ord_red_final = pd.DataFrame(np.concatenate((np.abs(coef_red_final), coef_red_final,
                                                          100.0*(np.exp(coef_red_final)-1)), axis=1),
                                    index=self.train.loc[:, cols_to_include].columns,
                                          columns=["Importance", "Coefficiants", "% increase in Prob"])
        coef_ord_red_final = coef_ord_red_final.sort_values('Importance', ascending=False)
        self.coef_ord_red_final = coef_ord_red_final
        
        self.Xtrain = self.train.loc[:, cols_to_include]
        self.Ytrain = self.train.loc[:, self.target_var]
        self.Ypred_train = self.steps['final_mod']['model'].predict(self.Xtrain)
        self.Yprob_train = self.steps['final_mod']['model'].predict_proba(self.Xtrain)

        self.Xtest = self.holdout.loc[:, cols_to_include]
        self.Ytest = self.holdout.loc[:, self.target_var]
        self.Ypred_test = self.steps['final_mod']['model'].predict(self.Xtest)
        self.Yprob_test = self.steps['final_mod']['model'].predict_proba(self.Xtest)[:, 1]
        
        if verbose:
            print_time("\nFinished...", t0, te=dt())
        
        
    def RunFullAnalysis(self, freshStart=True, downsample=False, dropCols=None,
                        dropCor=True, corr_cutoff=0.9, dropVar=True,
                        dropVarTol=0.001, cv_iterations=5,
                        random_state=123, default_penalty=0.01, verbose=True,
                        try_parallel=True, for_stride=2, reduce_features_by=30,
                        num_cs=100, red_metric='bic', red_log_low=-5,
                        red_log_high=1, n_jobs=1, remove_msg=True,
                        chunk=False):
        '''
        Performs a quick analysis by first considering which features/variables
        may be uninformative predictors based on preliminary models. Then,
        it performs a very basic forward selection procedure to limit
        predictors in the final model. Currently, this only supports
        Logistic Regression.

        Parameters
        ----------
        freshStart : bool, optional
            Whether to use a previous result, or start from scrath.
            The default is True.
        downsample : bool, optional
            Whether or not to downsample the majority class.
            The default is False.
        dropCols : list, optional
            List of columns to initially drop. The default is None.
        dropCor : bool, optional
            Drop variables with high correlations. The default is True.
        corr_cutoff : float, optional
            Cutoff for removing high correlation variables. Anything below the
            cutoff is kept. The default is 0.9.
        dropVar : bool, optional
            Drop variables with very low variance.
            Does nothing if data is standardized. The default is True.
        dropVarTol : float, optional
            Cutoff value for low variance. The default is 0.001.
        cv_iterations : int, optional
            The number of cross validation steps. The default is 5.
        random_state : int, optional
            The random seed for the process. The default is 123.
        default_penalty : float, optional
            The penalty term to be applied to models where the penalty is
            otherwise unspecified. The default is 0.01.
        verbose : bool, optional
            Print steps as they complete. The default is True.
        try_parallel : bool, optional
            Try to perform the operation in parallel. If it fails, it will
            continue without parallel operations. The default is True.
        for_stride : int, optional
            The gap between variables evaluated for the forward selection
            metric. The default is 2.
        reduce_features_by : int, optional
            The size of the final model. The default is 30.
        num_cs : int, optional
            number of penalty values to test (i.e. number of models to evaluate
            during AIC calculations). The default is 100.
        red_metric : str, optional
            The metric used to optimize model reductions. The default is 'bic'.
        red_log_low : float, optional
            The logspace minimum value for evaluating the L1 penalty.
            The default is -5.
        red_log_high : float, optional
            The logspace maximum value for evaluating the L1 penalty.. The default is 1.
        n_jobs : int, optional
            The number of processes to attempt. The default is 1.
        remove_msg : bool, optional
            Remove messagepacks. The default is True.
        chunk : bool, optional
            Chunk msgpacks? The default is False.

        Returns
        -------
        None.

        '''
        t0 = dt()
        self._full_analysis_arglist = {'freshStart': freshStart,
                                 'downsample': downsample,
                                 'dropCols': dropCols,
                                 'dropCor': dropCor,
                                 'corr_cutoff': corr_cutoff,
                                 'dropVar': dropVar,
                                 'dropVarTol': dropVarTol,
                                 'cv_iterations': cv_iterations,
                                 'random_state': random_state,
                                 'default_penalty': default_penalty,
                                 'try_parallel': try_parallel,
                                 'for_stride': for_stride,
                                 'reduce_features_by': reduce_features_by,
                                 'num_cs': num_cs,
                                 'red_metric': red_metric,
                                 'red_log_low': red_log_low,
                                 'red_log_high': red_log_high,
                                 'n_jobs': n_jobs,
                                 'verbose': verbose,
                                 'remove_msg': remove_msg}
        
        self.BaselineDropProcedures(freshStart=freshStart,
                                    downsample=downsample,
                                    dropCols=dropCols,
                                    dropCor=dropCor,
                                    corr_cutoff=corr_cutoff,
                                    dropVar=dropVar,
                                    dropVarTol=dropVarTol,
                                    cv_iterations=cv_iterations,
                                    random_state=random_state,
                                    default_penalty=default_penalty,
                                    verbose=verbose,
                                    t0=t0)
        
        if verbose:
            print_time("\nCreating Lasso Reduction Models...", t0, te=dt())

        self.reduction = np.logspace(red_log_low, red_log_high, num_cs)
        if try_parallel:
            try_again = True
            while(try_again):
                try:
                    self.calcAICs(num_cs, try_parallel, n_jobs,
                                  random_state=random_state,
                                  remove_msg=remove_msg, chunk=chunk)
                    try_again = False
                except (MemoryError):
                    n_jobs = int(n_jobs - (n_jobs/2))
                    print('\n\tReduced n_jobs to ', n_jobs)
                    if n_jobs == 1:
                        try_again = False
                    if not try_again:
                        self.calcAICs(num_cs, False, n_jobs=1,
                                      random_state=random_state,
                                      remove_msg=remove_msg, chunk=chunk)
        else:
            self.calcAICs(num_cs, False, n_jobs=1, random_state=random_state,
                          remove_msg=remove_msg, chunk=chunk)
                
        self.ForwardSelectionAnalysis(cv_iterations=cv_iterations,
                                      random_state=random_state,
                                      default_penalty=default_penalty,
                                      for_stride=for_stride,
                                      reduce_features_by=reduce_features_by,
                                      red_metric=red_metric,
                                      verbose=verbose, t0=t0)
