"""Test model scores, confidence intervals, searches, and feature selection."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from tsdst import feature_selection as f, modeling as m


def test_model_scores_and_cross_validation(classification, regression, capsys):
    x, y = classification
    model = LogisticRegression().fit(x, y)
    result = m.scoreModel(x, y, model, print_=False)
    assert result["Accuracy"] == accuracy_score(y, model.predict(x))
    assert m.scoreModel_legacy(x, y, model, print_=False)["Accuracy"] == result["Accuracy"]
    scores = m.runScorers(x, y, StratifiedKFold(2), model, "classification")
    assert len(scores["Out of Sample"]["Accuracy"]) == 2
    scores = m.crossVal(x, y, 2, model, random_state=3, print_=False,
                         calculate=["In Sample", "Out of Sample"], method_on_X=StandardScaler)
    m.printScores(scores)
    assert "Accuracy" in capsys.readouterr().out
    x, y = regression
    linear = LinearRegression().fit(x, y)
    assert m.scoreModel(x, y, linear, mtype="regression", print_=False)["R2"] > .9
    assert m.scoreModel_legacy(x, y, linear, mtype="regression", print_=False)["R2"] > .9
    assert len(m.crossVal(x, y, 2, linear, mtype="regression", print_=False)["Out of Sample"]["R2"]) == 2


def test_confusion_table_handles_default_and_custom_labels():
    truth, predicted = pd.Series([0, 0, 1, 1]), pd.Series([0, 1, 1, 1])
    for labels in [None, {"No": 0, "Yes": 1}]:
        table = m.prettyConfMat(truth, predicted, labels=labels, print_=False)
        assert table.iloc[-1, -1] == 4
    table = m.prettyConfMat(truth, predicted, labels={"No": 0, "Yes": 1}, margins=False, print_=False)
    assert table.shape == (2, 2)


@pytest.mark.parametrize("logit", [False, True])
def test_regression_covariance_and_intervals(logit, classification, regression):
    x, y = classification if logit else regression
    x, y = x.to_numpy(), y.to_numpy()
    model = (LogisticRegression() if logit else LinearRegression()).fit(x, y)
    covariance = m.GenerateRegCov(x, y, model, logit=logit)[0]
    assert covariance.shape == (4, 4)
    assert_allclose(m.RegressionSE(x, y, model, logit=logit)[0], np.sqrt(np.diag(covariance)))
    if logit:
        assert_allclose(m.GenerateRegCov(x, y, model, low_memory=True)[0], covariance)
    interval = m.predict(x[:5], x, y, model, logit=logit, interval="confidence", logit_transform=logit)
    assert interval.shape == (5, 3)
    assert np.all(interval.lower_int <= interval.yhat)
    assert np.all(interval.yhat <= interval.upper_int)


def test_vif_beta_and_priors(classification):
    x, y = classification
    assert m.corrmat_validity_check(x.corr().to_numpy())[0].sum() == 0
    assert np.all(m.vif(x).to_numpy() >= 1)
    assert m.beta_trans(np.log(2), "decimal") == pytest.approx(1)
    assert m.beta_trans(np.log(2), "odds") == pytest.approx(2)
    assert m.beta_trans(np.log(2), "percent") == pytest.approx(100)
    data = x.assign(category=np.tile([0, 1], len(x) // 2), y=y)
    assert m.getPriors(data, "y").shape[0] > 0


def test_estimator_search_summaries(classification):
    x, y = classification
    helper = m.EstimatorSelectionHelper({"lr": LogisticRegression()}, {"lr": {"C": [.1, 1]}},
              GridSearchCV, {"cv": 2, "scoring": {"accuracy": "accuracy"}, "refit": "accuracy"})
    assert helper.fit(x, y) is helper
    assert len(helper.score_summary()) == 2
    ea = m.EstimatorSelectionHelper({"lr": LogisticRegression()}, {"lr": {}}, GridSearchCV, {})
    ea.grid_searches = {"lr": SimpleNamespace(cv_results_={
        "params": [{"C": 1}], "min_test_score": [.6], "max_test_score": [.8],
        "mean_test_score": [.7], "std_test_score": [.1]})}
    assert len(ea._score_summary_ea("mean_score")) == 1


def test_bayesian_pca_imputes_missing_values():
    values = np.random.default_rng(3).normal(size=(30, 3))
    values[0, 0] = values[4, 1] = np.nan
    model = m.BPCA_initmodel(values, 1)
    assert np.isfinite(m.BPCA_dostep(model, values)["yest"]).all()
    result = m.bPCA(values, k=1, maxepoch=4)
    assert np.isfinite(result["yest"]).all()
    valid = ~np.isnan(values)
    assert_allclose(result["yest"][valid], values[valid])


def test_basic_feature_selection():
    frame = pd.DataFrame({"small": np.arange(10), "large": np.arange(10) * 2, "constant": 1})
    assert f.naiveVarDrop(frame, None, asList=True) == ["constant"]
    assert f.naiveScoreDrop(frame, pd.DataFrame({"score": [0., 1.]}, index=["small", "large"]), asList=True) == ["small"]
    pairs = f.getHighCorrs(frame.corr(), .9)
    assert pairs == {"small..&..large": 1.}
    assert f.dropHighCorrs(frame, pairs, asList=True) == ["small"]
    assert f.dropCorrProcedure(frame, .9).columns.tolist() == ["large", "constant"]
    with pytest.raises(NotImplementedError):
        f.vifDrop()


def test_permutation_importance_and_cross_validation(classification):
    x, y = classification
    before = x.copy(deep=True)
    model = LogisticRegression().fit(x, y)
    result = f.permutation_importance(model, x, y, accuracy_score, seed=2)
    assert result.shape == (3, 1)
    assert_frame_equal(x, before)
    assert f.permutation_importance_CV(LogisticRegression(), x, y, accuracy_score, num_splits=2, seed=3).shape == (3, 1)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("storage", ["feather", "parquet"])
def test_forward_selection_and_serialization(parallel, storage, regression, classification):
    x, y = regression
    assert np.isfinite(f._calcForwardAICs(x, y, LinearRegression(), "bic", "gaussian"))
    xc, yc = classification
    assert np.isfinite(f._calcForwardAICs(xc, yc, LogisticRegression(), "bic", "binomial"))
    data = x.assign(y=y)
    name, path, extension, _ = f._prepare_for_parallel(data, storage)
    score = f._doParallelForward(path, name, extension, "bic", LinearRegression(),
                                 "y", ["a"], False, [], "gaussian", serialize_flavor=storage)
    assert np.isfinite(score)
    selected = f.forwardSelection(data, "y", LinearRegression(), verbose=False,
                                  n_jobs=2 if parallel else 1, return_type="list",
                                  serialize_flavor=storage)
    assert "a" in selected
    limited = f.forwardSelection(data, "y", LinearRegression(), verbose=False,
                                  early_stop=True, stop_at_p=1, return_type="list")
    assert len(limited) == 1
