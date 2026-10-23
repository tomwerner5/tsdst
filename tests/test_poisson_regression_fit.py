"""Check independently fitted regression results saved from R 4.1.0."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from sklearn.linear_model import PoissonRegressor

from tsdst.distributions import posterior_poisson_lasso_od
from tsdst.estimators import BayesPoissonRegressor


def test_warpbreaks_regressions_match_saved_r_fits():
    folder = Path(__file__).parent / "fixtures" / "poisson_warpbreaks"
    data = pd.read_csv(folder / "data.csv")
    reference = json.loads((folder / "reference.json").read_text())
    features = data[["woolB", "tensionM", "tensionH"]].to_numpy()
    response = data.breaks.to_numpy()
    design = np.column_stack([np.ones(len(data)), features])
    quasi = PoissonRegressor(alpha=0, tol=1e-12, max_iter=10000).fit(features, response)
    assert_allclose(quasi.predict(features), data.quasi_prediction, rtol=1e-7)
    assert_allclose(
        np.r_[quasi.intercept_, quasi.coef_], reference["quasi_coefficients"], atol=1e-7
    )

    def objective(parameters):
        with np.errstate(all="ignore"):
            value = posterior_poisson_lasso_od(
                parameters, design, response, l_scale=2000
            )
        return -value if np.isfinite(value) else 1e100

    for evaluation in reference["log_posterior_evaluations"]:
        assert_allclose(
            -objective(np.array(evaluation[:-1])), evaluation[-1], atol=1e-10, rtol=0
        )
    initial = np.r_[quasi.intercept_, quasi.coef_, 0.3]
    mode = minimize(
        objective,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10},
    )
    assert mode.success
    assert_allclose(mode.x, reference["generalized_poisson_mode"], atol=2e-5, rtol=0)

    model = BayesPoissonRegressor(C=reference["C"], over_dispersion=True, verbose=False)
    model.intercept_ = mode.x[0]
    model.coef_ = mode.x[1:-1]
    model.dispersion_delta_ = mode.x[-1]
    model.n_features_in_ = features.shape[1]
    prediction_features = data.drop_duplicates(["woolB", "tensionM", "tensionH"])
    predicted = model.predict(
        prediction_features[["woolB", "tensionM", "tensionH"]].to_numpy()
    )
    assert_allclose(predicted, reference["mode_predictions"], rtol=2e-5)
