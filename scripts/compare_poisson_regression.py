"""Compare an actual tsdst fit with the independent R warpbreaks fits.

First run compare_poisson_regression.R and save its stdout. Then run this
script with --r-reference pointing to that file and --output to a folder.
The comparison uses the existing fixed-C posterior and built-in rwm sampler.
"""

import argparse
import json
import random
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from sklearn.linear_model import PoissonRegressor

from tsdst.distributions import posterior_poisson_lasso_od
from tsdst.estimators import BayesPoissonRegressor


def read_r_reference(path):
    """Read named CSV sections without an R package dependency."""
    parts = re.split(r"@@([A-Z]+)@@\s*\n", path.read_text())
    return {
        parts[index]: pd.read_csv(StringIO(parts[index + 1]))
        for index in range(1, len(parts), 2)
    }


def summarize(chains, names):
    """Calculate batch-means MCSE and ordinary split R-hat independently."""
    pooled = np.concatenate(chains)
    size = len(chains[0])
    batch_size = int(np.sqrt(size))
    batches = [
        chain[: size // batch_size * batch_size]
        .reshape(-1, batch_size, chain.shape[1])
        .mean(axis=1)
        for chain in chains
    ]
    batch_variance = np.mean([batch.var(axis=0, ddof=1) for batch in batches], axis=0)
    mcse = np.sqrt(batch_size * batch_variance / len(pooled))
    half_size = size // 2
    halves = [
        half
        for chain in chains
        for half in (chain[:half_size], chain[half_size : 2 * half_size])
    ]
    within = np.mean([half.var(axis=0, ddof=1) for half in halves], axis=0)
    between = np.var([half.mean(axis=0) for half in halves], axis=0, ddof=1)
    rhat = np.sqrt(((half_size - 1) / half_size * within + between) / within)
    sd = pooled.std(axis=0, ddof=1)
    return pd.DataFrame(
        {
            "name": names,
            "mean": pooled.mean(axis=0),
            "sd": sd,
            "mcse": mcse,
            "rhat": rhat,
            "ess": sd**2 / mcse**2,
            "lower": np.quantile(pooled, 0.025, axis=0),
            "upper": np.quantile(pooled, 0.975, axis=0),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100000)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    reference = read_r_reference(options.r_reference)
    data = reference["DATA"]
    feature_names = ["woolB", "tensionM", "tensionH"]
    features = data[feature_names].to_numpy()
    design = np.column_stack([np.ones(len(data)), features])
    response = data.breaks.to_numpy()
    dimension = design.shape[1]
    burn = options.iterations // 5

    def objective(parameters):
        with np.errstate(all="ignore"):
            value = posterior_poisson_lasso_od(
                parameters, design, response, l_scale=2000
            )
        return -value if np.isfinite(value) else 1e100

    for row in reference["EVALUATIONS"].to_numpy():
        assert_allclose(-objective(row[:-1]), row[-1], atol=1e-10, rtol=0)
    ordinary = PoissonRegressor(alpha=0, tol=1e-12, max_iter=10000).fit(
        features, response
    )
    assert_allclose(ordinary.predict(features), data.quasi_prediction, rtol=1e-7)
    initial = np.r_[ordinary.intercept_, ordinary.coef_, 0.3]
    mode = minimize(
        objective,
        initial,
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10},
    )
    if not mode.success:
        raise AssertionError(mode.message)
    r_mode = reference["PARAMETERS"]["mode"].to_numpy()
    assert_allclose(mode.x, r_mode, atol=2e-5, rtol=0)
    proposal = reference["PROPOSAL"].to_numpy()
    prediction_rows = data.drop_duplicates(feature_names)
    prediction_design = np.column_stack(
        [np.ones(len(prediction_rows)), prediction_rows[feature_names]]
    )
    names = reference["PARAMETERS"]["name"].tolist()
    prediction_names = reference["PREDICTIONS"]["name"].tolist()
    chains = []
    fit_rows = []
    for seed in (505, 606, 707, 808):
        np.random.seed(seed)
        random.seed(seed)
        start = mode.x + np.linalg.cholesky(proposal) @ np.random.normal(
            size=dimension + 1
        )
        model = BayesPoissonRegressor(
            C=1000,
            over_dispersion=True,
            niter=options.iterations,
            start=start,
            algo="rwm",
            algo_options={"progress": False, "prev_vals": {"E_0": proposal}},
            verbose=False,
        ).fit(features, response)
        chain = model.mcmc_params[burn:].copy()
        chains.append(chain)
        actual_prediction = model.predict(prediction_rows[feature_names].to_numpy())
        candidate = np.exp(prediction_design @ np.r_[model.intercept_, model.coef_])
        assert_allclose(actual_prediction, candidate, atol=1e-12)
        for name, actual, proposed in zip(
            prediction_names, actual_prediction, candidate
        ):
            fit_rows.append(
                {
                    "seed": seed,
                    "name": name,
                    "actual_predict": actual,
                    "candidate_predict": proposed,
                    "delta": model.dispersion_delta_,
                }
            )
        np.save(options.output / f"python-chain-{seed}.npy", chain)

    parameter_summary = summarize(chains, names)
    prediction_chains = [
        np.exp(chain[:, :dimension] @ prediction_design.T) for chain in chains
    ]
    prediction_summary = summarize(prediction_chains, prediction_names)
    pooled_mean = np.concatenate(chains).mean(axis=0)
    prediction_summary["plugin"] = np.exp(prediction_design @ pooled_mean[:dimension])
    prediction_summary["old_plugin"] = prediction_summary.plugin * (1 - pooled_mean[-1])

    comparisons = {}
    for name, summary in (
        ("parameters", parameter_summary),
        ("predictions", prediction_summary),
    ):
        comparison = summary.merge(
            reference[name.upper()], on="name", suffixes=("_python", "_r")
        )
        comparison["combined_mcse"] = np.hypot(
            comparison.mcse_python, comparison.mcse_r
        )
        comparison["difference_in_mcse"] = (
            abs(comparison.mean_python - comparison.mean_r) / comparison.combined_mcse
        )
        if not np.all(comparison.difference_in_mcse < 4):
            raise AssertionError(
                f"R and Python {name} disagree beyond four combined MCSE:\n{comparison}"
            )
        for language in ("python", "r"):
            if not np.all(comparison[f"rhat_{language}"] < 1.01):
                raise AssertionError(f"{language} {name} split R-hat exceeded 1.01")
            if not np.all(comparison[f"ess_{language}"] > 1000):
                raise AssertionError(
                    f"{language} {name} batch-means ESS was below 1000"
                )
        comparison.to_csv(options.output / f"{name}-comparison.csv", index=False)
        comparisons[name] = comparison

    pd.DataFrame(fit_rows).to_csv(
        options.output / "estimator-predictions.csv", index=False
    )
    for name, table in reference.items():
        table.to_csv(options.output / f"r-{name.lower()}.csv", index=False)
    predictions = comparisons["predictions"]
    report = {
        "iterations_per_chain": options.iterations,
        "burn_per_chain": burn,
        "chains_per_language": 4,
        "estimator_predict_matches_conditional_mean": True,
        "r_metadata": reference["METADATA"].iloc[0].to_dict(),
        "max_mode_difference": float(np.max(abs(mode.x - r_mode))),
        "max_parameter_difference_in_combined_mcse": float(
            comparisons["parameters"].difference_in_mcse.max()
        ),
        "max_prediction_difference_in_combined_mcse": float(
            predictions.difference_in_mcse.max()
        ),
        "python_negative_delta_fraction": float(
            np.mean(np.concatenate(chains)[:, -1] < 0)
        ),
        "quasi_rmse_old_plugin": float(
            np.sqrt(np.mean((predictions.old_plugin_python - predictions.quasi) ** 2))
        ),
        "quasi_rmse_candidate_plugin": float(
            np.sqrt(np.mean((predictions.plugin_python - predictions.quasi) ** 2))
        ),
        "r_bayesian_rmse_old_plugin": float(
            np.sqrt(
                np.mean((predictions.old_plugin_python - predictions.plugin_r) ** 2)
            )
        ),
        "r_bayesian_rmse_candidate_plugin": float(
            np.sqrt(np.mean((predictions.plugin_python - predictions.plugin_r) ** 2))
        ),
    }
    (options.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(
        predictions[
            ["name", "quasi", "plugin_r", "plugin_python", "old_plugin_python"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
