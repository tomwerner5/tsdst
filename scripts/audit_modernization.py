"""Compare selected functions from dc0a055 with its parent.

This is an audit, not a release check. Some checks reproduce regressions or
behavior changes. Read their classifications and the accompanying audit report.
The script reads committed source with Git and does not change package code.
"""

import argparse
import ast
import importlib
import io
import json
import random
import subprocess
import warnings
from bisect import bisect_left
from contextlib import redirect_stdout
from copy import deepcopy
from itertools import accumulate
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy import special, stats
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.utils.validation import _check_sample_weight, check_array
from sqlalchemy.engine import URL, make_url

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "dc0a055"
RESULTS = []


def source(revision, module):
    filename = "tsdst/" + module.replace(".", "/") + ".py"
    return subprocess.check_output(
        ["git", "show", revision + ":" + filename], cwd=ROOT, text=True
    )


def function(module, name, version, numpy_aliases=False, **overrides):
    """Load one committed function with explicit dependencies from this environment.

    This does not import the old package as a whole. Unchanged helpers come
    from the working package. Selected checks restore removed NumPy aliases
    in an isolated module object so that the old formulas can run.
    """
    revision = COMMIT + "^" if version == "before" else COMMIT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        node = ast.parse(source(revision, module))
    for part in name.split("."):
        node = next(
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and child.name == part
        )
    namespace = dict(vars(importlib.import_module("tsdst." + module)))
    namespace.update(
        bisect_left=bisect_left,
        accumulate=accumulate,
        check_array=check_array,
        URL=URL,
        gamma=special.gamma,
    )
    namespace.update(overrides)
    if numpy_aliases:
        compatible_numpy = ModuleType("audit_numpy")
        compatible_numpy.__dict__.update(vars(np))
        compatible_numpy.int = int
        compatible_numpy.product = np.prod
        namespace["np"] = compatible_numpy
    tree = ast.Module(body=[node], type_ignores=[])
    exec(compile(tree, f"<{revision}:{module}:{name}>", "exec"), namespace)  # noqa: S102 -- Compiled fixed commit source for comparison.
    return namespace[node.name]


def serializable(value):
    if isinstance(value, pd.DataFrame):
        return {"columns": list(value.columns), "values": value.to_numpy().tolist()}
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def outcome(call):
    try:
        with redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {"value": serializable(call())}
    except Exception as error:  # noqa: BLE001 -- Recorded exceptions as comparison outputs.
        return {"error": type(error).__name__, "message": str(error)}


def compare(
    identifier, module, name, arguments, classification, reference=None, **overrides
):
    before = function(module, name, "before", **overrides)
    after = function(module, name, "after", **overrides)
    old = outcome(lambda: before(*deepcopy(arguments)))
    new = outcome(lambda: after(*deepcopy(arguments)))
    RESULTS.append(
        {
            "id": identifier,
            "function": module + "." + name,
            "classification": classification,
            "before": old,
            "after": new,
            "reference": serializable(reference),
        }
    )
    return old, new


def run():
    from tsdst import estimators
    from tsdst.nn.model import NeuralNetwork

    truth = np.array([0, 0, 0, 1, 1])
    predicted = np.array([0, 1, 1, 0, 1])
    _, result = compare(
        "confusion",
        "metrics",
        "conf_mat_metrics",
        (truth, predicted),
        "bug",
        reference={"recall": 0.5, "specificity": 1 / 3, "precision": 1 / 3},
    )
    assert_allclose(result["value"]["Sens/Recall"], 0.5)

    classifier = estimators.LogReg()
    classifier.coef_, classifier.intercept_ = np.array([[1.0]]), np.array([0.0])
    classifier.classes_, classifier.n_features_in_ = np.array([0, 1]), 1
    for name in ("LogReg", "BayesLogRegClassifier"):
        _, result = compare(
            "probability-" + name,
            "estimators",
            name + ".predict_proba",
            (classifier, np.array([[1.0]])),
            "bug",
            reference=special.expit(1.0),
            softmax=lambda values, copy=False: special.softmax(values, axis=1),
        )
        assert_allclose(result["value"][0][1], special.expit(1.0))
        _, result = compare(
            "tie-" + name,
            "estimators",
            name + ".predict",
            (classifier, np.array([[0.0]])),
            "behavior choice",
            numpy_aliases=True,
        )
        assert result["value"] == [1]

    x = np.column_stack([np.ones(4), np.arange(4)])
    y, parameters = np.arange(1.0, 5.0), np.array([0.1, 0.2, np.log(1.3)])
    shape, scale = np.exp(x @ parameters[:-1]), np.exp(parameters[-1])
    prior = (
        stats.gamma.logpdf(scale, a=0.01, scale=100)
        + stats.norm.logpdf(parameters[0], scale=10)
        + stats.laplace.logpdf(parameters[1:-1]).sum()
        + parameters[-1]
    )
    for status in ([1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]):
        status = np.asarray(status)
        failed = status == 1
        expected = (
            stats.weibull_min.logpdf(y[failed], c=shape[failed], scale=scale).sum()
            + stats.weibull_min.logsf(y[~failed], c=shape[~failed], scale=scale).sum()
            + prior
        )
        _, result = compare(
            "weibull-censor-" + "".join(map(str, status)),
            "distributions",
            "weibull_regression_post",
            (parameters, x, y, status),
            "bug",
            expected,
        )
        assert_allclose(result["value"], expected, rtol=1e-13)

    expected = [1 / 3, 1 / (3 * np.sqrt(2))]
    _, result = compare(
        "exponential-mle",
        "distributions",
        "exponential_mle",
        (np.array([1.0, 2.0, 3.0]), np.array([1, 1, 0])),
        "bug",
        expected,
    )
    assert_allclose(result["value"][:2], expected)
    _, result = compare(
        "hotelling",
        "metrics",
        "HotellingsTwoSampleMVTtest",
        (np.ones(2), np.zeros(2), np.eye(2), np.eye(2), 10, 10),
        "bug",
        stats.f.sf(17 / 36 * 10, 2, 17),
    )
    assert_allclose(result["value"], stats.f.sf(17 / 36 * 10, 2, 17))
    _, result = compare(
        "scaled-sigmoid",
        "nn.activations",
        "sigmoid_scale",
        (np.array([0.3]), 0.1, 2.0),
        "bug",
        special.expit(0.1),
    )
    assert_allclose(result["value"], [special.expit(0.1)])
    _, result = compare(
        "decimal-beta", "modeling", "beta_trans", (np.log(2), "decimal"), "bug", 1.0
    )
    assert_allclose(result["value"], 1.0)
    compare(
        "zero-norm",
        "tmath",
        "norm",
        (np.array([-3.0, 0.0, 2.0]), 0),
        "restored old intent; not standard L0",
        reference=np.linalg.norm([-3.0, 0.0, 2.0], ord=0),
    )

    chain = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    _, result = compare(
        "spectral-zero-lag", "mcmc", "msvec", (chain, 1), "bug", [[1.0]]
    )
    assert_allclose(result["value"], [[1.0]])
    centered = np.array(
        [[-2.0, 1.0], [-1.0, -1.0], [0.0, 2.0], [1.0, -2.0], [2.0, 0.0]]
    )
    expected = centered.T @ centered / len(centered)
    for lag in (1, 2):
        cross = centered[:-lag].T @ centered[lag:] / len(centered)
        expected += (1 - lag / 3) * (cross + cross.T)
    _, result = compare(
        "spectral-lags", "mcmc", "msvec", (centered, 3), "bug", expected
    )
    assert_allclose(result["value"], expected)
    compare(
        "mcse-construction",
        "mcmc",
        "mcse_multi",
        (np.random.default_rng(8).normal(size=(100, 2)),),
        "bug",
    )

    features = np.arange(6.0, dtype=float)[:, None]
    response = np.array([0.2, 1.4, 1.8, 3.3, 4.2, 4.9])
    for penalty in (0.0, 2.0):
        model = (
            LinearRegression().fit(features, response)
            if penalty == 0
            else Ridge(alpha=penalty).fit(features, response)
        )
        old, new = compare(
            "covariance-" + str(penalty),
            "modeling",
            "GenerateRegCov",
            (features, response, model, False, False, penalty),
            "equivalent ordinary least squares"
            if penalty == 0
            else "intercept penalty choice",
        )
        if penalty == 0:
            assert_allclose(old["value"][0], new["value"][0])
        else:
            assert not np.allclose(old["value"][0], new["value"][0])

    weibull = estimators.BayesWeibullRegressor()
    weibull.coef_, weibull.intercept_ = np.array([0.0]), np.log(2.0)
    weibull.scale_, weibull.n_features_in_ = 3.0, 1
    old, result = compare(
        "weibull-prediction",
        "estimators",
        "BayesWeibullRegressor.predict",
        (weibull, np.zeros((1, 1))),
        "prediction meaning choice",
        stats.weibull_min.mean(c=2.0, scale=3.0),
    )
    assert_allclose(result["value"], [stats.weibull_min.mean(c=2.0, scale=3.0)])
    assert_allclose(old["value"], [2.0])

    poisson = estimators.BayesPoissonRegressor()
    poisson.coef_, poisson.intercept_ = np.array([1.0]), 0.0
    poisson.dispersion_delta_, poisson.n_features_in_ = 0.0, 1
    old, new = compare(
        "sparse-poisson-prediction",
        "estimators",
        "BayesPoissonRegressor.predict",
        (poisson, csr_matrix([[1.0]])),
        "regression",
    )
    assert_allclose(old["value"], [np.e])
    assert new["error"] == "TypeError"
    old, new = compare(
        "nonfinite-sample-weights",
        "estimators",
        "BayesPoissonRegressor._deviance_dispersion_update",
        (
            poisson,
            np.arange(4.0)[:, None],
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, np.nan, 1.0, 1.0]),
        ),
        "validation regression",
        _check_sample_weight=_check_sample_weight,
    )
    assert old["error"] == "ValueError"
    assert new["value"]["deviance_"] == "nan"

    model = LogisticRegression().fit(features, [0, 0, 1, 0, 1, 1])
    old, new = compare(
        "legacy-output-keys",
        "modeling",
        "scoreModel_legacy",
        (
            pd.DataFrame(features),
            pd.Series([0, 0, 1, 0, 1, 1]),
            model,
            None,
            "classification",
            False,
        ),
        "output contract loss",
    )
    assert "McFadden R2" in old["value"] and "McFadden R2" not in new["value"]

    for version in ("before", "after"):
        captured = []

        def engine_factory(url, captured=captured, **arguments):
            captured.append(make_url(url))
            return type("AuditEngine", (), {"dispose": lambda self: None})()

        fake_pandas = type(
            "AuditPandas",
            (),
            {"read_sql": staticmethod(lambda *args, **kwargs: "table")},
        )
        reader = function(
            "utils",
            "dsn_getTable",
            version,
            create_engine=engine_factory,
            pd=fake_pandas,
        )
        reader(
            "SELECT 1",
            "example",
            "postgresql",
            "localhost:5432/example?sslmode=require",
            pw="placeholder",
            create_engine_args={},
            read_sql_args={},
        )
        url = captured[0]
        RESULTS.append(
            {
                "id": "database-url-" + version,
                "classification": "regression",
                "value": {
                    "host": url.host,
                    "port": url.port,
                    "database": url.database,
                    "query": dict(url.query),
                },
            }
        )
    assert RESULTS[-2]["value"]["port"] == 5432 and RESULTS[-1]["value"]["port"] is None

    compare(
        "missing-values-type",
        "utils",
        "inferFeatureType",
        (pd.Series([1.0, 1.0, np.nan]), 1),
        "category count choice",
    )
    compare(
        "boolean-type",
        "utils",
        "inferFeatureType",
        (pd.Series([True, False]),),
        "equivalent",
    )
    compare(
        "string-column",
        "utils",
        "move_columns_to_end",
        (pd.DataFrame({"first": [1], "second": [2]}), "first"),
        "bug",
    )
    for version in ("before", "after"):
        shared = {"a": 10}
        worker = function("parallel", "p_prog_simp", version)
        result = outcome(
            lambda worker=worker, shared=shared: worker(
                shared,
                [{"b": 1}, {"b": 3}],
                lambda a, b: a + b,
                n_jobs=2,
                verbose=False,
            )
        )
        RESULTS.append(
            {
                "id": "parallel-input-" + version,
                "classification": "side effect fix",
                "result": result,
                "shared_arguments": shared,
            }
        )
    for version in ("before", "after"):
        state = random.getstate()
        weights = [0.5, 0.3, 0.2]
        sampler = function("utils", "sample", version)
        result = sampler(["a", "b", "c"], 3, weights, False, 11)
        RESULTS.append(
            {
                "id": "sampling-state-" + version,
                "classification": "side effect and RNG policy",
                "draws": result,
                "weights": weights,
                "global_random_state_changed": random.getstate() != state,
            }
        )
        random.setstate(state)

    spec = {
        "output": dict(
            depth=1,
            initializer="xavier_uniform",
            activation="sigmoid",
            activation_args={},
            derivative="sigmoid_cross_entropy_binary_der",
            derivative_args={},
            use_batch_norm=False,
            **{
                "lambda": {"Weight": 0, "activity": 0, "bias": 0},
                "lp_norm": {"Weight": 2, "activity": 2, "bias": 2},
            },
            cost="cross_entropy_binary",
            cost_args={},
        )
    }
    network = NeuralNetwork(spec, print_cost=False)
    features = np.array([[-1.0], [0.2], [1.0]])
    response = np.array([[0.0], [1.0], [1.0]])
    parameters = {"Weight0": np.array([[0.4]]), "bias0": np.array([[0.1]])}
    _, zs, batch_norm, activations, dropout, _ = network.forward_prop(
        features, parameters, {}
    )
    expected = {}
    for key in parameters:
        plus, minus = deepcopy(parameters), deepcopy(parameters)
        plus[key][0, 0] += 1e-5
        minus[key][0, 0] -= 1e-5
        from tsdst.nn.activations import cross_entropy_binary

        loss_plus = cross_entropy_binary(
            response, network.forward_prop(features, plus, {})[0]
        ) / len(features)
        loss_minus = cross_entropy_binary(
            response, network.forward_prop(features, minus, {})[0]
        ) / len(features)
        expected[key] = [[(loss_plus - loss_minus) / 2e-5]]
    _, result = compare(
        "neural-gradient",
        "nn.model",
        "NeuralNetwork.back_prop",
        (network, features, response, parameters, zs, batch_norm, activations, dropout),
        "bug",
        expected,
    )
    for key in expected:
        assert_allclose(result["value"][key], expected[key], rtol=1e-7)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = np.arange(12.0, dtype=float).reshape(6, 2)
        labels = [0, 0, 1, 0, 1, 1]
        old = LogisticRegression(penalty="l1", solver="liblinear", random_state=7).fit(
            features, labels
        )
        new = LogisticRegression(l1_ratio=1, solver="liblinear", random_state=7).fit(
            features, labels
        )
    assert_allclose(old.coef_, new.coef_)
    assert_allclose(old.intercept_, new.intercept_)
    RESULTS.append(
        {
            "id": "sklearn-l1-option",
            "classification": "equivalent current API",
            "before": old.coef_.tolist(),
            "after": new.coef_.tolist(),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run()
    report = {
        "commit": COMMIT,
        "parent": subprocess.check_output(
            ["git", "rev-parse", COMMIT + "^"], cwd=ROOT, text=True
        ).strip(),
        "scope": "Isolated committed function bodies with current dependencies; not full old-environment equivalence.",
        "checks": RESULTS,
    }
    arguments.output.write_text(
        json.dumps(serializable(report), indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(RESULTS)} audit cases in {arguments.output}.")
    print(
        "Cases include known regressions and behavior choices; this is not a release approval."
    )


if __name__ == "__main__":
    main()
