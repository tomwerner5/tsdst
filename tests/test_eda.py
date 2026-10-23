"""Test profiling, optional response comparisons, and HTML export."""

import base64

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from tsdst.eda import EDAReport, auto_eda


def test_target_free_statistics_and_no_mutation():
    data = pd.DataFrame({
        "number": [0., 1., 2., 3., 4., 5., 6., 100., np.inf, np.nan],
        "opposite": [0., -1., -2., -3., -4., -5., -6., -100., -np.inf, np.nan],
        "label": ["a", "b"] * 5,
        "constant": [1] * 10,
        "empty": [None] * 10,
    })
    before = data.copy(deep=True)
    report = auto_eda(data, plots=False, corr_threshold=.95)
    assert isinstance(report, EDAReport)
    assert report.response_summary is None
    assert report.response_correlations is None
    assert report.overview["rows"] == 10
    assert report.overview["missing_cells"] == 12
    numeric = report.variables.loc["number"]
    assert numeric["mean"] == pytest.approx(121 / 8)
    assert numeric["finite_count"] == 8
    assert numeric["infinite"] == 1
    assert numeric["outliers_iqr"] == 1
    assert numeric["zeros"] == 1
    assert len(report.high_correlations) == 1
    assert report.high_correlations.iloc[0]["correlation"] == pytest.approx(-1)
    assert set(report.alerts.kind) >= {"missing", "all_missing", "constant", "infinite", "high_correlation"}
    assert report.plots == {}
    assert_frame_equal(data, before)


def test_nullable_categorical_boolean_and_temporal_types():
    data = pd.DataFrame({
        "boolean": pd.Series([True, False, None], dtype="boolean"),
        "integer": pd.Series([1, 2, None], dtype="Int64"),
        "small_integer": pd.Series([1, 2, None], dtype="Int64"),
        "float": pd.Series([1., np.inf, None], dtype="Float64"),
        "category": pd.Series(pd.Categorical(["x", "y", None], categories=["x", "y", "unused"])),
        "string": pd.Series(["x", "y", None], dtype="string"),
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", None], utc=True),
        "duration": pd.to_timedelta([1, 2, None], unit="D"),
    })
    before = data.copy(deep=True)
    report = auto_eda(data, plots=True, n_unique=2,
                      feature_types={"integer": "numeric", "float": "numeric"})
    assert report.variables.at["boolean", "type"] == "categorical"
    assert report.variables.at["small_integer", "type"] == "categorical"
    assert report.variables.at["integer", "type"] == "numeric"
    assert report.variables.at["date", "type"] == "date"
    assert report.variables.at["duration", "type"] == "date"
    assert report.variables.at["float", "mean"] == 1
    # Count the categories that occur in the data. Exclude unused levels.
    assert report.variables.at["category", "unique"] == 2
    assert report.frequencies["boolean"].sum() == 3
    assert all(base64.b64decode(png).startswith(b"\x89PNG\r\n\x1a\n")
               for png in report.plots.values())
    assert_frame_equal(data, before)


@pytest.mark.parametrize("data", [pd.DataFrame(), pd.DataFrame(columns=["x"]),
                                      pd.DataFrame(index=range(3)),
                                      pd.DataFrame({"x": [np.nan] * 3}),
                                      pd.DataFrame({"x": [np.inf, -np.inf]}),
                                      pd.DataFrame({"x": [1]})])
def test_degenerate_data_can_render(data):
    report = auto_eda(data, n_unique=None, plot_pairs=True)
    assert report.overview["rows"] == len(data)
    assert "<!doctype html>" in report.to_html()
    assert report.high_correlations.empty


def test_duplicates_and_missing_fractions():
    report = auto_eda(pd.DataFrame({"a": [1, 1, None, None], "b": ["x"] * 4}), plots=False)
    assert report.overview["duplicate_rows"] == 2
    assert report.overview["duplicate_fraction"] == .5
    assert report.overview["missing_fraction"] == .25
    assert report.variables.at["a", "missing_fraction"] == .5


def test_arbitrary_column_labels_and_signed_unique_pairs():
    data = pd.DataFrame({1: range(10), "1": range(10, 0, -1), "x..&..y": range(10)})
    report = auto_eda(data, plots=False, corr_threshold=1)
    assert len(report.high_correlations) == 3
    assert report.high_correlations.iloc[0]["feature_1"] == 1
    assert report.high_correlations.iloc[0]["feature_2"] == "1"
    assert report.high_correlations.iloc[0]["correlation"] == -1


def test_correlations_match_pandas_and_can_be_disabled():
    data = pd.DataFrame({"x": range(8), "y": [0., 4., np.nan, 3., 2., 8., 7., 1.]})
    report = auto_eda(data, plots=False, corr_method="spearman", n_unique=None)
    assert_frame_equal(report.correlations, data.corr(method="spearman"))
    disabled = auto_eda(data, response=np.arange(8), plots=False, corr_threshold=None)
    assert disabled.correlations.empty
    assert disabled.high_correlations.empty
    assert disabled.response_correlations is None


def test_response_column_is_optional_and_excluded_from_features():
    data = pd.DataFrame({"x": range(10), "target": range(10, 0, -1)})
    before = data.copy(deep=True)
    report = auto_eda(data, response="target", plots=False)
    assert report.overview["columns"] == 1
    assert report.variables.index.tolist() == ["x"]
    assert report.response_summary["type"] == "numeric"
    assert report.response_correlations["x"] == pytest.approx(-1)
    assert_frame_equal(data, before)


@pytest.mark.parametrize("feature_kind,response_kind", [
    ("categorical", "categorical"), ("categorical", "numeric"),
    ("numeric", "categorical"), ("numeric", "numeric")])
def test_all_response_plot_branches_with_duplicate_indexes(feature_kind, response_kind):
    numeric = [0., 1., 2., 3., 4., 5., 6., np.inf, np.nan]
    category = ["a", "a", "b", "b", None, "b", "a", "c", "a"]
    data = pd.DataFrame({"x": numeric if feature_kind == "numeric" else category}, index=[0] * 9)
    response = pd.Series(numeric if response_kind == "numeric" else category,
                         index=data.index, name="target")
    before = response.copy(deep=True)
    report = auto_eda(data, response=response, response_type=response_kind,
                      feature_types={"x": feature_kind}, max_categories=2, max_plot_rows=6)
    assert ("comparison", "x") in report.plots
    assert report.response_summary is not None
    assert_series_equal(response, before)


def test_response_misalignment_and_bad_inputs_are_rejected():
    data = pd.DataFrame({"x": [1, 2]}, index=[10, 20])
    cases = [
        dict(response=pd.Series([1, 2])), dict(response=[1]),
        dict(response=[[1], [2]]), dict(response="missing"),
        dict(feature_types={"missing": "numeric"}),
        dict(feature_types={"x": "date"}), dict(feature_types={"x": "bad"}),
        dict(response_type="numeric"), dict(max_categories=0),
        dict(max_plot_rows=-1), dict(max_pair_columns=1.5),
        dict(n_unique=-1), dict(corr_threshold=2), dict(corr_method="bad"),
        dict(plot_pairs=True, plots=False),
    ]
    for kwargs in cases:
        with pytest.raises(ValueError):
            auto_eda(data, **kwargs)
    with pytest.raises(ValueError, match="unique"):
        auto_eda(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    with pytest.raises(ValueError, match="numeric dtype"):
        auto_eda(pd.DataFrame({"x": ["1", "2"]}), feature_types={"x": "numeric"})
    with pytest.raises(ValueError, match="Complex"):
        auto_eda(pd.DataFrame({"x": [1 + 2j]}))
    with pytest.raises(ValueError, match="scalar, hashable"):
        auto_eda(pd.DataFrame({"x": [[1], [2]]}))


def test_plot_sampling_limits_do_not_change_statistics_or_leak_figures():
    import matplotlib.pyplot as plt

    data = pd.DataFrame({str(i): np.arange(30) * (i + 1) for i in range(4)})
    open_figures = plt.get_fignums()
    report = auto_eda(data, plot_pairs=True, max_pair_columns=2, max_plot_rows=8)
    full = auto_eda(data, plots=False)
    assert_frame_equal(report.variables, full.variables)
    assert_frame_equal(report.correlations, full.correlations)
    assert any("8 of 30" in note for note in report.notes)
    assert any("2 of 4" in note for note in report.notes)
    assert ("dataset", "pairs") in report.plots
    assert plt.get_fignums() == open_figures


def test_html_escapes_input_and_is_self_contained(tmp_path):
    data = pd.DataFrame({"<script>bad()</script>": ["<img onerror='bad'>", "safe"]})
    path = tmp_path / "nested" / "report.html"
    report = auto_eda(data, title="<script>title</script>", output_file=path)
    html = path.read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "<img onerror" not in html
    assert "&lt;script&gt;" in html
    assert "data:image/png;base64," in html
    assert "https://" not in html and "http://" not in html
    assert report.to_file(path) == path
    assert report.to_html() == html
    assert report._repr_html_() == html


def test_array_input_and_no_implicit_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = auto_eda([[1, "a"], [2, "b"]], plots=False)
    assert report.overview["rows"] == 2
    assert list(tmp_path.iterdir()) == []
