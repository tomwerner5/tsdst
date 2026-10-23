"""Explore a dataset without a response variable.

This module builds on ``eda`` in the Kaggle
``Common_Data_Science_Functions.py`` file. It uses the same type rules,
correlation checks, and plot choices. The analysis keeps the input values
and features unchanged. HTML reports include their styles and plot images.
"""

import base64
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_complex_dtype, is_datetime64_any_dtype,
    is_numeric_dtype, is_timedelta64_dtype,
)

from .utils import inferFeatureType

__all__ = ["auto_eda", "EDAReport"]


class EDAReport:
    """Store the tables and PNG plots from :func:`auto_eda`.

    Attributes
    ----------
    overview : dict
        Row and column counts, missing values, duplicate rows, and memory use.
    variables : pandas.DataFrame
        One row per feature. Numeric statistics use finite values only.
        Missing values and infinities have separate counts. Missing fractions
        use the full row count. Unique counts exclude missing values.
        Unique fractions use the count of nonmissing values.
    frequencies : dict
        Full value counts for categorical features, including missing values.
    correlations : pandas.DataFrame
        Correlations between numeric features. Each pair uses rows with
        valid values for both features.
    high_correlations : pandas.DataFrame
        One row per pair that meets the absolute threshold.
        Correlations keep their signs.
    alerts : pandas.DataFrame
        Checks for missing values, constants, unique values, infinities,
        duplicate rows, and correlated features. Use these checks to review
        the data. The checks do not remove features.
    plots : dict
        PNG images encoded as base64 text. Keys are
        ``(section, variable)`` tuples.
    response_summary : pandas.Series or None
        A separate profile of the optional response.
    response_correlations : pandas.Series or None
        Correlations with numeric features, if the response is numeric.
    notes : list of str
        Row sampling and limits on the plots.
    """

    def __init__(self, title, overview, variables, frequencies, correlations,
                 high_correlations, alerts, plots, response_summary,
                 response_correlations, notes, max_categories):
        self.title = title
        self.overview = overview
        self.variables = variables
        self.frequencies = frequencies
        self.correlations = correlations
        self.high_correlations = high_correlations
        self.alerts = alerts
        self.plots = plots
        self.response_summary = response_summary
        self.response_correlations = response_correlations
        self.notes = notes
        self.max_categories = max_categories

    def to_html(self, path=None):
        """Return an HTML report with its styles and plot images.

        If you supply a path, write the report as a UTF-8 file.
        Create missing parent directories. Replace an existing file at that
        path. Escape dataset labels and values for HTML.
        """
        parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
                 "<meta name='viewport' content='width=device-width, initial-scale=1'>",
                 "<title>{}</title>".format(escape(str(self.title))),
                 "<style>body{font:15px system-ui,sans-serif;margin:2rem auto;"
                 "max-width:1200px;padding:0 1rem;color:#243447}"
                 "table{border-collapse:collapse;font-size:.85rem}"
                 "th,td{padding:.4rem .6rem;border:1px solid #d8dee5;text-align:right}"
                 "th{background:#eef3f7}img{max-width:100%;height:auto}"
                 ".table{overflow:auto;margin-bottom:1.5rem}"
                 "h2{margin-top:2rem}figure{margin:1rem 0}</style></head><body>",
                 "<h1>{}</h1>".format(escape(str(self.title)))]

        def table(title, data):
            parts.append("<h2>{}</h2><div class='table'>{}</div>".format(
                escape(title), data.to_html(escape=True, border=0)))

        table("Overview", pd.Series(self.overview, name="value").to_frame())
        for note in self.notes:
            parts.append("<p>{}</p>".format(escape(note)))
        table("Variables", self.variables)
        table("Alerts", self.alerts)
        table("Numeric correlations", self.correlations)
        table("High correlations", self.high_correlations)
        if self.response_summary is not None:
            table("Response profile", self.response_summary.to_frame("value"))
        if self.response_correlations is not None:
            table("Correlations with response",
                  self.response_correlations.to_frame("correlation"))
        for column, counts in self.frequencies.items():
            table("Frequencies: {} (first {} values)".format(
                column, self.max_categories),
                counts.head(self.max_categories).to_frame("count"))
        for key, png in self.plots.items():
            label = escape("{}: {}".format(*key), quote=True)
            parts.append("<figure><figcaption>{}</figcaption>"
                         "<img alt='{}' src='data:image/png;base64,{}'>"
                         "</figure>".format(label, label, png))
        parts.append("</body></html>")
        html = "\n".join(parts)
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(html, encoding="utf-8")
        return html

    def to_file(self, path):
        """Write an HTML report and return its path."""
        self.to_html(path)
        return Path(path)

    def _repr_html_(self):
        return self.to_html()


def _feature_type(series, n_unique):
    # Use the existing type rules. Treat booleans as categories.
    # NumPy gives boolean dtypes the kind '?'.
    if is_bool_dtype(series.dtype):
        return "categorical"
    if is_complex_dtype(series.dtype):
        raise ValueError("Complex-valued columns are not supported: {!r}".format(series.name))
    return inferFeatureType(series.dropna(), n_unique=n_unique)


def _finite(series):
    values = series.dropna().astype(float)
    return values[np.isfinite(values)]


def _profile(series, kind):
    size = len(series)
    present = series.dropna()
    counts = series.value_counts(dropna=True)
    counts = counts[counts > 0]
    for value in counts.index:
        hash(value)
    unique = len(counts)
    row = dict(dtype=str(series.dtype), type=kind, count=int(series.count()),
               missing=int(series.isna().sum()),
               missing_fraction=float(series.isna().mean()) if size else 0.0,
               unique=unique, unique_fraction=unique / len(present) if len(present) else 0.0,
               top=counts.index[0] if unique else None,
               top_count=int(counts.iloc[0]) if unique else 0)
    if kind == "numeric":
        finite = _finite(series)
        quantiles = finite.quantile([0.25, 0.5, 0.75])
        q1, median, q3 = quantiles.tolist()
        iqr = q3 - q1
        row.update(finite_count=len(finite),
                   infinite=int(np.isinf(present.astype(float)).sum()),
                   zeros=int((finite == 0).sum()), negative=int((finite < 0).sum()),
                   mean=finite.mean(), std=finite.std(), min=finite.min(),
                   q1=q1, median=median, q3=q3, max=finite.max(),
                   skew=finite.skew(), kurtosis=finite.kurt(),
                   outliers_iqr=int(((finite < q1 - 1.5 * iqr) |
                                     (finite > q3 + 1.5 * iqr)).sum()))
    elif kind == "date":
        row.update(min=present.min(), max=present.max())
    return row


def _high_correlations(matrix, threshold):
    # Check each column pair once, as getHighCorrs does.
    # Compare absolute correlations and keep their signs.
    # Store labels in separate columns to keep their original values.
    pairs = []
    for i, left in enumerate(matrix.columns):
        for j in range(i + 1, len(matrix.columns)):
            value = matrix.iloc[i, j]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((left, matrix.columns[j], value))
    result = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "correlation"])
    if len(result):
        result = result.iloc[np.argsort(-result.correlation.abs().to_numpy(),
                                       kind="stable")].reset_index(drop=True)
    return result


def _new_figure(width=8, height=4):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    return figure


def _encode_plot(figure):
    try:
        figure.tight_layout()
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=110)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    finally:
        figure.clear()


def _category_label(value):
    return "(missing)" if pd.isna(value) else str(value)


def _distribution(series, kind, max_categories):
    figure = _new_figure()
    ax = figure.subplots()
    ax.set_title(str(series.name))
    if kind == "numeric":
        values = _finite(series)
        if len(values):
            ax.hist(values, bins=30, color="#387eae", edgecolor="white")
        else:
            ax.text(.5, .5, "No finite values", ha="center", transform=ax.transAxes)
        ax.set_xlabel(str(series.name))
    elif kind == "date":
        values = series.dropna()
        if len(values):
            if is_timedelta64_dtype(series.dtype):
                ax.hist(values.dt.total_seconds(), bins=30)
                ax.set_xlabel("Duration (seconds)")
            else:
                ax.hist(values.array.to_pydatetime(), bins=30)
                figure.autofmt_xdate()
        else:
            ax.text(.5, .5, "No dates", ha="center", transform=ax.transAxes)
    else:
        counts = series.value_counts(dropna=False)
        shown = counts.head(max_categories)
        labels = [_category_label(value) for value in shown.index]
        heights = shown.tolist()
        if len(counts) > max_categories:
            labels.append("(other values)")
            heights.append(int(counts.iloc[max_categories:].sum()))
        ax.bar(range(len(heights)), heights, color="#387eae")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    return _encode_plot(figure)


def _comparison(feature, response, feature_type, response_type, max_categories):
    """Choose a bar, box, or scatter plot from the two variable types."""
    if "date" in (feature_type, response_type):
        return None
    figure = _new_figure()
    ax = figure.subplots()
    if feature_type == response_type == "numeric":
        x = feature.astype(float).to_numpy()
        y = response.astype(float).to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=12, alpha=.5)
        ax.set_xlabel(str(feature.name))
        ax.set_ylabel(str(response.name))
    elif feature_type == response_type == "categorical":
        # Match the variables by row position to support duplicate indexes.
        # Give missing categories their own codes.
        x_codes, x_values = pd.factorize(feature)
        y_codes, y_values = pd.factorize(response)
        table = pd.crosstab(x_codes, y_codes)
        rows = table.sum(axis=1).sort_values(ascending=False).head(max_categories).index
        cols = table.sum(axis=0).sort_values(ascending=False).head(max_categories).index
        table = table.loc[rows, cols]
        width = .8 / max(len(cols), 1)
        slots = np.arange(len(rows))
        for i, code in enumerate(cols):
            label = "(missing)" if code == -1 else str(y_values[code])
            ax.bar(slots + (i - (len(cols) - 1) / 2) * width,
                   table[code], width=width, label=label)
        ax.set_xticks(slots)
        ax.set_xticklabels(["(missing)" if code == -1 else str(x_values[code])
                            for code in rows], rotation=45, ha="right")
        if len(cols):
            ax.legend(title=str(response.name))
        ax.set_xlabel(str(feature.name))
        ax.set_ylabel("Count")
    else:
        numeric, categorical = ((feature, response) if feature_type == "numeric"
                                else (response, feature))
        codes, values = pd.factorize(categorical)
        common = pd.Series(codes).value_counts().head(max_categories).index
        groups, labels = [], []
        for code in common:
            group = _finite(numeric.iloc[np.flatnonzero(codes == code)])
            if len(group):
                groups.append(group.to_numpy())
                labels.append("(missing)" if code == -1 else str(values[code]))
        if groups:
            ax.boxplot(groups, notch=True)
            ax.set_xticks(np.arange(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel(str(categorical.name))
        ax.set_ylabel(str(numeric.name))
    ax.set_title("{} by {}".format(feature.name, response.name))
    return _encode_plot(figure)


def _pair_plot(numeric):
    count = len(numeric.columns)
    figure = _new_figure(max(6, count * 2), max(6, count * 2))
    axes = figure.subplots(count, count, squeeze=False)
    for i, left in enumerate(numeric.columns):
        for j, right in enumerate(numeric.columns):
            ax = axes[i, j]
            if i == j:
                ax.hist(numeric[left].dropna(), bins=20)
            else:
                ax.scatter(numeric[right], numeric[left], s=5, alpha=.4)
            if i == count - 1:
                ax.set_xlabel(str(right))
            if j == 0:
                ax.set_ylabel(str(left))
    return _encode_plot(figure)


def auto_eda(features, response=None, *, title="Exploratory data analysis",
             n_unique=5, feature_types=None, response_type=None,
             corr_threshold=0.3, corr_method="pearson", plots=True,
             plot_pairs=False, max_categories=20, max_plot_rows=5000,
             max_pair_columns=8, random_state=0, output_file=None):
    """Explore tabular data and return an EDA report.

    Parameters
    ----------
    features : pandas.DataFrame or two-dimensional array-like
        Data to explore. Each column must have a unique label.
        Each cell must contain one scalar, hashable value.
        The analysis keeps the input unchanged.
    response : str, pandas.Series, or one-dimensional array-like, optional
        A column name or a separate response. If you supply a column name,
        exclude that column from the features. A Series must have the same
        index as the features. Arrays match by row position.
        The response is optional. This function does not fit a model.
    title : str
        Report title.
    n_unique : int or None
        Treat numeric columns as categorical if their unique count is at
        most this value. Exclude missing values from the count.
        Set None to disable this rule. Booleans are categorical.
    feature_types : dict, optional
        Set a column type to 'numeric', 'categorical', or 'date'.
        Numeric and date types require a compatible dtype.
        Convert strings to numbers or dates before you call this function.
    response_type : str, optional
        Set the response type with the same rules as feature_types.
    corr_threshold : float or None
        Absolute threshold from 0 to 1, inclusive. Set None to skip
        correlations. Use numeric features only. Exclude infinite values.
    corr_method : {'pearson', 'spearman', 'kendall'}
        Correlation method. Each pair uses rows with valid values for
        both variables.
    plots : bool
        Create plots for distributions, missing values, and correlations.
        If you supply a response, add comparison plots.
        Set False to compute tables only.
    plot_pairs : bool
        Include a scatter matrix for numeric features. Set plots=True
        to use this option.
    max_categories : int
        Maximum number of categories to show. Distribution bars combine
        the other categories. Response comparisons show the most frequent
        categories only.
    max_plot_rows : int or None
        Row limit for scatter matrices and response comparisons.
        Statistics, frequencies, and distribution plots use all rows.
        Set None to use all rows in every plot.
        Use random_state to repeat the same row sample.
    max_pair_columns : int
        Maximum number of numeric columns in the scatter matrix.
        Select columns in input order. Correlations use all numeric features.
    random_state : int
        Seed for the row sample in the plots.
    output_file : str or pathlib.Path, optional
        Path for the HTML report. If you omit this path, return the report
        without writing a file.

    Returns
    -------
    EDAReport
        Summary tables, alerts, and plots. Export the report to HTML or
        display it in a notebook.
    """
    frame = features.copy() if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
    if not frame.columns.is_unique:
        raise ValueError("Feature column labels must be unique")
    if n_unique is not None and (isinstance(n_unique, bool) or
                                not isinstance(n_unique, (int, np.integer)) or n_unique < 0):
        raise ValueError("n_unique must be a nonnegative integer or None")
    for name, value in (("max_categories", max_categories),
                        ("max_pair_columns", max_pair_columns),
                        ("max_plot_rows", max_plot_rows)):
        if name == "max_plot_rows" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("{} must be a positive integer".format(name))
    if corr_threshold is not None and not 0 <= corr_threshold <= 1:
        raise ValueError("corr_threshold must be between 0 and 1, or None")
    if corr_method not in ("pearson", "spearman", "kendall"):
        raise ValueError("corr_method must be pearson, spearman, or kendall")
    if plot_pairs and not plots:
        raise ValueError("plot_pairs requires plots=True")
    if isinstance(response, str):
        if response not in frame.columns:
            raise ValueError("Response column not found: {!r}".format(response))
        response = frame.pop(response)
    elif response is not None:
        if isinstance(response, pd.Series):
            if not response.index.equals(frame.index):
                raise ValueError("Response Series index must exactly match the feature index")
            response = response.copy()
        else:
            values = np.asarray(response)
            if values.ndim != 1 or len(values) != len(frame):
                raise ValueError("Response must be one-dimensional with one value per row")
            response = pd.Series(values, index=frame.index, name="response")
    if response is not None and response.name is None:
        response = response.rename("response")
    if response is None and response_type is not None:
        raise ValueError("response_type requires a response")
    overrides = {} if feature_types is None else dict(feature_types)
    unknown = [column for column in overrides if column not in frame.columns]
    if unknown:
        raise ValueError("Unknown feature_types columns: {!r}".format(unknown))

    def kind_for(series, override=None):
        inferred = _feature_type(series, n_unique)
        kind = inferred if override is None else override
        if kind not in ("numeric", "categorical", "date"):
            raise ValueError("Feature types must be numeric, categorical, or date")
        if kind == "numeric" and not is_numeric_dtype(series.dtype):
            raise ValueError("Numeric type requires a numeric dtype: {!r}".format(series.name))
        if kind == "date" and not (is_datetime64_any_dtype(series.dtype) or
                                    is_timedelta64_dtype(series.dtype)):
            raise ValueError("Date type requires a datetime or timedelta dtype: {!r}".format(series.name))
        return kind

    types, rows, frequencies, alert_rows = {}, [], {}, []
    for column, series in frame.items():
        kind = kind_for(series, overrides.get(column))
        types[column] = kind
        try:
            row = _profile(series, kind)
        except TypeError as error:
            raise ValueError("Column {!r} must contain scalar, hashable values".format(column)) from error
        rows.append(row)
        if kind == "categorical":
            frequencies[column] = series.value_counts(dropna=False)
        for condition, code, message in (
                (row["missing"] > 0, "missing", "{} missing values".format(row["missing"])),
                (row["count"] == 0 and len(series) > 0, "all_missing", "All values are missing"),
                (row["unique"] == 1, "constant", "One distinct nonmissing value"),
                (row["count"] > 1 and row["unique"] == row["count"], "unique", "Every nonmissing value is unique"),
                (row.get("infinite", 0) > 0, "infinite", "Infinite values excluded from numeric statistics")):
            if condition:
                alert_rows.append((column, code, message))
    variables = pd.DataFrame(rows, index=pd.Index(list(frame.columns),
                                                name="variable", tupleize_cols=False))
    if not len(variables.columns):
        variables = pd.DataFrame(columns=["dtype", "type", "count", "missing",
                                          "missing_fraction", "unique", "unique_fraction"],
                                 index=variables.index)
    n_rows, n_columns = frame.shape
    missing = int(frame.isna().sum().sum())
    duplicates = int(frame.duplicated().sum()) if n_columns else 0
    overview = dict(rows=n_rows, columns=n_columns, missing_cells=missing,
                    missing_fraction=missing / frame.size if frame.size else 0.0,
                    duplicate_rows=duplicates,
                    duplicate_fraction=duplicates / n_rows if n_rows else 0.0,
                    memory_bytes=int(frame.memory_usage(index=True, deep=True).sum()))
    if duplicates:
        alert_rows.append((None, "duplicates", "{} duplicate rows after the first occurrence".format(duplicates)))
    numeric_columns = [column for column in frame.columns if types[column] == "numeric"]
    numeric = frame[numeric_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    correlations = numeric.corr(method=corr_method, min_periods=2) if corr_threshold is not None else pd.DataFrame()
    high = _high_correlations(correlations, corr_threshold)
    for left, right, value in high.itertuples(index=False, name=None):
        alert_rows.append((left, "high_correlation", "Correlation with {}: {:.4g}".format(right, value)))
    response_summary = response_correlations = None
    if response is not None:
        response_type = kind_for(response, response_type)
        response_summary = pd.Series(_profile(response, response_type), name=response.name)
        if response_type == "numeric" and corr_threshold is not None:
            target = response.astype(float).replace([np.inf, -np.inf], np.nan)
            # Reset indexes to match the response by row position.
            # This also supports input with duplicate row labels.
            response_correlations = pd.Series(
                [numeric[column].reset_index(drop=True).corr(
                    target.reset_index(drop=True), method=corr_method, min_periods=2)
                 for column in numeric_columns],
                index=pd.Index(numeric_columns, tupleize_cols=False),
                name="correlation", dtype=float)
    notes = ["Numeric statistics exclude missing and infinite values. "
             "Correlations use numeric features. Each pair uses rows with two valid values."]
    images = {}
    if plots:
        notes.append("Category displays show at most {} values. Response comparisons "
                     "show the most frequent categories.".format(max_categories))
        for column, series in frame.items():
            images[("distribution", column)] = _distribution(series, types[column], max_categories)
        if missing:
            figure = _new_figure()
            ax = figure.subplots()
            counts = frame.isna().sum()
            ax.barh(range(n_columns), counts.to_numpy())
            ax.set_yticks(range(n_columns))
            ax.set_yticklabels([str(column) for column in frame.columns])
            ax.set_xlabel("Missing values")
            images[("dataset", "missingness")] = _encode_plot(figure)
        if len(correlations) and correlations.notna().any().any():
            figure = _new_figure()
            ax = figure.subplots()
            heatmap = ax.imshow(np.ma.masked_invalid(correlations.to_numpy()),
                                vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(range(len(correlations)))
            ax.set_yticks(range(len(correlations)))
            ax.set_xticklabels([str(c) for c in correlations.columns], rotation=90)
            ax.set_yticklabels([str(c) for c in correlations.columns])
            figure.colorbar(heatmap, ax=ax, label=corr_method)
            images[("dataset", "correlations")] = _encode_plot(figure)
        positions = np.arange(n_rows)
        if max_plot_rows is not None and n_rows > max_plot_rows and (plot_pairs or response is not None):
            positions = pd.Series(positions).sample(max_plot_rows, random_state=random_state).to_numpy()
            notes.append("Pair and response plots use a fixed sample of {} of {} rows. "
                         "All statistics use the full dataset.".format(max_plot_rows, n_rows))
        if response is not None:
            images[("response", "distribution")] = _distribution(response, response_type, max_categories)
            for column, series in frame.items():
                image = _comparison(series.iloc[positions], response.iloc[positions],
                                    types[column], response_type, max_categories)
                if image is not None:
                    images[("comparison", column)] = image
            if response_type == "date" or "date" in types.values():
                notes.append("Date variables have distribution plots. Response comparisons skip date variables.")
        if plot_pairs and len(numeric_columns) and n_rows:
            selected = numeric_columns[:max_pair_columns]
            images[("dataset", "pairs")] = _pair_plot(numeric.loc[:, selected].iloc[positions])
            if len(selected) < len(numeric_columns):
                notes.append("Pair plot shows the first {} of {} numeric columns.".format(
                    len(selected), len(numeric_columns)))
    report = EDAReport(title, overview, variables, frequencies, correlations,
                       high, pd.DataFrame(alert_rows, columns=["variable", "kind", "message"]),
                       images, response_summary, response_correlations, notes, max_categories)
    if output_file is not None:
        report.to_file(output_file)
    return report
