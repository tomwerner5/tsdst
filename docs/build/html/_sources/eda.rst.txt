Automatic exploratory data analysis
===================================

Use ``tsdst.eda`` to explore a dataset without a response variable.
This module builds on the type rules, correlation checks, and plots in
the Kaggle ``Common_Data_Science_Functions.py`` file. It adds data summaries,
checks for common data issues, and an HTML report.

Get started
-----------

.. code-block:: python

    import pandas as pd
    from tsdst.eda import auto_eda

    data = pd.read_csv("my_data.csv")
    report = auto_eda(data, output_file="reports/eda.html")

    report.overview             # Check row counts, missing values, and duplicates.
    report.variables            # Review types, counts, and summary statistics.
    report.frequencies          # Get full category counts, including missing values.
    report.correlations         # Review numeric correlations.
    report.high_correlations    # Get each pair that meets the threshold.
    report.alerts               # Check data issues that need review.

    # Save another copy of the report.
    report.to_file("reports/another_copy.html")

Display ``report`` in a notebook to view the tables and plots.
Each HTML report includes its styles and plot images.
You can open the report without an internet connection.

If you omit the output path, the function returns the report without writing
a file. The analysis keeps the input unchanged. It does not fit a model,
fill missing values, or remove features.

Types and statistics
--------------------

The module treats strings, categories, and booleans as categorical variables.
Datetime and timedelta columns receive date summaries. A numeric column with
at most five unique nonmissing values is also categorical.
This rule follows the starter code. Use ``n_unique=None`` to disable it.

To set a type for a specific column, use ``feature_types``:

.. code-block:: python

    report = auto_eda(
        data,
        feature_types={"account_id": "categorical", "rating": "numeric"},
        corr_method="spearman",
        corr_threshold=0.8,
    )

A type override does not convert the data. If a string represents a date or
number, convert it in pandas before you run the analysis.
Each cell must contain one scalar, hashable value.
Each column must have a unique label. The module does not support complex numbers.

Missing fractions use the full row or cell count. Unique counts exclude
missing values. Unique fractions use the count of nonmissing values.
Numeric statistics exclude missing and infinite values.
The report counts infinities separately and uses sample standard deviation.

The outlier count uses the interquartile range (IQR).
It counts values below the first quartile minus 1.5 times the IQR.
It also counts values above the third quartile plus 1.5 times the IQR.

Correlations use numeric features only. Each pair uses rows with valid values
for both features. Each pair that meets the absolute threshold appears once
and keeps its correlation sign. Use ``corr_threshold=None`` to skip correlations.

Use the alerts to review the data. Constants, unique values, strong
correlations, and outliers can be useful. The alerts do not remove features.

Compare an optional response
----------------------------

If you have a response variable, supply its column name, a separate Series,
or a one-dimensional array. A Series must have the same index as the features.
An array must have one value per row.

.. code-block:: python

    report = auto_eda(data, response="outcome", response_type="categorical")

If you supply a column name, the module excludes that column from the feature
summaries. The response receives a separate profile.

Plot types depend on the two variables:

* Two categorical variables use grouped bar plots.
* One numeric and one categorical variable use box plots.
* Two numeric variables use scatter plots.

Numeric responses also receive feature correlations. Date variables have
distribution plots. Response comparison plots skip date variables.
These comparisons do not fit models or select features.

Control work and output
-----------------------

.. code-block:: python

    # Compute tables without plots.
    report = auto_eda(data, plots=False)

    # Add a scatter matrix with row and column limits.
    report = auto_eda(
        data,
        plot_pairs=True,
        max_pair_columns=8,
        max_plot_rows=5000,
        max_categories=20,
        random_state=42,
    )

Statistics and category counts use the full dataset. Only response plots
and scatter matrices can use a row sample. The scatter matrix selects numeric
columns in input order, up to the column limit.

Correlations use all numeric features. Their computation cost grows with
the square of the number of numeric features.

Category distribution bars combine values beyond the display limit.
Grouped response plots show the most frequent categories only.
Full category counts remain available in ``report.frequencies``.
The report records plot limits and row sampling in ``report.notes`` and the HTML.

API
---

.. autofunction:: tsdst.eda.auto_eda
   :no-index:

.. autoclass:: tsdst.eda.EDAReport
   :members: to_html, to_file
   :no-index:
