# Tom's Statistics and Data Science Toolkit (tsdst)

![Build](https://github.com/tomwerner5/tsdst/workflows/Build/badge.svg)

[![Anaconda-Server Badge](https://anaconda.org/tomwerner5/tsdst/badges/version.svg)](https://anaconda.org/tomwerner5/tsdst)

[![Anaconda-Server Badge](https://anaconda.org/tomwerner5/tsdst/badges/latest_release_date.svg)](https://anaconda.org/tomwerner5/tsdst)

## Introduction and Motivation

This project began as a list of functions for tasks I did repeatedly and hated coding every new instance. Now, it has evolved into a statistical utility tool and a notepad. Some of these functions were created because I wanted to learn and understand the innards of some mathematical concept, and are not optimized in any way. Other functions have been used repeatedly on projects and are hopefully optimized to some extent. 

Some of the more useful code here extends functionality of sklearn and other libraries. A majority of the functions are either related or similar to popular functions in [sklearn](https://scikit-learn.org/) or [statsmodels](https://www.statsmodels.org/stable/index.html). What makes the implementations here different are generally two things:

1. Remove Code Abstractions
    - For a majority of the functions, I tried to make it easy for someone (including me) to review and understand what is happening inside a function or set of functions so that the data science or statistical concepts can be clearly read and understood. I tried to avoid writing code that didn't make it clear what was happening mathmatically. This sometimes means that I ignore (purposefully) programming concepts that could make these functions much faster. However, most of the time I simply missed something that could easily be done another way and not affect the readability at all. I am not opposed to this being practically useful to others, so if there are obvious improvements, please let me know.
2. Extend Existing Functionality
    - There are some functions in existing data science libraries that I just don't like, or that I wish were done differently. Other times, there are methods or functions that just don't exist anywhere else (to my knowledge). And still other times, there are things that I could never figure out myself, but once the groundwork was laid, could extend to other useful situations. 

## Installation

Version 2 requires Python 3.12 or newer. It supports current scientific packages,
including pandas 3.1, NumPy 2.5, and scikit-learn 1.9.
See [the change notes](CHANGELOG.md) for changes that affect existing code.

To Install:

```{python}
pip install tsdst
```

Or:

```{python}
conda install -c tomwerner5 tsdst
```

## Using This Package

After installing, simply import to your module with `import tsdst`. A detailed description of each module/function/class (with examples) can be found on the documentation page. Documentation can be found [here:](https://tomwerner5.github.io/tsdst/)

## Automatic EDA

Use `auto_eda` to explore a dataset without a response variable.
Supply an output path to save an HTML report with its plots:

```python
from tsdst.eda import auto_eda

report = auto_eda(data, output_file="reports/eda.html")
report.variables
report.high_correlations
report.alerts
```

The report includes missing values, duplicate rows, variable summaries,
distributions, and numeric correlations. Use `plots=False` to compute tables
only. Use `plot_pairs=True` to add a scatter matrix with row and column limits.
If you have a response variable, use `response="column_name"` to compare it
with the features. The analysis keeps the input unchanged and does not fit
a model. See [the EDA guide](docs/source/eda.rst) for types and output options.

## Development

Install the test dependencies and run the suite:

```bash
python -m pip install -e '.[test]'
python -m pytest --cov=tsdst --cov-report=term-missing --cov-report=json:coverage.json
python scripts/check_function_coverage.py coverage.json
```

The suite checks each package function and method. Line coverage has a separate
85% minimum. The test workflow covers Python 3.12, 3.13, and 3.14.
Install `tsdst[io]` to use the Feather and Parquet helpers without the test tools.

Build the documentation in a uv environment:

```bash
uv pip install --python .venv/bin/python -e '.[docs]'
.venv/bin/python -m sphinx -b html -W --keep-going -d .venv/sphinx-doctrees docs/source docs/build/html
```

Open `docs/build/html/index.html` to review the result. The build reports
missing references and formatting problems as errors.


## Licensing

I am feeling pretty liberal with what can/can't be done with this project, so I have provisioned it with an MIT license. If anyone does end up using this for something cool, I wouldn't mind a name drop :)

## Final Remarks

I have received much assistance over the years both in documenting the conceptual aspects of these functions and writing the actual code, so here I offer a general thank-you to those individuals (most of you know who you are).

I have tried to attribute what is not my own work when appropriate, as well as acknowledge individuals who have given direct or indirect help. All of these attributions can be found in the documentation page (if I appear to have missed anything/anyone, please let me know). Please reach out if you have any feedback.
