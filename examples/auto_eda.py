"""Create an EDA report from sample data without a response variable.

Install tsdst. Run this command from the repository root:
    python examples/auto_eda.py /tmp/tsdst-eda-example.html
"""

import argparse

import numpy as np
import pandas as pd

from tsdst.eda import auto_eda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Destination HTML file")
    args = parser.parse_args()
    rng = np.random.RandomState(42)
    size = 200
    measurements = rng.lognormal(mean=2, sigma=.6, size=size)
    data = pd.DataFrame({
        "measurement": measurements,
        "related_measurement": measurements * 2 + rng.normal(size=size),
        "region": rng.choice(["north", "south", "west"], size=size),
        "active": rng.choice([True, False], size=size),
        "recorded_at": pd.date_range("2025-01-01", periods=size, freq="h"),
        "constant": "example",
    })
    data.loc[::10, "measurement"] = np.nan
    data.loc[::17, "region"] = None
    data = pd.concat([data, data.iloc[:3]], ignore_index=True)
    report = auto_eda(data, title="Example data exploration", plot_pairs=True,
                      corr_threshold=.8, output_file=args.output)
    print(pd.Series(report.overview).to_string())
    print("Report saved to {}".format(args.output))


if __name__ == "__main__":
    main()
