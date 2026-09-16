#!/usr/bin/env python
"""
Plot core Market Geometry diagnostics against SPY.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("market_geometry_dataset.csv")
SVD = Path("market_geometry_svd_diagnostics.csv")
OUT = Path("market_geometry_diagnostics.png")

def main():
    market = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date")
    svd = pd.read_csv(SVD, parse_dates=["Date"]).set_index("Date")

    df = market[["SPY_AdjClose"]].join(
        svd[["SigmaMin", "ConditionNumber", "WeakDirectionRotationDeg"]],
        how="inner",
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(df.index, df["SPY_AdjClose"])
    axes[0].set_ylabel("SPY")
    axes[0].set_title("Market Geometry Diagnostics")

    axes[1].plot(df.index, df["SigmaMin"])
    axes[1].set_ylabel("Sigma min")

    axes[2].plot(df.index, df["ConditionNumber"])
    axes[2].set_ylabel("Condition #")

    axes[3].plot(df.index, df["WeakDirectionRotationDeg"])
    axes[3].set_ylabel("Rotation (deg)")
    axes[3].set_xlabel("Date")

    for ax in axes:
        for date in ["2008-09-15", "2020-03-16", "2022-06-13"]:
            ax.axvline(pd.Timestamp(date), linestyle="--", alpha=0.45)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"Saved plot to {OUT.resolve()}")

if __name__ == "__main__":
    main()
