#!/usr/bin/env python
"""
Composite Market Geometry Stress + lead/lag analysis.

Inputs
------
market_geometry_dataset.csv
market_geometry_svd_diagnostics.csv

Outputs
-------
market_geometry_stress.csv
market_geometry_event_summary.csv
market_geometry_leadlag_summary.csv
market_geometry_stress_full.png
market_geometry_zoom_2008.png
market_geometry_zoom_2020.png
market_geometry_zoom_2022.png

Method
------
Creates a leakage-safe composite geometry stress score from:
1) log(condition number)      -> high is stressed
2) weak-direction rotation   -> high is stressed
3) smallest active singular value -> low is stressed

Each component is converted to a trailing z-score using only the PRIOR
252 trading days. The composite is the mean of the three available
component z-scores.

This is a regime-geometry diagnostic. It is not, by itself, proof of a
Thom catastrophe.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("market_geometry_dataset.csv")
SVD = Path("market_geometry_svd_diagnostics.csv")

OUT_STRESS = Path("market_geometry_stress.csv")
OUT_EVENTS = Path("market_geometry_event_summary.csv")
OUT_LEADLAG = Path("market_geometry_leadlag_summary.csv")

BASELINE_WINDOW = 252
MIN_BASELINE = 126

EVENTS = {
    "Global Financial Crisis": "2008-09-15",
    "COVID Crash": "2020-03-16",
    "2022 Rate Shock": "2022-06-13",
}

ZOOM_WINDOWS = {
    "2008": ("2007-01-01", "2009-06-30"),
    "2020": ("2019-07-01", "2020-08-31"),
    "2022": ("2021-07-01", "2022-12-31"),
}


def trailing_zscore(s: pd.Series) -> pd.Series:
    """
    Standardize today's value against the PRIOR trailing history only.
    The baseline is shifted by one day to avoid look-ahead.
    """
    hist = s.shift(1)
    mean = hist.rolling(BASELINE_WINDOW, min_periods=MIN_BASELINE).mean()
    std = hist.rolling(BASELINE_WINDOW, min_periods=MIN_BASELINE).std()
    return (s - mean) / std.replace(0, np.nan)


def forward_min_return(px: pd.Series, horizon: int) -> pd.Series:
    """
    Worst close-to-close return experienced over the next `horizon`
    trading days, relative to today's close.
    """
    arr = px.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)

    for i in range(len(arr)):
        j = min(len(arr), i + horizon + 1)
        future = arr[i + 1:j]
        if len(future) == 0 or not np.isfinite(arr[i]):
            continue
        rets = future / arr[i] - 1.0
        if np.isfinite(rets).any():
            out[i] = np.nanmin(rets)

    return pd.Series(out, index=px.index)


def forward_return(px: pd.Series, horizon: int) -> pd.Series:
    return px.shift(-horizon) / px - 1.0


def build_stress_table() -> pd.DataFrame:
    market = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date").sort_index()
    svd = pd.read_csv(SVD, parse_dates=["Date"]).set_index("Date").sort_index()

    cols = ["SigmaMin", "ConditionNumber", "WeakDirectionRotationDeg"]
    df = market[["SPY_AdjClose"]].join(svd[cols], how="inner").copy()

    # Stabilize condition number before standardization.
    df["LogConditionNumber"] = np.log1p(df["ConditionNumber"])

    # Leakage-safe component z-scores.
    df["Z_LogCondition"] = trailing_zscore(df["LogConditionNumber"])
    df["Z_Rotation"] = trailing_zscore(df["WeakDirectionRotationDeg"])
    df["Z_NegSigmaMin"] = trailing_zscore(-df["SigmaMin"])

    component_cols = ["Z_LogCondition", "Z_Rotation", "Z_NegSigmaMin"]
    df["GeometryStress"] = df[component_cols].mean(axis=1, skipna=True)

    # Trailing percentile is often easier to interpret than a z-score.
    # Current observation is ranked against the prior 252 days only.
    stress = df["GeometryStress"]

    def prior_percentile(i: int) -> float:
        if i < MIN_BASELINE or not np.isfinite(stress.iloc[i]):
            return np.nan
        start = max(0, i - BASELINE_WINDOW)
        hist = stress.iloc[start:i].dropna()
        if len(hist) < MIN_BASELINE:
            return np.nan
        return 100.0 * (hist < stress.iloc[i]).mean()

    df["GeometryStressPercentile"] = [
        prior_percentile(i) for i in range(len(df))
    ]

    # Forward market outcomes for lead/lag testing.
    for h in (5, 10, 20, 40):
        df[f"SPY_FwdReturn_{h}D"] = forward_return(df["SPY_AdjClose"], h)
        df[f"SPY_FwdWorst_{h}D"] = forward_min_return(df["SPY_AdjClose"], h)

    return df


def event_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for name, date_str in EVENTS.items():
        event_date = pd.Timestamp(date_str)
        if event_date < df.index.min() or event_date > df.index.max():
            continue

        # Use nearest trading day at or before the event date.
        prior = df.index[df.index <= event_date]
        if len(prior) == 0:
            continue
        event_day = prior[-1]
        event_pos = df.index.get_loc(event_day)

        row = {
            "Event": name,
            "EventDate": event_day,
            "StressOnEvent": df.loc[event_day, "GeometryStress"],
            "StressPercentileOnEvent": df.loc[event_day, "GeometryStressPercentile"],
        }

        for lead in (5, 10, 20, 40):
            start_pos = max(0, event_pos - lead)
            window = df.iloc[start_pos:event_pos]
            if len(window):
                max_idx = window["GeometryStress"].idxmax()
                row[f"MaxStressPrior{lead}D"] = window["GeometryStress"].max()
                row[f"MaxStressPrior{lead}D_Date"] = max_idx
                row[f"MaxStressPctPrior{lead}D"] = window.loc[
                    max_idx, "GeometryStressPercentile"
                ]
            else:
                row[f"MaxStressPrior{lead}D"] = np.nan
                row[f"MaxStressPrior{lead}D_Date"] = pd.NaT
                row[f"MaxStressPctPrior{lead}D"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def leadlag_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare subsequent SPY outcomes after increasingly high geometry-stress
    percentiles. This is descriptive, not a trading backtest.
    """
    rows = []

    for threshold in (80, 90, 95, 97.5, 99):
        mask = df["GeometryStressPercentile"] >= threshold
        n = int(mask.sum())
        if n == 0:
            continue

        for h in (5, 10, 20, 40):
            fwd = df.loc[mask, f"SPY_FwdReturn_{h}D"].dropna()
            worst = df.loc[mask, f"SPY_FwdWorst_{h}D"].dropna()

            rows.append({
                "StressPercentileThreshold": threshold,
                "HorizonDays": h,
                "Observations": min(len(fwd), len(worst)),
                "MeanForwardReturn": fwd.mean(),
                "MedianForwardReturn": fwd.median(),
                "MeanForwardWorstReturn": worst.mean(),
                "MedianForwardWorstReturn": worst.median(),
                "PctForwardReturnsNegative": (fwd < 0).mean() if len(fwd) else np.nan,
                "PctWorstReturnBelowMinus5Pct": (worst <= -0.05).mean() if len(worst) else np.nan,
                "PctWorstReturnBelowMinus10Pct": (worst <= -0.10).mean() if len(worst) else np.nan,
            })

    return pd.DataFrame(rows)


def plot_window(df: pd.DataFrame, start: str, end: str, outfile: Path, title: str):
    w = df.loc[start:end].copy()
    if w.empty:
        return

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)

    axes[0].plot(w.index, w["SPY_AdjClose"])
    axes[0].set_ylabel("SPY")
    axes[0].set_title(title)

    axes[1].plot(w.index, w["SigmaMin"])
    axes[1].set_ylabel("Sigma min")

    axes[2].plot(w.index, w["ConditionNumber"])
    axes[2].set_ylabel("Condition #")

    axes[3].plot(w.index, w["WeakDirectionRotationDeg"])
    axes[3].set_ylabel("Rotation")

    axes[4].plot(w.index, w["GeometryStress"])
    axes[4].axhline(0, linestyle="--", alpha=0.4)
    axes[4].set_ylabel("Geometry stress")
    axes[4].set_xlabel("Date")

    # Mark event date if it is inside the window.
    for event_name, event_date in EVENTS.items():
        d = pd.Timestamp(event_date)
        if pd.Timestamp(start) <= d <= pd.Timestamp(end):
            for ax in axes:
                ax.axvline(d, linestyle="--", alpha=0.6)
            axes[0].text(
                d,
                axes[0].get_ylim()[1],
                event_name,
                rotation=90,
                va="top",
                ha="right",
            )

    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


def plot_full(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df.index, df["SPY_AdjClose"])
    axes[0].set_ylabel("SPY")
    axes[0].set_title("Market Geometry Stress")

    axes[1].plot(df.index, df["GeometryStress"])
    axes[1].axhline(0, linestyle="--", alpha=0.4)
    axes[1].set_ylabel("Stress z")

    axes[2].plot(df.index, df["GeometryStressPercentile"])
    axes[2].axhline(95, linestyle="--", alpha=0.4)
    axes[2].set_ylabel("Stress pctile")
    axes[2].set_xlabel("Date")

    for event_date in EVENTS.values():
        d = pd.Timestamp(event_date)
        for ax in axes:
            ax.axvline(d, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig("market_geometry_stress_full.png", dpi=160)
    plt.close(fig)


def main():
    df = build_stress_table()
    events = event_summary(df)
    leadlag = leadlag_summary(df)

    df.to_csv(OUT_STRESS, float_format="%.10g")
    events.to_csv(OUT_EVENTS, index=False, float_format="%.10g")
    leadlag.to_csv(OUT_LEADLAG, index=False, float_format="%.10g")

    plot_full(df)

    for tag, (start, end) in ZOOM_WINDOWS.items():
        plot_window(
            df,
            start,
            end,
            Path(f"market_geometry_zoom_{tag}.png"),
            f"Market Geometry — {tag} Window",
        )

    print(f"Saved: {OUT_STRESS.resolve()}")
    print(f"Saved: {OUT_EVENTS.resolve()}")
    print(f"Saved: {OUT_LEADLAG.resolve()}")
    print("Saved: market_geometry_stress_full.png")
    print("Saved: market_geometry_zoom_2008.png")
    print("Saved: market_geometry_zoom_2020.png")
    print("Saved: market_geometry_zoom_2022.png")

    print("\nEvent summary:")
    if len(events):
        print(events.to_string(index=False))

    print("\nLead/lag summary:")
    if len(leadlag):
        show = leadlag[
            [
                "StressPercentileThreshold",
                "HorizonDays",
                "Observations",
                "MeanForwardReturn",
                "MeanForwardWorstReturn",
                "PctForwardReturnsNegative",
                "PctWorstReturnBelowMinus5Pct",
                "PctWorstReturnBelowMinus10Pct",
            ]
        ]
        print(show.to_string(index=False))

if __name__ == "__main__":
    main()
