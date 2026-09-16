#!/usr/bin/env python
"""
Precursor-vs-aftermath analysis for Market Geometry transitions.

Goal
----
Separate:
1) PRECURSOR transitions: geometry changes strongly before the market is already deeply stressed
2) IN-STRESS transitions: geometry changes after a major selloff is already underway

Inputs
------
market_geometry_transition_metrics.csv

Outputs
-------
market_geometry_precursor_events.csv
market_geometry_precursor_summary.csv
market_geometry_precursor_full.png
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INFILE = Path("market_geometry_transition_metrics.csv")
OUT_EVENTS = Path("market_geometry_precursor_events.csv")
OUT_SUMMARY = Path("market_geometry_precursor_summary.csv")

TRANSITION_PCT_THRESHOLD = 97.5
MIN_SEPARATION_DAYS = 20

def rolling_drawdown(px: pd.Series, window: int) -> pd.Series:
    peak = px.rolling(window, min_periods=1).max()
    return px / peak - 1.0

def classify_state(row) -> str:
    # "Precursor" means the geometry transition occurs before the market is
    # already in a major drawdown / extreme geometry-stress state.
    if (
        row["GeometryStressPercentile"] < 95
        and row["SPY_Drawdown20D"] > -0.05
        and row["SPY_Drawdown60D"] > -0.08
    ):
        return "PRECURSOR"
    return "IN_STRESS"

def forward_metrics(df: pd.DataFrame, pos: int, horizon: int):
    px0 = df.iloc[pos]["SPY_AdjClose"]
    if pos + 1 >= len(df):
        return np.nan, np.nan
    end = min(len(df), pos + horizon + 1)
    future = df.iloc[pos+1:end]["SPY_AdjClose"]
    if len(future) == 0:
        return np.nan, np.nan
    ret = future.iloc[-1] / px0 - 1.0
    worst = (future / px0 - 1.0).min()
    return ret, worst

def main():
    df = pd.read_csv(INFILE, parse_dates=["Date"]).set_index("Date").sort_index()

    df["SPY_Drawdown20D"] = rolling_drawdown(df["SPY_AdjClose"], 20)
    df["SPY_Drawdown60D"] = rolling_drawdown(df["SPY_AdjClose"], 60)

    # Candidate transition peaks.
    cand = df[df["TransitionScorePercentile"] >= TRANSITION_PCT_THRESHOLD].copy()

    # Collapse nearby candidates: keep the highest transition score in each
    # 20-trading-day neighborhood.
    positions = [df.index.get_loc(d) for d in cand.index]
    selected = []

    for pos in positions:
        if not selected:
            selected.append(pos)
            continue

        if pos - selected[-1] < MIN_SEPARATION_DAYS:
            if df.iloc[pos]["TransitionScore"] > df.iloc[selected[-1]]["TransitionScore"]:
                selected[-1] = pos
        else:
            selected.append(pos)

    rows = []
    for event_id, pos in enumerate(selected, 1):
        row = df.iloc[pos]
        rec = {
            "EventID": event_id,
            "Date": df.index[pos],
            "TransitionScore": row["TransitionScore"],
            "TransitionPercentile": row["TransitionScorePercentile"],
            "GeometryStress": row["GeometryStress"],
            "GeometryStressPercentile": row["GeometryStressPercentile"],
            "SPY_Drawdown20D": row["SPY_Drawdown20D"],
            "SPY_Drawdown60D": row["SPY_Drawdown60D"],
            "SigmaMin": row["SigmaMin"],
            "ConditionNumber": row["ConditionNumber"],
            "WeakDirectionRotationDeg": row["WeakDirectionRotationDeg"],
        }
        rec["StateClass"] = classify_state(rec)

        for h in (5, 10, 20, 40):
            ret, worst = forward_metrics(df, pos, h)
            rec[f"SPY_ReturnAfter{h}D"] = ret
            rec[f"SPY_WorstAfter{h}D"] = worst

        rows.append(rec)

    events = pd.DataFrame(rows)
    events.to_csv(OUT_EVENTS, index=False, float_format="%.10g")

    summary_rows = []
    for state, g in events.groupby("StateClass"):
        for h in (5, 10, 20, 40):
            summary_rows.append({
                "StateClass": state,
                "HorizonDays": h,
                "Events": len(g),
                "MeanForwardReturn": g[f"SPY_ReturnAfter{h}D"].mean(),
                "MedianForwardReturn": g[f"SPY_ReturnAfter{h}D"].median(),
                "MeanWorstReturn": g[f"SPY_WorstAfter{h}D"].mean(),
                "MedianWorstReturn": g[f"SPY_WorstAfter{h}D"].median(),
                "PctNegativeForwardReturn": (g[f"SPY_ReturnAfter{h}D"] < 0).mean(),
                "PctWorstBelowMinus5Pct": (g[f"SPY_WorstAfter{h}D"] <= -0.05).mean(),
                "PctWorstBelowMinus10Pct": (g[f"SPY_WorstAfter{h}D"] <= -0.10).mean(),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False, float_format="%.10g")

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(df.index, df["SPY_AdjClose"])
    axes[0].set_ylabel("SPY")
    axes[0].set_title("Precursor vs In-Stress Geometry Transitions")

    axes[1].plot(df.index, df["TransitionScore"])
    axes[1].set_ylabel("Transition score")

    axes[2].plot(df.index, df["GeometryStressPercentile"])
    axes[2].axhline(95, linestyle="--", alpha=0.4)
    axes[2].set_ylabel("Stress percentile")
    axes[2].set_xlabel("Date")

    for _, r in events.iterrows():
        d = pd.Timestamp(r["Date"])
        for ax in axes:
            ax.axvline(d, linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig("market_geometry_precursor_full.png", dpi=160)
    plt.close(fig)

    print(f"Saved: {OUT_EVENTS.resolve()}")
    print(f"Saved: {OUT_SUMMARY.resolve()}")
    print("Saved: market_geometry_precursor_full.png")

    print("\nCandidate transition events:")
    cols = [
        "EventID", "Date", "StateClass",
        "TransitionPercentile", "GeometryStressPercentile",
        "SPY_Drawdown20D", "SPY_Drawdown60D",
        "SPY_ReturnAfter20D", "SPY_WorstAfter20D"
    ]
    print(events[cols].to_string(index=False))

    print("\nPrecursor vs in-stress summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
