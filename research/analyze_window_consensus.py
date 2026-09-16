#!/usr/bin/env python
"""
Collapse robustness results to independent WINDOW-level consensus.

Why
---
The earlier 9-spec robustness grid used 3 rolling windows x 3 ridge penalties.
The ridge variants produced nearly identical results, so they should not be
counted as 9 independent confirmations.

This script:
1. collapses precursor detections across ridge values within each window
2. groups nearby detections across windows into consensus episodes
3. reports whether an episode appears in 1/3, 2/3, or 3/3 windows
4. averages sign-invariant weak-direction contributions within each episode

Input
-----
robustness_precursor_events.csv

Outputs
-------
window_consensus_precursors.csv
window_consensus_driver_profiles.csv
window_consensus_summary.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INFILE = Path("robustness_precursor_events.csv")
OUT_EVENTS = Path("window_consensus_precursors.csv")
OUT_DRIVERS = Path("window_consensus_driver_profiles.csv")

WINDOWS = [63, 126, 252]
MATCH_CALENDAR_DAYS = 7

VARS = [
    "VIX",
    "Treasury10Y",
    "Curve10Y2Y",
    "CreditSpread_BAA10Y",
    "WTI",
    "BroadUSD",
    "RSP_SPY_Ratio",
    "SPY_Volume_Z20",
]
CONTRIB = [f"Contribution_{v}" for v in VARS]


def collapse_within_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ridge variants within the same rolling window are treated as replicates,
    not independent confirmations. Nearby detections are collapsed to the
    highest-transition-percentile representative date.
    """
    rows = []

    for window, g in df.groupby("Window"):
        g = g.sort_values("Date").copy()
        if g.empty:
            continue

        groups = []
        current = [g.iloc[0]]

        for _, row in g.iloc[1:].iterrows():
            last_date = pd.Timestamp(current[-1]["Date"])
            this_date = pd.Timestamp(row["Date"])

            if (this_date - last_date).days <= MATCH_CALENDAR_DAYS:
                current.append(row)
            else:
                groups.append(pd.DataFrame(current))
                current = [row]

        groups.append(pd.DataFrame(current))

        for block in groups:
            # Average across ridge replicates, but preserve a representative date
            # from the strongest transition observation.
            idx = block["TransitionPercentile"].idxmax()
            rep = block.loc[idx]

            rec = {
                "Window": int(window),
                "Date": pd.Timestamp(rep["Date"]),
                "TransitionPercentile": block["TransitionPercentile"].mean(),
                "GeometryStressPercentile": block["GeometryStressPercentile"].mean(),
                "SPY_ReturnAfter20D": block["SPY_ReturnAfter20D"].mean(),
                "SPY_WorstAfter20D": block["SPY_WorstAfter20D"].mean(),
                "RidgeReplicates": block["Ridge"].nunique(),
            }

            for c in CONTRIB:
                rec[c] = block[c].mean()

            rows.append(rec)

    return pd.DataFrame(rows).sort_values("Date")


def build_cross_window_consensus(collapsed: pd.DataFrame):
    if collapsed.empty:
        return pd.DataFrame(), pd.DataFrame()

    records = collapsed.sort_values("Date").to_dict("records")
    used = [False] * len(records)
    episodes = []

    for i, rec in enumerate(records):
        if used[i]:
            continue

        group = [i]
        used[i] = True
        anchor = pd.Timestamp(rec["Date"])

        # Gather all detections within +/- MATCH_CALENDAR_DAYS of anchor.
        changed = True
        while changed:
            changed = False
            dates = [pd.Timestamp(records[j]["Date"]) for j in group]
            lo = min(dates) - pd.Timedelta(days=MATCH_CALENDAR_DAYS)
            hi = max(dates) + pd.Timedelta(days=MATCH_CALENDAR_DAYS)

            for j, other in enumerate(records):
                if used[j]:
                    continue
                d = pd.Timestamp(other["Date"])
                if lo <= d <= hi:
                    group.append(j)
                    used[j] = True
                    changed = True

        block = pd.DataFrame([records[j] for j in group])

        # At most one collapsed detection per window should matter for consensus.
        # If a window contributes more than one nearby detection, keep strongest.
        keep_rows = []
        for window, wg in block.groupby("Window"):
            keep_rows.append(wg.loc[wg["TransitionPercentile"].idxmax()])
        block = pd.DataFrame(keep_rows)

        rep_idx = block["TransitionPercentile"].idxmax()
        rep_date = pd.Timestamp(block.loc[rep_idx, "Date"])

        ep = {
            "ConsensusDate": rep_date,
            "WindowsDetecting": block["Window"].nunique(),
            "TotalWindows": len(WINDOWS),
            "WindowConsensusFraction": block["Window"].nunique() / len(WINDOWS),
            "DetectedBy63": int(63 in block["Window"].values),
            "DetectedBy126": int(126 in block["Window"].values),
            "DetectedBy252": int(252 in block["Window"].values),
            "MeanTransitionPercentile": block["TransitionPercentile"].mean(),
            "MeanGeometryStressPercentile": block["GeometryStressPercentile"].mean(),
            "MeanSPYReturnAfter20D": block["SPY_ReturnAfter20D"].mean(),
            "MeanSPYWorstAfter20D": block["SPY_WorstAfter20D"].mean(),
        }

        for c in CONTRIB:
            ep[c] = block[c].mean()

        episodes.append(ep)

    events = pd.DataFrame(episodes).sort_values(
        ["WindowsDetecting", "ConsensusDate"],
        ascending=[False, True],
    )

    driver_rows = []
    for _, r in events.iterrows():
        contributions = pd.Series(
            {v: r[f"Contribution_{v}"] for v in VARS}
        ).sort_values(ascending=False)

        driver_rows.append({
            "ConsensusDate": r["ConsensusDate"],
            "WindowsDetecting": r["WindowsDetecting"],
            "TopDriver1": contributions.index[0],
            "TopDriver1Contribution": contributions.iloc[0],
            "TopDriver2": contributions.index[1],
            "TopDriver2Contribution": contributions.iloc[1],
            "TopDriver3": contributions.index[2],
            "TopDriver3Contribution": contributions.iloc[2],
        })

    drivers = pd.DataFrame(driver_rows)

    return events, drivers


def main():
    df = pd.read_csv(INFILE, parse_dates=["Date"])
    collapsed = collapse_within_window(df)
    events, drivers = build_cross_window_consensus(collapsed)

    events.to_csv(OUT_EVENTS, index=False, float_format="%.10g")
    drivers.to_csv(OUT_DRIVERS, index=False, float_format="%.10g")

    print("Window-level consensus precursor events:")
    show = [
        "ConsensusDate", "WindowsDetecting",
        "DetectedBy63", "DetectedBy126", "DetectedBy252",
        "MeanTransitionPercentile",
        "MeanGeometryStressPercentile",
        "MeanSPYReturnAfter20D",
        "MeanSPYWorstAfter20D",
    ]
    print(events[show].to_string(index=False))

    print("\nConsensus driver profiles:")
    print(drivers.to_string(index=False))

    # Simple plot of consensus strength and subsequent 20D outcome.
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].bar(events["ConsensusDate"], events["WindowsDetecting"], width=10)
    axes[0].set_ylabel("Windows detecting")
    axes[0].set_title("Independent Window-Level Precursor Consensus")
    axes[0].set_ylim(0, 3.3)

    axes[1].bar(events["ConsensusDate"], events["MeanSPYReturnAfter20D"], width=10)
    axes[1].axhline(0, linestyle="--", alpha=0.4)
    axes[1].set_ylabel("Mean SPY return after 20D")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig("window_consensus_summary.png", dpi=160)
    plt.close(fig)

    print("\nSaved:")
    print(OUT_EVENTS.resolve())
    print(OUT_DRIVERS.resolve())
    print("window_consensus_summary.png")


if __name__ == "__main__":
    main()
