#!/usr/bin/env python
"""
Episode-based Market Geometry transition analysis.

Builds on market_geometry_stress.csv and asks:
- Is geometry stress rising?
- Is it accelerating?
- Are singular values collapsing persistently?
- Is the weak direction rotating persistently?
- Do extreme geometry states cluster into distinct episodes?

Outputs
-------
market_geometry_transition_metrics.csv
market_geometry_episodes.csv
market_geometry_episode_outcomes.csv
market_geometry_transition_full.png
market_geometry_episode_<n>.png

Notes
-----
This is still an empirical regime-geometry analysis, not proof of a Thom
catastrophe. The purpose is to move from "high stress" to "trajectory into
a transition."
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STRESS_FILE = Path("market_geometry_stress.csv")
OUT_METRICS = Path("market_geometry_transition_metrics.csv")
OUT_EPISODES = Path("market_geometry_episodes.csv")
OUT_OUTCOMES = Path("market_geometry_episode_outcomes.csv")

# Episode definition: geometry stress percentile above threshold.
EPISODE_THRESHOLD = 95.0

# Allow short gaps inside the same episode so a brief 1-2 day dip does not
# split one regime transition into many separate events.
MAX_GAP_DAYS = 2

# Require at least this many above-threshold observations in an episode.
MIN_EPISODE_HITS = 2


def slope(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling OLS slope versus time index.
    Uses only current and past observations.
    """
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)

    def f(a):
        if np.isnan(a).any():
            return np.nan
        return float(np.dot(x, a - a.mean()) / denom)

    return series.rolling(window, min_periods=window).apply(f, raw=True)


def add_transition_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Geometry-stress velocity and acceleration.
    out["StressVelocity1D"] = out["GeometryStress"].diff()
    out["StressVelocity5D"] = slope(out["GeometryStress"], 5)
    out["StressVelocity10D"] = slope(out["GeometryStress"], 10)
    out["StressAcceleration"] = out["StressVelocity5D"].diff()

    # Directional trends in the underlying SVD quantities.
    out["SigmaMinSlope10D"] = slope(out["SigmaMin"], 10)
    out["LogConditionSlope10D"] = slope(np.log1p(out["ConditionNumber"]), 10)
    out["RotationSlope10D"] = slope(out["WeakDirectionRotationDeg"], 10)

    # Persistence measures over prior 10 observations.
    out["SigmaDeclineFraction10D"] = (
        (out["SigmaMin"].diff() < 0).astype(float).rolling(10, min_periods=10).mean()
    )
    out["ConditionRiseFraction10D"] = (
        (out["ConditionNumber"].diff() > 0).astype(float).rolling(10, min_periods=10).mean()
    )

    # A transition score emphasizing movement INTO stress.
    # Trailing z-scores use only prior observations.
    def trailing_z(s, window=252, minp=126):
        hist = s.shift(1)
        mu = hist.rolling(window, min_periods=minp).mean()
        sd = hist.rolling(window, min_periods=minp).std()
        return (s - mu) / sd.replace(0, np.nan)

    out["Z_StressVelocity5D"] = trailing_z(out["StressVelocity5D"])
    out["Z_StressAcceleration"] = trailing_z(out["StressAcceleration"])
    out["Z_NegSigmaSlope10D"] = trailing_z(-out["SigmaMinSlope10D"])
    out["Z_ConditionSlope10D"] = trailing_z(out["LogConditionSlope10D"])
    out["Z_RotationSlope10D"] = trailing_z(out["RotationSlope10D"])

    trans_components = [
        "Z_StressVelocity5D",
        "Z_StressAcceleration",
        "Z_NegSigmaSlope10D",
        "Z_ConditionSlope10D",
        "Z_RotationSlope10D",
    ]
    out["TransitionScore"] = out[trans_components].mean(axis=1, skipna=True)

    # Prior-history percentile of transition score.
    score = out["TransitionScore"]

    vals = []
    for i in range(len(out)):
        if i < 126 or not np.isfinite(score.iloc[i]):
            vals.append(np.nan)
            continue
        hist = score.iloc[max(0, i-252):i].dropna()
        if len(hist) < 126:
            vals.append(np.nan)
        else:
            vals.append(100.0 * (hist < score.iloc[i]).mean())
    out["TransitionScorePercentile"] = vals

    return out


def identify_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse consecutive/near-consecutive high-stress observations into
    distinct geometry episodes.
    """
    hit_positions = np.flatnonzero(
        df["GeometryStressPercentile"].to_numpy(dtype=float) >= EPISODE_THRESHOLD
    )

    if len(hit_positions) == 0:
        return pd.DataFrame()

    groups = []
    current = [hit_positions[0]]

    for pos in hit_positions[1:]:
        if pos - current[-1] <= MAX_GAP_DAYS + 1:
            current.append(pos)
        else:
            groups.append(current)
            current = [pos]
    groups.append(current)

    rows = []
    episode_id = 1

    for g in groups:
        if len(g) < MIN_EPISODE_HITS:
            continue

        start_pos = g[0]
        end_pos = g[-1]
        block = df.iloc[start_pos:end_pos+1]

        peak_stress_date = block["GeometryStress"].idxmax()
        peak_transition_date = block["TransitionScore"].idxmax()

        row = {
            "EpisodeID": episode_id,
            "StartDate": df.index[start_pos],
            "EndDate": df.index[end_pos],
            "CalendarTradingSpan": end_pos - start_pos + 1,
            "HighStressHits": len(g),
            "PeakStressDate": peak_stress_date,
            "PeakGeometryStress": block.loc[peak_stress_date, "GeometryStress"],
            "PeakStressPercentile": block.loc[peak_stress_date, "GeometryStressPercentile"],
            "PeakTransitionDate": peak_transition_date,
            "PeakTransitionScore": block.loc[peak_transition_date, "TransitionScore"],
            "PeakTransitionPercentile": block.loc[peak_transition_date, "TransitionScorePercentile"],
            "MinSigmaMin": block["SigmaMin"].min(),
            "MaxConditionNumber": block["ConditionNumber"].max(),
            "MaxRotationDeg": block["WeakDirectionRotationDeg"].max(),
        }
        rows.append(row)
        episode_id += 1

    return pd.DataFrame(rows)


def episode_outcomes(df: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, ep in episodes.iterrows():
        peak_date = pd.Timestamp(ep["PeakStressDate"])
        if peak_date not in df.index:
            continue
        pos = df.index.get_loc(peak_date)

        row = {
            "EpisodeID": int(ep["EpisodeID"]),
            "PeakStressDate": peak_date,
            "PeakGeometryStress": ep["PeakGeometryStress"],
            "PeakTransitionScore": ep["PeakTransitionScore"],
        }

        # Look backward to see if transition score rose before peak stress.
        for lead in (5, 10, 20, 40):
            a = max(0, pos - lead)
            pre = df.iloc[a:pos]
            row[f"MaxTransitionPrior{lead}D"] = (
                pre["TransitionScore"].max() if len(pre) else np.nan
            )
            row[f"MaxTransitionPctPrior{lead}D"] = (
                pre["TransitionScorePercentile"].max() if len(pre) else np.nan
            )

        # Market outcomes from peak stress date.
        px0 = df.iloc[pos]["SPY_AdjClose"]
        for h in (5, 10, 20, 40):
            if pos + h < len(df):
                pxh = df.iloc[pos+h]["SPY_AdjClose"]
                row[f"SPY_ReturnAfter{h}D"] = pxh / px0 - 1.0

                future = df.iloc[pos+1:pos+h+1]["SPY_AdjClose"]
                row[f"SPY_WorstAfter{h}D"] = (
                    (future / px0 - 1.0).min() if len(future) else np.nan
                )
            else:
                row[f"SPY_ReturnAfter{h}D"] = np.nan
                row[f"SPY_WorstAfter{h}D"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def plot_full(df: pd.DataFrame, episodes: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(df.index, df["SPY_AdjClose"])
    axes[0].set_ylabel("SPY")
    axes[0].set_title("Market Geometry Transition Analysis")

    axes[1].plot(df.index, df["GeometryStress"])
    axes[1].set_ylabel("Geometry stress")

    axes[2].plot(df.index, df["TransitionScore"])
    axes[2].set_ylabel("Transition score")

    axes[3].plot(df.index, df["TransitionScorePercentile"])
    axes[3].axhline(95, linestyle="--", alpha=0.4)
    axes[3].set_ylabel("Transition pctile")
    axes[3].set_xlabel("Date")

    for _, ep in episodes.iterrows():
        s = pd.Timestamp(ep["StartDate"])
        e = pd.Timestamp(ep["EndDate"])
        for ax in axes:
            ax.axvspan(s, e, alpha=0.08)

    fig.tight_layout()
    fig.savefig("market_geometry_transition_full.png", dpi=160)
    plt.close(fig)


def plot_episode(df: pd.DataFrame, ep: pd.Series):
    peak = pd.Timestamp(ep["PeakStressDate"])
    start_pos = df.index.get_loc(peak)
    a = max(0, start_pos - 60)
    b = min(len(df), start_pos + 41)
    w = df.iloc[a:b]

    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)

    axes[0].plot(w.index, w["SPY_AdjClose"])
    axes[0].set_ylabel("SPY")
    axes[0].set_title(
        f"Geometry Episode {int(ep['EpisodeID'])} — peak {peak.date()}"
    )

    axes[1].plot(w.index, w["GeometryStress"])
    axes[1].set_ylabel("Stress")

    axes[2].plot(w.index, w["TransitionScore"])
    axes[2].set_ylabel("Transition")

    axes[3].plot(w.index, w["SigmaMin"])
    axes[3].set_ylabel("Sigma min")

    axes[4].plot(w.index, w["WeakDirectionRotationDeg"])
    axes[4].set_ylabel("Rotation")
    axes[4].set_xlabel("Date")

    for ax in axes:
        ax.axvline(peak, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(
        f"market_geometry_episode_{int(ep['EpisodeID']):02d}.png",
        dpi=160,
    )
    plt.close(fig)


def main():
    df = pd.read_csv(STRESS_FILE, parse_dates=["Date"]).set_index("Date").sort_index()
    df = add_transition_metrics(df)
    episodes = identify_episodes(df)
    outcomes = episode_outcomes(df, episodes)

    df.to_csv(OUT_METRICS, float_format="%.10g")
    episodes.to_csv(OUT_EPISODES, index=False, float_format="%.10g")
    outcomes.to_csv(OUT_OUTCOMES, index=False, float_format="%.10g")

    plot_full(df, episodes)

    # Plot up to the 15 largest episodes by peak stress.
    if len(episodes):
        top = episodes.nlargest(15, "PeakGeometryStress")
        for _, ep in top.iterrows():
            plot_episode(df, ep)

    print(f"Saved: {OUT_METRICS.resolve()}")
    print(f"Saved: {OUT_EPISODES.resolve()}")
    print(f"Saved: {OUT_OUTCOMES.resolve()}")
    print("Saved: market_geometry_transition_full.png")

    if len(episodes):
        print("\nTop geometry episodes by peak stress:")
        cols = [
            "EpisodeID", "StartDate", "EndDate", "PeakStressDate",
            "PeakGeometryStress", "PeakStressPercentile",
            "PeakTransitionDate", "PeakTransitionScore",
            "PeakTransitionPercentile", "MinSigmaMin",
            "MaxConditionNumber", "MaxRotationDeg"
        ]
        print(
            episodes.nlargest(15, "PeakGeometryStress")[cols]
            .to_string(index=False)
        )

    if len(outcomes):
        print("\nEpisode outcomes:")
        show_cols = [
            "EpisodeID", "PeakStressDate",
            "MaxTransitionPctPrior5D", "MaxTransitionPctPrior10D",
            "MaxTransitionPctPrior20D",
            "SPY_ReturnAfter5D", "SPY_ReturnAfter10D",
            "SPY_ReturnAfter20D", "SPY_WorstAfter20D"
        ]
        print(outcomes[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
