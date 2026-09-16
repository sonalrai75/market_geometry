#!/usr/bin/env python
"""
Robustness study for Market Geometry precursor clustering.

What this tests
---------------
Recompute the rolling market-geometry SVD over several:
- rolling windows: 63, 126, 252 trading days
- ridge penalties: 1e-4, 1e-3, 1e-2

For each specification:
1. estimate leakage-safe rolling Jacobians
2. compute SVD diagnostics
3. build geometry stress
4. build transition score
5. identify PRECURSOR events
6. cluster PRECURSOR weak-direction compositions
7. compare resulting event/cluster structure across specifications

Outputs
-------
robustness_spec_summary.csv
robustness_precursor_events.csv
robustness_cluster_profiles.csv
robustness_cluster_outcomes.csv
robustness_event_recurrence.csv
robustness_driver_recurrence.csv
robustness_summary.png

Notes
-----
This is exploratory robustness analysis, not proof of a Thom catastrophe.
"""

from __future__ import annotations

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA = Path("market_geometry_dataset.csv")

WINDOWS = [63, 126, 252]
RIDGES = [1e-4, 1e-3, 1e-2]
HORIZON = 5

XCOLS = [
    "VIX",
    "Treasury10Y",
    "Curve10Y2Y",
    "CreditSpread_BAA10Y",
    "WTI",
    "BroadUSD",
    "RSP_SPY_Ratio",
    "SPY_Volume_Z20",
]

YCOLS = [
    "SPY_Fwd5D_Return",
    "QQQ_Fwd5D_Return",
    "SPY_RVol20",
    "SPY_Drawdown63",
]

VARS = XCOLS[:]

TRANSITION_PCT_THRESHOLD = 97.5
MIN_SEPARATION_DAYS = 20
RANDOM_STATE = 42


def zscore(a):
    mu = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0, ddof=1)
    sd[sd == 0] = np.nan
    return (a - mu) / sd


def principal_angle(v1, v2):
    c = np.clip(abs(float(np.dot(v1, v2))), 0.0, 1.0)
    return np.degrees(np.arccos(c))


def local_jacobian_ridge(X, Y, lam):
    p = X.shape[1]
    B = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ Y)
    return B.T


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)

    def f(a):
        if np.isnan(a).any():
            return np.nan
        return float(np.dot(x, a - a.mean()) / denom)

    return series.rolling(window, min_periods=window).apply(f, raw=True)


def trailing_z(s: pd.Series, baseline=252, minp=126):
    hist = s.shift(1)
    mu = hist.rolling(baseline, min_periods=minp).mean()
    sd = hist.rolling(baseline, min_periods=minp).std()
    return (s - mu) / sd.replace(0, np.nan)


def trailing_percentile(s: pd.Series, baseline=252, minp=126):
    vals = []
    for i in range(len(s)):
        if i < minp or not np.isfinite(s.iloc[i]):
            vals.append(np.nan)
            continue
        hist = s.iloc[max(0, i-baseline):i].dropna()
        if len(hist) < minp:
            vals.append(np.nan)
        else:
            vals.append(100.0 * (hist < s.iloc[i]).mean())
    return pd.Series(vals, index=s.index)


def build_svd(df, window, ridge):
    use = df[XCOLS + YCOLS].replace([np.inf, -np.inf], np.nan)

    rows = []
    prev_vmin = None
    first_i = window - 1 + HORIZON

    for i in range(first_i, len(use)):
        train_end = i - HORIZON
        train_start = train_end - window + 1
        w = use.iloc[train_start:train_end + 1].dropna()

        if len(w) < int(window * 0.80):
            continue

        X = zscore(w[XCOLS].to_numpy(float))
        Y = zscore(w[YCOLS].to_numpy(float))
        mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X, Y = X[mask], Y[mask]

        if len(X) < max(50, int(window * 0.5)):
            continue

        J = local_jacobian_ridge(X, Y, ridge)
        U, s, Vt = np.linalg.svd(J, full_matrices=False)

        positive = s[s > 1e-12]
        cond = positive.max() / positive.min() if len(positive) else np.nan

        vmin = Vt[-1].copy()
        rotation = np.nan if prev_vmin is None else principal_angle(prev_vmin, vmin)
        prev_vmin = vmin

        row = {
            "Date": use.index[i],
            "SigmaMin": s[-1],
            "ConditionNumber": cond,
            "WeakDirectionRotationDeg": rotation,
        }

        for col, val in zip(XCOLS, vmin):
            row[f"Vmin_{col}"] = val

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("Date")


def add_geometry_and_transition(market, svd):
    df = market[["SPY_AdjClose"]].join(svd, how="inner").copy()

    df["LogConditionNumber"] = np.log1p(df["ConditionNumber"])

    df["Z_LogCondition"] = trailing_z(df["LogConditionNumber"])
    df["Z_Rotation"] = trailing_z(df["WeakDirectionRotationDeg"])
    df["Z_NegSigmaMin"] = trailing_z(-df["SigmaMin"])

    df["GeometryStress"] = df[
        ["Z_LogCondition", "Z_Rotation", "Z_NegSigmaMin"]
    ].mean(axis=1, skipna=True)

    df["GeometryStressPercentile"] = trailing_percentile(df["GeometryStress"])

    df["StressVelocity5D"] = rolling_slope(df["GeometryStress"], 5)
    df["StressAcceleration"] = df["StressVelocity5D"].diff()
    df["SigmaMinSlope10D"] = rolling_slope(df["SigmaMin"], 10)
    df["LogConditionSlope10D"] = rolling_slope(df["LogConditionNumber"], 10)
    df["RotationSlope10D"] = rolling_slope(df["WeakDirectionRotationDeg"], 10)

    df["Z_StressVelocity5D"] = trailing_z(df["StressVelocity5D"])
    df["Z_StressAcceleration"] = trailing_z(df["StressAcceleration"])
    df["Z_NegSigmaSlope10D"] = trailing_z(-df["SigmaMinSlope10D"])
    df["Z_ConditionSlope10D"] = trailing_z(df["LogConditionSlope10D"])
    df["Z_RotationSlope10D"] = trailing_z(df["RotationSlope10D"])

    df["TransitionScore"] = df[
        [
            "Z_StressVelocity5D",
            "Z_StressAcceleration",
            "Z_NegSigmaSlope10D",
            "Z_ConditionSlope10D",
            "Z_RotationSlope10D",
        ]
    ].mean(axis=1, skipna=True)

    df["TransitionScorePercentile"] = trailing_percentile(df["TransitionScore"])

    peak20 = df["SPY_AdjClose"].rolling(20, min_periods=1).max()
    peak60 = df["SPY_AdjClose"].rolling(60, min_periods=1).max()
    df["SPY_Drawdown20D"] = df["SPY_AdjClose"] / peak20 - 1
    df["SPY_Drawdown60D"] = df["SPY_AdjClose"] / peak60 - 1

    return df


def select_precursors(df):
    cand = df[df["TransitionScorePercentile"] >= TRANSITION_PCT_THRESHOLD].copy()

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

    for pos in selected:
        r = df.iloc[pos]

        is_precursor = (
            r["GeometryStressPercentile"] < 95
            and r["SPY_Drawdown20D"] > -0.05
            and r["SPY_Drawdown60D"] > -0.08
        )

        if not is_precursor:
            continue

        rec = {
            "Date": df.index[pos],
            "TransitionPercentile": r["TransitionScorePercentile"],
            "GeometryStressPercentile": r["GeometryStressPercentile"],
            "SPY_Drawdown20D": r["SPY_Drawdown20D"],
            "SPY_Drawdown60D": r["SPY_Drawdown60D"],
        }

        px0 = r["SPY_AdjClose"]
        for h in (5, 10, 20, 40):
            if pos + h < len(df):
                future = df.iloc[pos+1:pos+h+1]["SPY_AdjClose"]
                rec[f"SPY_ReturnAfter{h}D"] = future.iloc[-1] / px0 - 1
                rec[f"SPY_WorstAfter{h}D"] = (future / px0 - 1).min()
            else:
                rec[f"SPY_ReturnAfter{h}D"] = np.nan
                rec[f"SPY_WorstAfter{h}D"] = np.nan

        v = np.array([r[f"Vmin_{c}"] for c in XCOLS], dtype=float)
        sq = v ** 2
        sq = sq / sq.sum()

        for var, val in zip(VARS, sq):
            rec[f"Contribution_{var}"] = val

        rows.append(rec)

    return pd.DataFrame(rows)


def choose_k(X):
    if len(X) < 4:
        return np.nan, np.nan

    max_k = min(5, len(X)-1)
    best_k = None
    best_score = -np.inf

    for k in range(2, max_k+1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score


def main():
    market = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date").sort_index()

    all_events = []
    spec_rows = []
    profile_rows = []
    outcome_rows = []

    for window, ridge in itertools.product(WINDOWS, RIDGES):
        spec = f"W{window}_R{ridge:g}"
        print(f"Running {spec}...")

        svd = build_svd(market, window, ridge)
        if svd.empty:
            continue

        df = add_geometry_and_transition(market, svd)
        prec = select_precursors(df)

        if len(prec) == 0:
            spec_rows.append({
                "Spec": spec,
                "Window": window,
                "Ridge": ridge,
                "PrecursorEvents": 0,
                "SelectedK": np.nan,
                "Silhouette": np.nan,
            })
            continue

        prec["Spec"] = spec
        prec["Window"] = window
        prec["Ridge"] = ridge

        contrib_cols = [f"Contribution_{v}" for v in VARS]
        X = prec[contrib_cols].to_numpy(float)

        k, sil = choose_k(X)

        spec_rows.append({
            "Spec": spec,
            "Window": window,
            "Ridge": ridge,
            "PrecursorEvents": len(prec),
            "SelectedK": k,
            "Silhouette": sil,
        })

        if np.isfinite(k):
            km = KMeans(
                n_clusters=int(k),
                random_state=RANDOM_STATE,
                n_init=100,
            )
            prec["Cluster"] = km.fit_predict(X) + 1

            for cluster, g in prec.groupby("Cluster"):
                means = g[contrib_cols].mean()
                top = means.sort_values(ascending=False).head(3)

                profile_rows.append({
                    "Spec": spec,
                    "Window": window,
                    "Ridge": ridge,
                    "Cluster": int(cluster),
                    "Events": len(g),
                    "TopDriver1": top.index[0].replace("Contribution_", ""),
                    "TopDriver1Contribution": top.iloc[0],
                    "TopDriver2": top.index[1].replace("Contribution_", ""),
                    "TopDriver2Contribution": top.iloc[1],
                    "TopDriver3": top.index[2].replace("Contribution_", ""),
                    "TopDriver3Contribution": top.iloc[2],
                })

                for h in (5, 10, 20, 40):
                    outcome_rows.append({
                        "Spec": spec,
                        "Window": window,
                        "Ridge": ridge,
                        "Cluster": int(cluster),
                        "HorizonDays": h,
                        "Events": len(g),
                        "MeanForwardReturn": g[f"SPY_ReturnAfter{h}D"].mean(),
                        "MeanWorstReturn": g[f"SPY_WorstAfter{h}D"].mean(),
                        "PctNegativeForwardReturn": (
                            g[f"SPY_ReturnAfter{h}D"] < 0
                        ).mean(),
                        "PctWorstBelowMinus5Pct": (
                            g[f"SPY_WorstAfter{h}D"] <= -0.05
                        ).mean(),
                    })
        else:
            prec["Cluster"] = np.nan

        all_events.append(prec)

    specs = pd.DataFrame(spec_rows)
    profiles = pd.DataFrame(profile_rows)
    outcomes = pd.DataFrame(outcome_rows)

    if all_events:
        events = pd.concat(all_events, ignore_index=True)
    else:
        events = pd.DataFrame()

    # Event recurrence across specifications.
    recurrence_rows = []
    if len(events):
        unique_dates = sorted(events["Date"].dropna().unique())

        for d in unique_dates:
            d = pd.Timestamp(d)
            hits = events[
                (events["Date"] >= d - pd.Timedelta(days=5))
                & (events["Date"] <= d + pd.Timedelta(days=5))
            ]

            recurrence_rows.append({
                "ApproxDate": d,
                "SpecificationsDetectingNearbyPrecursor": hits["Spec"].nunique(),
                "TotalSpecifications": specs["Spec"].nunique(),
                "RecurrenceFraction": (
                    hits["Spec"].nunique() / specs["Spec"].nunique()
                    if specs["Spec"].nunique() else np.nan
                ),
            })

    recurrence = pd.DataFrame(recurrence_rows)

    # Driver recurrence: how often each variable appears as a top-3 cluster driver.
    driver_rows = []
    if len(profiles):
        for var in VARS:
            count = (
                (profiles["TopDriver1"] == var).sum()
                + (profiles["TopDriver2"] == var).sum()
                + (profiles["TopDriver3"] == var).sum()
            )
            driver_rows.append({
                "Driver": var,
                "Top3Appearances": int(count),
                "ClusterProfiles": len(profiles),
                "AppearanceFraction": count / len(profiles),
            })

    driver_rec = pd.DataFrame(driver_rows).sort_values(
        "AppearanceFraction", ascending=False
    )

    specs.to_csv("robustness_spec_summary.csv", index=False, float_format="%.10g")
    events.to_csv("robustness_precursor_events.csv", index=False, float_format="%.10g")
    profiles.to_csv("robustness_cluster_profiles.csv", index=False, float_format="%.10g")
    outcomes.to_csv("robustness_cluster_outcomes.csv", index=False, float_format="%.10g")
    recurrence.to_csv("robustness_event_recurrence.csv", index=False, float_format="%.10g")
    driver_rec.to_csv("robustness_driver_recurrence.csv", index=False, float_format="%.10g")

    # Summary plot
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    if len(specs):
        x = np.arange(len(specs))
        axes[0].bar(x, specs["PrecursorEvents"])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(specs["Spec"], rotation=45, ha="right")
        axes[0].set_ylabel("Precursor events")
        axes[0].set_title("Robustness Across SVD Specifications")

        axes[1].bar(x, specs["Silhouette"])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(specs["Spec"], rotation=45, ha="right")
        axes[1].set_ylabel("Silhouette")

    if len(driver_rec):
        axes[2].bar(driver_rec["Driver"], driver_rec["AppearanceFraction"])
        axes[2].set_ylabel("Top-3 appearance fraction")
        axes[2].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig("robustness_summary.png", dpi=160)
    plt.close(fig)

    print("\nSpecification summary:")
    print(specs.to_string(index=False))

    print("\nDriver recurrence:")
    print(driver_rec.to_string(index=False))

    if len(recurrence):
        print("\nMost recurrent precursor dates:")
        print(
            recurrence.nlargest(20, "RecurrenceFraction")
            .to_string(index=False)
        )

    print("\nSaved robustness CSV files and robustness_summary.png")


if __name__ == "__main__":
    main()
