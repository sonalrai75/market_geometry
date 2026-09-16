#!/usr/bin/env python
"""
Cluster PRECURSOR market-geometry events by weakest SVD direction.

Why this matters
----------------
The right singular vector v_min identifies the combination of market-state
variables defining the weakest active direction of the local response map.

Because a singular vector has arbitrary sign (v and -v are equivalent),
clustering raw signed coefficients is inappropriate. This script therefore
clusters on squared loadings:

    contribution_j = v_j^2

For a unit singular vector, these contributions sum to ~1 and can be read as
the fractional composition of the weak direction.

Inputs
------
market_geometry_precursor_events.csv
market_geometry_svd_diagnostics.csv

Outputs
-------
market_geometry_precursor_clusters.csv
market_geometry_cluster_profiles.csv
market_geometry_cluster_outcomes.csv
market_geometry_cluster_k_selection.csv
market_geometry_cluster_pca.png
market_geometry_cluster_profiles.png

This is exploratory: there are only a small number of PRECURSOR events.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

PRECURSOR_FILE = Path("market_geometry_precursor_events.csv")
SVD_FILE = Path("market_geometry_svd_diagnostics.csv")

OUT_EVENTS = Path("market_geometry_precursor_clusters.csv")
OUT_PROFILES = Path("market_geometry_cluster_profiles.csv")
OUT_OUTCOMES = Path("market_geometry_cluster_outcomes.csv")
OUT_K = Path("market_geometry_cluster_k_selection.csv")

RANDOM_STATE = 42

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

V_COLS = [f"Vmin_{v}" for v in VARS]


def canonicalize_sign(v: np.ndarray) -> np.ndarray:
    """
    For reporting signed coefficients only:
    flip the vector so its largest-magnitude component is positive.
    Clustering itself uses v^2 and is sign-invariant.
    """
    if not np.isfinite(v).all():
        return v
    j = int(np.argmax(np.abs(v)))
    return -v if v[j] < 0 else v


def choose_k(X: np.ndarray) -> tuple[int, pd.DataFrame]:
    n = len(X)
    max_k = min(5, n - 1)

    if max_k < 2:
        raise RuntimeError("Need at least 3 precursor events for clustering.")

    rows = []
    best_k = None
    best_score = -np.inf

    for k in range(2, max_k + 1):
        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=50,
        )
        labels = km.fit_predict(X)

        # Silhouette requires >1 cluster and fewer clusters than observations.
        score = silhouette_score(X, labels)

        rows.append({
            "K": k,
            "SilhouetteScore": score,
            "Inertia": km.inertia_,
        })

        if score > best_score:
            best_score = score
            best_k = k

    return int(best_k), pd.DataFrame(rows)


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    prec = pd.read_csv(PRECURSOR_FILE, parse_dates=["Date"])
    prec = prec[prec["StateClass"] == "PRECURSOR"].copy()

    svd = pd.read_csv(SVD_FILE, parse_dates=["Date"])

    keep = ["Date"] + V_COLS
    merged = prec.merge(svd[keep], on="Date", how="left", validate="one_to_one")

    missing = merged[V_COLS].isna().any(axis=1)
    if missing.any():
        print("Warning: dropping precursor events without complete Vmin data:")
        print(merged.loc[missing, ["EventID", "Date"]].to_string(index=False))
        merged = merged.loc[~missing].copy()

    # Canonicalized signed loadings for human-readable reporting.
    signed = merged[V_COLS].to_numpy(float)
    signed = np.vstack([canonicalize_sign(v) for v in signed])

    signed_cols = []
    for j, var in enumerate(VARS):
        col = f"Signed_{var}"
        merged[col] = signed[:, j]
        signed_cols.append(col)

    # Sign-invariant contributions. Normalize defensively in case numerical
    # rounding makes ||v|| differ slightly from 1.
    sq = signed ** 2
    row_sum = sq.sum(axis=1, keepdims=True)
    contrib = sq / row_sum

    contrib_cols = []
    for j, var in enumerate(VARS):
        col = f"Contribution_{var}"
        merged[col] = contrib[:, j]
        contrib_cols.append(col)

    return merged, contrib_cols


def summarize_profiles(df: pd.DataFrame, contrib_cols: list[str]) -> pd.DataFrame:
    rows = []

    for cluster, g in df.groupby("Cluster"):
        means = g[contrib_cols].mean()

        # Top three drivers.
        top = means.sort_values(ascending=False).head(3)
        row = {
            "Cluster": int(cluster),
            "Events": len(g),
            "TopDriver1": top.index[0].replace("Contribution_", ""),
            "TopDriver1MeanContribution": top.iloc[0],
            "TopDriver2": top.index[1].replace("Contribution_", ""),
            "TopDriver2MeanContribution": top.iloc[1],
            "TopDriver3": top.index[2].replace("Contribution_", ""),
            "TopDriver3MeanContribution": top.iloc[2],
        }

        for col in contrib_cols:
            row[col] = means[col]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("Cluster")


def summarize_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cluster, g in df.groupby("Cluster"):
        for h in (5, 10, 20, 40):
            ret_col = f"SPY_ReturnAfter{h}D"
            worst_col = f"SPY_WorstAfter{h}D"

            rows.append({
                "Cluster": int(cluster),
                "HorizonDays": h,
                "Events": len(g),
                "MeanForwardReturn": g[ret_col].mean(),
                "MedianForwardReturn": g[ret_col].median(),
                "MeanWorstReturn": g[worst_col].mean(),
                "MedianWorstReturn": g[worst_col].median(),
                "PctNegativeForwardReturn": (g[ret_col] < 0).mean(),
                "PctWorstBelowMinus5Pct": (g[worst_col] <= -0.05).mean(),
                "PctWorstBelowMinus10Pct": (g[worst_col] <= -0.10).mean(),
            })

    return pd.DataFrame(rows).sort_values(["Cluster", "HorizonDays"])


def plot_pca(df: pd.DataFrame, X: np.ndarray):
    pca = PCA(n_components=2)
    z = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Let matplotlib choose colors automatically; no explicit color styling.
    for cluster in sorted(df["Cluster"].unique()):
        mask = df["Cluster"].to_numpy() == cluster
        ax.scatter(
            z[mask, 0],
            z[mask, 1],
            label=f"Cluster {cluster}",
            s=70,
        )

        for x, y, date in zip(
            z[mask, 0],
            z[mask, 1],
            df.loc[mask, "Date"],
        ):
            ax.annotate(
                pd.Timestamp(date).strftime("%Y-%m-%d"),
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Precursor Weak-Direction Geometry")
    ax.legend()
    fig.tight_layout()
    fig.savefig("market_geometry_cluster_pca.png", dpi=160)
    plt.close(fig)


def plot_profiles(profiles: pd.DataFrame, contrib_cols: list[str]):
    plot_df = profiles.set_index("Cluster")[contrib_cols].copy()
    plot_df.columns = [c.replace("Contribution_", "") for c in plot_df.columns]

    ax = plot_df.T.plot(kind="bar", figsize=(13, 7))
    ax.set_ylabel("Mean squared loading contribution")
    ax.set_title("Weak-Direction Composition by Precursor Cluster")
    ax.legend(title="Cluster")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("market_geometry_cluster_profiles.png", dpi=160)
    plt.close()


def main():
    df, contrib_cols = build_dataset()

    X = df[contrib_cols].to_numpy(float)

    best_k, k_table = choose_k(X)
    k_table.to_csv(OUT_K, index=False, float_format="%.10g")

    print("\nK selection:")
    print(k_table.to_string(index=False))
    print(f"\nSelected K = {best_k} by maximum silhouette score")

    km = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=100,
    )
    labels = km.fit_predict(X)

    # Human-friendly labels 1..K rather than 0..K-1.
    df["Cluster"] = labels + 1

    profiles = summarize_profiles(df, contrib_cols)
    outcomes = summarize_outcomes(df)

    df.to_csv(OUT_EVENTS, index=False, float_format="%.10g")
    profiles.to_csv(OUT_PROFILES, index=False, float_format="%.10g")
    outcomes.to_csv(OUT_OUTCOMES, index=False, float_format="%.10g")

    plot_pca(df, X)
    plot_profiles(profiles, contrib_cols)

    print(f"\nSaved: {OUT_EVENTS.resolve()}")
    print(f"Saved: {OUT_PROFILES.resolve()}")
    print(f"Saved: {OUT_OUTCOMES.resolve()}")
    print(f"Saved: {OUT_K.resolve()}")
    print("Saved: market_geometry_cluster_pca.png")
    print("Saved: market_geometry_cluster_profiles.png")

    print("\nCluster profiles:")
    show = [
        "Cluster", "Events",
        "TopDriver1", "TopDriver1MeanContribution",
        "TopDriver2", "TopDriver2MeanContribution",
        "TopDriver3", "TopDriver3MeanContribution",
    ]
    print(profiles[show].to_string(index=False))

    print("\nPrecursor events by cluster:")
    cols = [
        "EventID", "Date", "Cluster",
        "TransitionPercentile", "GeometryStressPercentile",
        "SPY_ReturnAfter20D", "SPY_WorstAfter20D",
    ]
    print(df[cols].sort_values(["Cluster", "Date"]).to_string(index=False))

    print("\nCluster outcome summary:")
    print(outcomes.to_string(index=False))


if __name__ == "__main__":
    main()
