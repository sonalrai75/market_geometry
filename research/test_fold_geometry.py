#!/usr/bin/env python
"""
Local fold-like geometry test for robust Market Geometry precursor events.

Purpose
-------
For each cross-window consensus event, reconstruct the local rolling Jacobian,
compute its weakest singular directions, and examine the local response surface
in those projected coordinates.

For each event/window:
    J = U S V^T

Define:
    q = standardized market-state coordinates projected on v_min
    r = standardized response coordinates projected on u_min

Then fit:
    linear:    r = a + b q
    quadratic: r = a + b q + c q^2

A classical fold normal form is locally quadratic, so we test whether:
1. the quadratic fit materially improves over the linear fit
2. the fitted derivative dr/dq changes sign inside the observed neighborhood
3. the quadratic vertex lies inside the observed q-range
4. the weakest active singular value is small relative to the largest one

This is a fold-LIKE diagnostic only. A statistical market map is not the same
thing as a known physical equilibrium manifold, so these tests do not prove a
Thom catastrophe.

Inputs
------
market_geometry_dataset.csv
window_consensus_precursors.csv

Outputs
-------
fold_geometry_results.csv
fold_geometry_summary.csv
fold_geometry_<date>_W<window>.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path("market_geometry_dataset.csv")
CONSENSUS = Path("window_consensus_precursors.csv")

RIDGE = 1e-3
HORIZON = 5
LOCAL_HALF_WINDOW = 20  # trading days before/after event

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

WINDOW_FLAGS = {
    63: "DetectedBy63",
    126: "DetectedBy126",
    252: "DetectedBy252",
}


def safe_r2(y, yhat):
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 3:
        return np.nan
    yy = y[mask]
    yh = yhat[mask]
    ss_res = np.sum((yy - yh) ** 2)
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def reconstruct_local_geometry(df, event_pos, window):
    train_end = event_pos - HORIZON
    train_start = train_end - window + 1
    if train_start < 0:
        return None

    train = df.iloc[train_start:train_end + 1][XCOLS + YCOLS].dropna()
    if len(train) < int(window * 0.80):
        return None

    Xraw = train[XCOLS].to_numpy(float)
    Yraw = train[YCOLS].to_numpy(float)

    xmu = np.nanmean(Xraw, axis=0)
    xsd = np.nanstd(Xraw, axis=0, ddof=1)
    ymu = np.nanmean(Yraw, axis=0)
    ysd = np.nanstd(Yraw, axis=0, ddof=1)

    if np.any(xsd == 0) or np.any(ysd == 0):
        return None

    X = (Xraw - xmu) / xsd
    Y = (Yraw - ymu) / ysd

    p = X.shape[1]
    B = np.linalg.solve(X.T @ X + RIDGE * np.eye(p), X.T @ Y)
    J = B.T

    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    umin = U[:, -1]
    vmin = Vt[-1]

    # Local neighborhood around the event.
    a = max(0, event_pos - LOCAL_HALF_WINDOW)
    b = min(len(df), event_pos + LOCAL_HALF_WINDOW + 1)
    local = df.iloc[a:b][XCOLS + YCOLS].dropna().copy()

    Xloc = (local[XCOLS].to_numpy(float) - xmu) / xsd
    Yloc = (local[YCOLS].to_numpy(float) - ymu) / ysd

    q = Xloc @ vmin
    r = Yloc @ umin

    return {
        "local": local,
        "q": q,
        "r": r,
        "s": s,
        "umin": umin,
        "vmin": vmin,
        "event_local_index": np.where(local.index == df.index[event_pos])[0],
    }


def fit_models(q, r):
    mask = np.isfinite(q) & np.isfinite(r)
    q = q[mask]
    r = r[mask]

    if len(q) < 10 or np.nanstd(q) == 0:
        return None

    # Linear fit.
    lin = np.polyfit(q, r, 1)
    rlin = np.polyval(lin, q)
    r2_lin = safe_r2(r, rlin)

    # Quadratic fit.
    quad = np.polyfit(q, r, 2)
    rquad = np.polyval(quad, q)
    r2_quad = safe_r2(r, rquad)

    c, b, a = quad

    qmin = float(np.nanmin(q))
    qmax = float(np.nanmax(q))

    if abs(c) > 1e-12:
        q_vertex = -b / (2 * c)
        vertex_in_range = qmin <= q_vertex <= qmax
        deriv_min = 2 * c * qmin + b
        deriv_max = 2 * c * qmax + b
        derivative_sign_change = deriv_min * deriv_max < 0
    else:
        q_vertex = np.nan
        vertex_in_range = False
        derivative_sign_change = False

    return {
        "linear_coef": lin,
        "quadratic_coef": quad,
        "R2Linear": r2_lin,
        "R2Quadratic": r2_quad,
        "R2Gain": r2_quad - r2_lin if np.isfinite(r2_lin) and np.isfinite(r2_quad) else np.nan,
        "QuadraticCoefficient": c,
        "VertexQ": q_vertex,
        "VertexInObservedRange": vertex_in_range,
        "DerivativeSignChange": derivative_sign_change,
        "QMin": qmin,
        "QMax": qmax,
        "n": len(q),
    }


def fold_flag(result):
    """
    Conservative heuristic flag, not a theorem:
    - quadratic improves R2 by at least 0.10
    - vertex lies inside observed range
    - derivative changes sign across observed range
    - sigma_min / sigma_max <= 0.25
    """
    return bool(
        np.isfinite(result["R2Gain"])
        and result["R2Gain"] >= 0.10
        and result["VertexInObservedRange"]
        and result["DerivativeSignChange"]
        and result["SigmaRatio"] <= 0.25
    )


def plot_case(event_date, window, q, r, fit, outfile):
    order = np.argsort(q)
    qsort = q[order]

    lin = np.polyval(fit["linear_coef"], qsort)
    quad = np.polyval(fit["quadratic_coef"], qsort)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(q, r, s=35, alpha=0.8, label="Observed local trajectory")
    ax.plot(qsort, lin, label=f"Linear fit R²={fit['R2Linear']:.3f}")
    ax.plot(qsort, quad, label=f"Quadratic fit R²={fit['R2Quadratic']:.3f}")

    if np.isfinite(fit["VertexQ"]):
        ax.axvline(fit["VertexQ"], linestyle="--", alpha=0.5, label="Quadratic vertex")

    ax.set_xlabel("q = projection on weakest input direction")
    ax.set_ylabel("r = projection on weakest output direction")
    ax.set_title(f"Fold-like Local Geometry — {event_date} — W{window}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date").sort_index()
    cons = pd.read_csv(CONSENSUS, parse_dates=["ConsensusDate"])

    # Primary analysis: all 2-of-3 consensus events.
    targets = cons[cons["WindowsDetecting"] >= 2].copy()

    # Also include severe one-window downside benchmarks for comparison.
    severe = cons[
        (cons["WindowsDetecting"] == 1)
        & (cons["MeanSPYWorstAfter20D"] <= -0.05)
    ].copy()

    targets = pd.concat([targets, severe], ignore_index=True)
    targets = targets.drop_duplicates(subset=["ConsensusDate"])

    rows = []

    for _, ev in targets.iterrows():
        event_date = pd.Timestamp(ev["ConsensusDate"])

        # Nearest trading day at or before consensus date.
        eligible = df.index[df.index <= event_date]
        if len(eligible) == 0:
            continue
        trading_date = eligible[-1]
        event_pos = df.index.get_loc(trading_date)

        for window, flagcol in WINDOW_FLAGS.items():
            if int(ev.get(flagcol, 0)) != 1:
                continue

            geom = reconstruct_local_geometry(df, event_pos, window)
            if geom is None:
                continue

            fit = fit_models(geom["q"], geom["r"])
            if fit is None:
                continue

            s = geom["s"]
            sigma_ratio = s[-1] / s[0] if s[0] > 0 else np.nan

            row = {
                "ConsensusDate": event_date,
                "TradingDate": trading_date,
                "Window": window,
                "WindowsDetectingConsensusEvent": int(ev["WindowsDetecting"]),
                "SigmaMax": s[0],
                "SigmaMin": s[-1],
                "SigmaRatio": sigma_ratio,
                "ConditionNumber": s[0] / s[-1] if s[-1] > 0 else np.nan,
                "R2Linear": fit["R2Linear"],
                "R2Quadratic": fit["R2Quadratic"],
                "R2Gain": fit["R2Gain"],
                "QuadraticCoefficient": fit["QuadraticCoefficient"],
                "VertexQ": fit["VertexQ"],
                "VertexInObservedRange": fit["VertexInObservedRange"],
                "DerivativeSignChange": fit["DerivativeSignChange"],
                "LocalObservations": fit["n"],
                "MeanSPYReturnAfter20D": ev["MeanSPYReturnAfter20D"],
                "MeanSPYWorstAfter20D": ev["MeanSPYWorstAfter20D"],
            }

            row["FoldLikeFlag"] = fold_flag(row)
            rows.append(row)

            outfile = f"fold_geometry_{event_date.strftime('%Y%m%d')}_W{window}.png"
            plot_case(
                event_date.strftime("%Y-%m-%d"),
                window,
                geom["q"],
                geom["r"],
                fit,
                outfile,
            )

    results = pd.DataFrame(rows)
    results.to_csv("fold_geometry_results.csv", index=False, float_format="%.10g")

    if len(results):
        summary = (
            results.groupby("ConsensusDate")
            .agg(
                WindowsTested=("Window", "count"),
                FoldLikeWindows=("FoldLikeFlag", "sum"),
                MeanSigmaRatio=("SigmaRatio", "mean"),
                MeanR2Linear=("R2Linear", "mean"),
                MeanR2Quadratic=("R2Quadratic", "mean"),
                MeanR2Gain=("R2Gain", "mean"),
                AnyDerivativeSignChange=("DerivativeSignChange", "max"),
                AnyVertexInRange=("VertexInObservedRange", "max"),
                MeanSPYReturnAfter20D=("MeanSPYReturnAfter20D", "mean"),
                MeanSPYWorstAfter20D=("MeanSPYWorstAfter20D", "mean"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()

    summary.to_csv("fold_geometry_summary.csv", index=False, float_format="%.10g")

    print("\nFold-like geometry results:")
    if len(results):
        show = [
            "ConsensusDate", "Window", "SigmaRatio", "ConditionNumber",
            "R2Linear", "R2Quadratic", "R2Gain",
            "VertexInObservedRange", "DerivativeSignChange",
            "FoldLikeFlag",
            "MeanSPYWorstAfter20D",
        ]
        print(results[show].to_string(index=False))

    print("\nEvent summary:")
    if len(summary):
        print(summary.to_string(index=False))

    print("\nSaved fold_geometry_results.csv, fold_geometry_summary.csv, and case plots.")


if __name__ == "__main__":
    main()
