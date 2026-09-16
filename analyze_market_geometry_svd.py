#!/usr/bin/env python
"""
Prototype evolving-SVD analysis for the Market Geometry dataset.

Method:
1. Standardize inputs and outputs.
2. On each rolling window, fit a local multivariate ridge model:
      Y ~= X B
   so the local Jacobian estimate is J = B^T.
3. Compute SVD(J) and track:
      singular values,
      condition number,
      weakest input-space direction,
      rotation of that direction through time.
4. Export diagnostics for later fold/cusp investigation.

This is a first empirical geometry diagnostic, not proof of a Thom catastrophe.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("market_geometry_dataset.csv")
OUT = Path("market_geometry_svd_diagnostics.csv")

WINDOW = 126       # about 6 months
RIDGE = 1.0e-3

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

def zscore(a):
    mu = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0, ddof=1)
    sd[sd == 0] = np.nan
    return (a - mu) / sd

def principal_angle(v1, v2):
    # Singular-vector sign is arbitrary, so use abs(dot).
    c = np.clip(abs(float(np.dot(v1, v2))), 0.0, 1.0)
    return np.degrees(np.arccos(c))

def local_jacobian_ridge(X, Y, lam=RIDGE):
    # X: n x p, Y: n x q; B: p x q; J: q x p
    p = X.shape[1]
    B = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ Y)
    return B.T

def main():
    df = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date").sort_index()
    use = df[XCOLS + YCOLS].replace([np.inf, -np.inf], np.nan)

    rows = []
    prev_vmin = None

    for i in range(WINDOW - 1, len(use)):
        w = use.iloc[i-WINDOW+1:i+1].dropna()
        if len(w) < int(WINDOW * 0.80):
            continue

        X = zscore(w[XCOLS].to_numpy(float))
        Y = zscore(w[YCOLS].to_numpy(float))
        mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X, Y = X[mask], Y[mask]
        if len(X) < 50:
            continue

        J = local_jacobian_ridge(X, Y)
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        positive = s[s > 1e-12]
        cond = positive.max() / positive.min() if len(positive) else np.nan

        vmin = Vt[-1].copy()
        rotation = np.nan if prev_vmin is None else principal_angle(prev_vmin, vmin)
        prev_vmin = vmin

        row = {
            "Date": use.index[i],
            "SigmaMax": s[0],
            "SigmaMin": s[-1],
            "ConditionNumber": cond,
            "WeakDirectionRotationDeg": rotation,
        }
        for k, val in enumerate(s, 1):
            row[f"Sigma{k}"] = val
        for col, val in zip(XCOLS, vmin):
            row[f"Vmin_{col}"] = val
        rows.append(row)

    out = pd.DataFrame(rows).set_index("Date")
    out.to_csv(OUT, float_format="%.10g")
    print(f"Saved {len(out):,} diagnostic rows to {OUT.resolve()}")

    if len(out):
        print("\nLargest weak-direction rotations:")
        print(out.nlargest(10, "WeakDirectionRotationDeg")[
            ["SigmaMin", "ConditionNumber", "WeakDirectionRotationDeg"]
        ].to_string())

        print("\nSmallest singular values:")
        print(out.nsmallest(10, "SigmaMin")[
            ["SigmaMin", "ConditionNumber", "WeakDirectionRotationDeg"]
        ].to_string())

if __name__ == "__main__":
    main()
