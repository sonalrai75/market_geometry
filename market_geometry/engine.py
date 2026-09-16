from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WINDOWS, HORIZON, RIDGE, XCOLS, YCOLS, FRIENDLY


def _standardize(a: np.ndarray) -> np.ndarray:
    mean = np.nanmean(a, axis=0)
    std = np.nanstd(a, axis=0, ddof=1)
    std[std == 0] = np.nan
    return (a - mean) / std


def _jacobian(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Y ≈ X B, therefore J = B^T maps standardized market state to
    # standardized response coordinates.
    p = X.shape[1]
    B = np.linalg.solve(X.T @ X + RIDGE * np.eye(p), X.T @ Y)
    return B.T


def _principal_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    # v and -v are the same singular direction.
    cosine = np.clip(abs(float(np.dot(v1, v2))), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _single_date_svd(use: pd.DataFrame, i: int, window: int):
    train_end = i - HORIZON
    train_start = train_end - window + 1
    if train_start < 0:
        return None

    train = use.iloc[train_start:train_end + 1].dropna()
    if len(train) < int(window * 0.80):
        return None

    X = _standardize(train[XCOLS].to_numpy(float))
    Y = _standardize(train[YCOLS].to_numpy(float))

    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X, Y = X[mask], Y[mask]

    if len(X) < max(50, int(window * 0.50)):
        return None

    J = _jacobian(X, Y)
    _, singular_values, Vt = np.linalg.svd(J, full_matrices=False)

    vmin = Vt[-1]
    squared = vmin ** 2
    contribution = squared / squared.sum()

    return {
        "singular_values": singular_values,
        "vmin": vmin,
        "contribution": contribution,
    }


def build_history(df: pd.DataFrame, window: int) -> pd.DataFrame:
    use = df[XCOLS + YCOLS].replace([np.inf, -np.inf], np.nan)

    rows = []
    previous_vmin = None

    for i in range(window - 1 + HORIZON, len(use)):
        result = _single_date_svd(use, i, window)
        if result is None:
            continue

        s = result["singular_values"]
        vmin = result["vmin"]

        rotation = (
            np.nan
            if previous_vmin is None
            else _principal_angle(previous_vmin, vmin)
        )
        previous_vmin = vmin

        row = {
            "Date": use.index[i],
            "SigmaMin": float(s[-1]),
            "SigmaMax": float(s[0]),
            "SigmaRatio": float(s[-1] / s[0]) if s[0] > 0 else np.nan,
            "ConditionNumber": float(s[0] / s[-1]) if s[-1] > 0 else np.inf,
            "RotationDeg": rotation,
        }

        for variable, value in zip(XCOLS, result["contribution"]):
            row[f"Contribution_{variable}"] = float(value)

        rows.append(row)

    if not rows:
        raise RuntimeError(f"No usable SVD history for {window}-day window.")

    return pd.DataFrame(rows).set_index("Date")


def _percentile(reference: pd.Series, current: float) -> float:
    ref = reference.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ref) == 0 or not np.isfinite(current):
        return np.nan
    return float(100.0 * (ref < current).mean())


def _fraction(series: pd.Series, rising: bool, sessions: int = 10) -> float:
    recent = series.dropna().tail(sessions + 1)
    if len(recent) < 3:
        return np.nan

    delta = recent.diff().dropna()
    return float((delta > 0).mean() if rising else (delta < 0).mean())


def _classify(metrics: dict) -> tuple[str, list[str], int]:
    score = 0
    reasons: list[str] = []

    # Low sigma ratio = closer to active degeneracy.
    if metrics["sigma_ratio_percentile"] <= 10:
        score += 2
        reasons.append("sigma-min / sigma-max is in the bottom 10% of its trailing-year range")
    elif metrics["sigma_ratio_percentile"] <= 20:
        score += 1
        reasons.append("the active singular-value ratio is below its normal range")

    if metrics["condition_percentile"] >= 95:
        score += 2
        reasons.append("condition number is above the 95th trailing-year percentile")
    elif metrics["condition_percentile"] >= 85:
        score += 1
        reasons.append("condition number is elevated")

    if metrics["rotation_percentile"] >= 95:
        score += 2
        reasons.append("weak-direction rotation is above the 95th trailing-year percentile")
    elif metrics["rotation_percentile"] >= 85:
        score += 1
        reasons.append("weak-direction rotation is elevated")

    if metrics["sigma_decline_fraction_10d"] >= 0.70:
        score += 1
        reasons.append("sigma-min has declined on most of the last 10 sessions")

    if metrics["condition_rise_fraction_10d"] >= 0.70:
        score += 1
        reasons.append("conditioning has deteriorated on most of the last 10 sessions")

    if score >= 6:
        status = "HIGH DEGENERACY"
    elif score >= 4:
        status = "APPROACHING DEGENERACY"
    elif score >= 2:
        status = "GEOMETRY SHIFT"
    else:
        status = "NORMAL"

    if not reasons:
        reasons = ["geometry remains within its recent historical range"]

    return status, reasons, score


def analyze_window(df: pd.DataFrame, window: int) -> dict:
    history = build_history(df, window)

    current = history.iloc[-1]
    reference = history.iloc[max(0, len(history) - 253):-1]

    metrics = {
        "window": window,
        "date": history.index[-1].strftime("%Y-%m-%d"),
        "sigma_min": float(current["SigmaMin"]),
        "sigma_max": float(current["SigmaMax"]),
        "sigma_ratio": float(current["SigmaRatio"]),
        "condition_number": float(current["ConditionNumber"]),
        "rotation_deg": (
            None if pd.isna(current["RotationDeg"])
            else float(current["RotationDeg"])
        ),
        "sigma_ratio_percentile": _percentile(
            reference["SigmaRatio"], current["SigmaRatio"]
        ),
        "condition_percentile": _percentile(
            reference["ConditionNumber"], current["ConditionNumber"]
        ),
        "rotation_percentile": _percentile(
            reference["RotationDeg"], current["RotationDeg"]
        ),
        "sigma_decline_fraction_10d": _fraction(
            history["SigmaMin"], rising=False
        ),
        "condition_rise_fraction_10d": _fraction(
            history["ConditionNumber"], rising=True
        ),
    }

    contributions = [
        {
            "key": variable,
            "name": FRIENDLY[variable],
            "value": float(current[f"Contribution_{variable}"]),
        }
        for variable in XCOLS
    ]
    contributions.sort(key=lambda item: item["value"], reverse=True)
    metrics["contributions"] = contributions

    status, reasons, score = _classify(metrics)
    metrics["status"] = status
    metrics["reasons"] = reasons
    metrics["degeneracy_score"] = score

    recent = history.tail(252).reset_index()
    metrics["history"] = [
        {
            "date": row["Date"].strftime("%Y-%m-%d"),
            "sigma_ratio": float(row["SigmaRatio"]),
            "condition": float(row["ConditionNumber"]),
            "rotation": (
                None if pd.isna(row["RotationDeg"])
                else float(row["RotationDeg"])
            ),
        }
        for _, row in recent.iterrows()
    ]

    return metrics


def build_dashboard(df: pd.DataFrame) -> dict:
    windows = [analyze_window(df, window) for window in WINDOWS]

    severity = {
        "NORMAL": 0,
        "GEOMETRY SHIFT": 1,
        "APPROACHING DEGENERACY": 2,
        "HIGH DEGENERACY": 3,
    }

    confirming = sum(
        severity[item["status"]] >= 2
        for item in windows
    )
    shifting_or_worse = sum(
        severity[item["status"]] >= 1
        for item in windows
    )

    highest = max(windows, key=lambda item: severity[item["status"]])

    if confirming >= 2:
        overall = (
            "HIGH DEGENERACY"
            if highest["status"] == "HIGH DEGENERACY"
            else "APPROACHING DEGENERACY"
        )
    elif confirming == 1 or shifting_or_worse >= 2:
        overall = "GEOMETRY SHIFT"
    else:
        overall = "NORMAL"

    aggregate: dict[str, list[float]] = {}
    for item in windows:
        for contribution in item["contributions"]:
            aggregate.setdefault(contribution["name"], []).append(
                contribution["value"]
            )

    aggregate_contributions = [
        {"name": name, "value": float(np.mean(values))}
        for name, values in aggregate.items()
    ]
    aggregate_contributions.sort(key=lambda item: item["value"], reverse=True)

    leaders = aggregate_contributions[:3]
    leader_text = ", ".join(
        f'{item["name"]} ({item["value"]:.0%})'
        for item in leaders
    )

    if overall == "NORMAL":
        interpretation = (
            "The estimated market-response geometry is currently within its "
            "recent historical range. The dominant weak-direction components "
            f"are {leader_text}."
        )
    elif overall == "GEOMETRY SHIFT":
        interpretation = (
            "One or more time scales show an unusual change in conditioning "
            "or the weak SVD direction. The current weak-direction composition "
            f"is led by {leader_text}. This indicates a geometry/regime shift, "
            "not a directional market forecast."
        )
    else:
        interpretation = (
            "The estimated response map is showing signs of local degeneracy "
            "on one or more time scales through singular-value contraction, "
            "deteriorating conditioning, and/or rapid weak-direction rotation. "
            f"The current weak-direction composition is led by {leader_text}. "
            "This is a structural warning, not evidence that a crash or a "
            "classical Thom catastrophe must follow."
        )

    return {
        "as_of": max(item["date"] for item in windows),
        "overall_status": overall,
        "cross_window_confirmation": f"{confirming} / {len(windows)}",
        "windows": windows,
        "aggregate_contributions": aggregate_contributions,
        "interpretation": interpretation,
        "research_note": (
            "Degeneracy flags describe the estimated local response geometry. "
            "They are not trading signals, probability forecasts, or proof of "
            "a classical fold/cusp catastrophe."
        ),
    }
