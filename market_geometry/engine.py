from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WINDOWS, HORIZON, RIDGE, XCOLS, YCOLS, FRIENDLY


SEVERITY = {
    "NORMAL": 0,
    "GEOMETRY SHIFT": 1,
    "APPROACHING DEGENERACY": 2,
    "HIGH DEGENERACY": 3,
}

DRIVER_GROUPS = {
    "USD": ["BroadUSD"],
    "Rates": ["Treasury10Y", "Curve10Y2Y"],
    "Credit": ["CreditSpread_BAA10Y"],
    "Energy": ["WTI"],
    "Equities": ["RSP_SPY_Ratio", "SPY_Volume_Z20"],
    "Volatility": ["VIX"],
}


def _standardize(a: np.ndarray) -> np.ndarray:
    mean = np.nanmean(a, axis=0)
    std = np.nanstd(a, axis=0, ddof=1)
    std[std == 0] = np.nan
    return (a - mean) / std


def _jacobian(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    p = X.shape[1]
    B = np.linalg.solve(X.T @ X + RIDGE * np.eye(p), X.T @ Y)
    return B.T


def _principal_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    cosine = np.clip(abs(float(np.dot(v1, v2))), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _subspace_rotation_deg(basis1: np.ndarray, basis2: np.ndarray) -> float:
    """
    Maximum principal angle between consecutive active right-singular
    subspaces. Rows of each basis are orthonormal right singular vectors.
    """
    singular_values = np.linalg.svd(basis1 @ basis2.T, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    angles = np.degrees(np.arccos(singular_values))
    return float(np.max(angles))


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
        "vbasis": Vt.copy(),
        "contribution": contribution,
    }


def build_history(df: pd.DataFrame, window: int) -> pd.DataFrame:
    use = df[XCOLS + YCOLS].replace([np.inf, -np.inf], np.nan)

    rows = []
    previous_vmin = None
    previous_vbasis = None

    for i in range(window - 1 + HORIZON, len(use)):
        result = _single_date_svd(use, i, window)
        if result is None:
            continue

        s = result["singular_values"]
        vmin = result["vmin"]
        vbasis = result["vbasis"]

        rotation = (
            np.nan if previous_vmin is None
            else _principal_angle(previous_vmin, vmin)
        )
        tangent_rotation = (
            np.nan if previous_vbasis is None
            else _subspace_rotation_deg(previous_vbasis, vbasis)
        )

        previous_vmin = vmin
        previous_vbasis = vbasis

        date = use.index[i]
        spy_close = (
            float(df.loc[date, "SPY_AdjClose"])
            if "SPY_AdjClose" in df.columns and date in df.index
            and pd.notna(df.loc[date, "SPY_AdjClose"])
            else np.nan
        )

        row = {
            "Date": date,
            "SPYClose": spy_close,
            "SigmaMin": float(s[-1]),
            "SigmaMax": float(s[0]),
            "SigmaRatio": float(s[-1] / s[0]) if s[0] > 0 else np.nan,
            "ConditionNumber": float(s[0] / s[-1]) if s[-1] > 0 else np.inf,
            "RotationDeg": rotation,
            "TangentPlaneRotationDeg": tangent_rotation,
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

    ratio_pct = metrics.get("sigma_ratio_percentile")
    cond_pct = metrics.get("condition_percentile")
    rot_pct = metrics.get("rotation_percentile")
    sigma_decline = metrics.get("sigma_decline_fraction_10d")
    cond_rise = metrics.get("condition_rise_fraction_10d")

    if ratio_pct is not None and np.isfinite(ratio_pct):
        if ratio_pct <= 10:
            score += 2
            reasons.append("sigma-min / sigma-max is in the bottom 10% of its trailing-year range")
        elif ratio_pct <= 20:
            score += 1
            reasons.append("the active singular-value ratio is below its normal range")

    if cond_pct is not None and np.isfinite(cond_pct):
        if cond_pct >= 95:
            score += 2
            reasons.append("condition number is above the 95th trailing-year percentile")
        elif cond_pct >= 85:
            score += 1
            reasons.append("condition number is elevated")

    if rot_pct is not None and np.isfinite(rot_pct):
        if rot_pct >= 95:
            score += 2
            reasons.append("weak-direction rotation is above the 95th trailing-year percentile")
        elif rot_pct >= 85:
            score += 1
            reasons.append("weak-direction rotation is elevated")

    if sigma_decline is not None and np.isfinite(sigma_decline) and sigma_decline >= 0.70:
        score += 1
        reasons.append("sigma-min has declined on most of the last 10 sessions")

    if cond_rise is not None and np.isfinite(cond_rise) and cond_rise >= 0.70:
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


def _group_contributions(current: pd.Series) -> list[dict]:
    groups = []
    for group, variables in DRIVER_GROUPS.items():
        value = sum(float(current[f"Contribution_{v}"]) for v in variables)
        groups.append({"name": group, "value": value})
    groups.sort(key=lambda x: x["value"], reverse=True)
    return groups


def _detail_row(history: pd.DataFrame, i: int) -> dict:
    current = history.iloc[i]
    start = max(0, i - 252)
    reference = history.iloc[start:i]

    sigma_series = history["SigmaMin"].iloc[max(0, i - 10):i + 1]
    cond_series = history["ConditionNumber"].iloc[max(0, i - 10):i + 1]

    metrics = {
        "sigma_ratio_percentile": _percentile(reference["SigmaRatio"], current["SigmaRatio"]),
        "condition_percentile": _percentile(reference["ConditionNumber"], current["ConditionNumber"]),
        "rotation_percentile": _percentile(reference["RotationDeg"], current["RotationDeg"]),
        "sigma_decline_fraction_10d": _fraction(sigma_series, rising=False),
        "condition_rise_fraction_10d": _fraction(cond_series, rising=True),
    }

    status, reasons, score = _classify(metrics)

    contributions = [
        {
            "key": variable,
            "name": FRIENDLY[variable],
            "value": float(current[f"Contribution_{variable}"]),
        }
        for variable in XCOLS
    ]
    contributions.sort(key=lambda item: item["value"], reverse=True)
    groups = _group_contributions(current)

    return {
        "date": history.index[i].strftime("%Y-%m-%d"),
        "spy_close": None if pd.isna(current["SPYClose"]) else float(current["SPYClose"]),
        "sigma_min": float(current["SigmaMin"]),
        "sigma_max": float(current["SigmaMax"]),
        "sigma_ratio": float(current["SigmaRatio"]),
        "condition_number": float(current["ConditionNumber"]),
        "rotation_deg": None if pd.isna(current["RotationDeg"]) else float(current["RotationDeg"]),
        "tangent_plane_rotation_deg": (
            None if pd.isna(current["TangentPlaneRotationDeg"])
            else float(current["TangentPlaneRotationDeg"])
        ),
        **metrics,
        "status": status,
        "degeneracy_score": score,
        "reasons": reasons,
        "contributions": contributions,
        "driver_groups": groups,
        "dominant_driver": groups[0]["name"],
    }


def analyze_window(df: pd.DataFrame, window: int) -> dict:
    history = build_history(df, window)

    all_rows = [_detail_row(history, i) for i in range(len(history))]
    history_detail = all_rows[-252:]
    current = all_rows[-1]

    daily_table = []
    for row in all_rows:
        group_map = {x["name"]: x["value"] for x in row["driver_groups"]}
        daily_table.append({
            "date": row["date"],
            "status": row["status"],
            "degeneracy_score": row["degeneracy_score"],
            "sigma_min": row["sigma_min"],
            "condition_number": row["condition_number"],
            "weak_rotation_deg": row["rotation_deg"],
            "tangent_plane_rotation_deg": row["tangent_plane_rotation_deg"],
            "dominant_driver": row["dominant_driver"],
            "driver_usd_pct": 100.0 * group_map.get("USD", 0.0),
            "driver_rates_pct": 100.0 * group_map.get("Rates", 0.0),
            "driver_credit_pct": 100.0 * group_map.get("Credit", 0.0),
            "driver_energy_pct": 100.0 * group_map.get("Energy", 0.0),
            "driver_equities_pct": 100.0 * group_map.get("Equities", 0.0),
            "driver_volatility_pct": 100.0 * group_map.get("Volatility", 0.0),
            "spy_close": row["spy_close"],
        })

    return {
        "window": window,
        **current,
        "history": [
            {
                "date": row["date"],
                "sigma_ratio": row["sigma_ratio"],
                "condition": row["condition_number"],
                "rotation": row["rotation_deg"],
                "tangent_rotation": row["tangent_plane_rotation_deg"],
                "status": row["status"],
            }
            for row in history_detail
        ],
        "history_detail": history_detail,
        "daily_table": daily_table,
    }


def _overall_status(window_rows: list[dict]) -> tuple[str, int, int]:
    confirming = sum(SEVERITY.get(row["status"], 0) >= 2 for row in window_rows)
    shifting_or_worse = sum(SEVERITY.get(row["status"], 0) >= 1 for row in window_rows)
    highest = max(window_rows, key=lambda row: SEVERITY.get(row["status"], 0))

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

    return overall, confirming, shifting_or_worse


def _build_status_history(windows: list[dict]) -> list[dict]:
    by_window = {
        item["window"]: {row["date"]: row for row in item["history_detail"]}
        for item in windows
    }
    common_dates = set.intersection(*(set(rows.keys()) for rows in by_window.values()))
    result = []

    for date in sorted(common_dates):
        rows = [by_window[window][date] for window in WINDOWS]
        overall, confirming, shifting_or_worse = _overall_status(rows)
        result.append({
            "date": date,
            "overall_status": overall,
            "confirming_windows": confirming,
            "shifting_or_worse_windows": shifting_or_worse,
            "window_statuses": {
                str(window): by_window[window][date]["status"]
                for window in WINDOWS
            },
        })

    return result[-252:]


def build_dashboard(df: pd.DataFrame) -> dict:
    windows = [analyze_window(df, window) for window in WINDOWS]
    overall, confirming, shifting_or_worse = _overall_status(windows)

    aggregate: dict[str, list[float]] = {}
    for item in windows:
        for contribution in item["contributions"]:
            aggregate.setdefault(contribution["name"], []).append(contribution["value"])

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
        "status_history": _build_status_history(windows),
        "research_note": (
            "Degeneracy flags describe the estimated local response geometry. "
            "They are not trading signals, probability forecasts, or proof of "
            "a classical fold/cusp catastrophe."
        ),
    }
