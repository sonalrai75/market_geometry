from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Make the repository root importable when this script is run as:
# python research/build_dashboard_snapshot.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_geometry.data import build_dataset
from market_geometry.engine import build_dashboard

DATA_DIR = ROOT / "data"
SNAPSHOT = DATA_DIR / "dashboard_snapshot.json"


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading latest public market and macro data...")
    df = build_dataset(save=True)

    print("Computing 63 / 126 / 252-day SVD dashboard...")
    payload = build_dashboard(df)
    payload["snapshot_generated_at"] = datetime.now(timezone.utc).isoformat()

    clean = sanitize(payload)

    with SNAPSHOT.open("w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, allow_nan=False)

    print(f"Saved snapshot: {SNAPSHOT}")
    print(f"As of: {payload['as_of']}")
    print(f"Overall status: {payload['overall_status']}")


if __name__ == "__main__":
    main()
