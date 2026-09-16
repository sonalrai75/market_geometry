from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
SNAPSHOT = ROOT / "data" / "dashboard_snapshot.json"

app = FastAPI(
    title="Market Geometry Monitor",
    version="0.3.0",
    description="Daily evolving-SVD market geometry and degeneracy monitor.",
)

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "market-geometry-monitor",
        "snapshot_exists": SNAPSHOT.exists(),
    }


def _read_snapshot():
    if not SNAPSHOT.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Dashboard snapshot is not available. "
                "Run: python research/build_dashboard_snapshot.py"
            ),
        )

    try:
        with SNAPSHOT.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read dashboard snapshot: {exc}",
        )


@app.get("/api/dashboard")
def dashboard():
    return _read_snapshot()


@app.get("/api/history")
def history(window: int = 126):
    if window not in (63, 126, 252):
        raise HTTPException(status_code=400, detail="window must be 63, 126, or 252")

    payload = _read_snapshot()
    for item in payload.get("windows", []):
        if item.get("window") == window:
            return {
                "window": window,
                "as_of": item.get("date"),
                "history": item.get("history", []),
            }

    raise HTTPException(status_code=404, detail="Window not found in snapshot.")
