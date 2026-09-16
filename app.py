from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
SNAPSHOT = ROOT / "data" / "dashboard_snapshot.json"

app = FastAPI(
    title="Market Geometry Monitor",
    version="1.1.0",
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


def _read_snapshot() -> dict:
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
def history(window: int = Query(126, enum=[63, 126, 252])):
    payload = _read_snapshot()
    for item in payload.get("windows", []):
        if item.get("window") == window:
            return {
                "window": window,
                "as_of": item.get("date"),
                "history": item.get("history", []),
                "history_detail": item.get("history_detail", []),
            }

    raise HTTPException(status_code=404, detail="Window not found.")


@app.get("/api/status-history")
def status_history():
    payload = _read_snapshot()
    return {
        "as_of": payload.get("as_of"),
        "history": payload.get("status_history", []),
    }


@app.get("/api/explorer")
def explorer(date: str):
    payload = _read_snapshot()
    matches = []

    for item in payload.get("windows", []):
        for row in item.get("history_detail", []):
            if row.get("date") == date:
                matches.append(
                    {
                        "window": item.get("window"),
                        **row,
                    }
                )
                break

    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No historical geometry is available for {date}.",
        )

    overall = next(
        (
            row
            for row in payload.get("status_history", [])
            if row.get("date") == date
        ),
        None,
    )

    return {
        "date": date,
        "overall": overall,
        "windows": sorted(matches, key=lambda x: x["window"]),
    }
