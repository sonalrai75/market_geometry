from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from market_geometry.data import load_dataset
from market_geometry.engine import build_dashboard

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = FastAPI(
    title="Market Geometry Monitor",
    version="0.2.0",
    description="Daily evolving-SVD market geometry and degeneracy monitor.",
)

# Vercel now promotes FastAPI StaticFiles content to its CDN.
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_memory_cache: dict[str, object] = {
    "payload": None,
    "refreshed_at": None,
}


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "market-geometry-monitor"}


def _dashboard(refresh: bool = False):
    try:
        df = load_dataset(refresh=refresh)
        payload = build_dashboard(df)
        payload["refreshed_at"] = datetime.now(timezone.utc).isoformat()
        _memory_cache["payload"] = payload
        _memory_cache["refreshed_at"] = payload["refreshed_at"]
        return payload
    except Exception as exc:
        cached = _memory_cache.get("payload")
        if cached is not None:
            stale = dict(cached)
            stale["warning"] = f"Refresh failed; showing prior in-memory result: {exc}"
            return stale
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/api/dashboard")
def dashboard():
    return _dashboard(refresh=False)


@app.get("/api/refresh")
def refresh():
    # GET makes this endpoint compatible with a future Vercel Cron invocation.
    return _dashboard(refresh=True)


@app.get("/api/history")
def history(window: int = Query(126, enum=[63, 126, 252])):
    payload = _dashboard(refresh=False)
    for item in payload["windows"]:
        if item["window"] == window:
            return {
                "window": window,
                "as_of": item["date"],
                "history": item["history"],
            }
    raise HTTPException(status_code=404, detail="Unsupported window.")
