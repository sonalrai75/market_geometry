# Market Geometry Monitor v1.1

This upgrade adds:

- Historical Explorer
- Degeneracy Details
- Daily Status History
- Incremental local data refresh with a 45-day overlap
- Snapshot-only Vercel serving, so the public dashboard does not call FRED or Yahoo live

## Replace these files

- `app.py`
- `market_geometry/data.py`
- `market_geometry/engine.py`
- `research/build_dashboard_snapshot.py`
- `static/index.html`
- `static/app.js`
- `static/styles.css`

Keep your existing:

- `market_geometry/config.py`
- `market_geometry/__init__.py`
- `requirements.txt`
- `pyproject.toml`
- research scripts and outputs

## Build a new snapshot

From Git Bash:

```bash
cd /c/projects/market_geometry
source .venv/Scripts/activate
python research/build_dashboard_snapshot.py
```

The first run after this patch may download more recent overlapping data.
Later runs only refresh the recent tail rather than downloading the full
five-year market history again.

## Test locally

```bash
uvicorn app:app --reload
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/api/dashboard`
- `http://127.0.0.1:8000/api/status-history`

## Git

```bash
git add app.py market_geometry/data.py market_geometry/engine.py \
  research/build_dashboard_snapshot.py static/index.html static/app.js \
  static/styles.css data/dashboard_snapshot.json UPGRADE_README.md

git commit -m "Add historical explorer and degeneracy detail views"
git push
```
