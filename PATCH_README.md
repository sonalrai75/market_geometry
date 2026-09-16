# Snapshot Patch

The Vercel logs show `/` and `/health` returning 200 while `/api/dashboard`
returns 503 with a read timeout. That confirms the FastAPI route exists and
the timeout occurs during the live data-download / SVD pipeline.

## New architecture

Browser -> `/api/dashboard` -> reads `data/dashboard_snapshot.json`

Heavy work is performed separately:

```bash
python research/build_dashboard_snapshot.py
```

That script downloads the public data, computes the 63/126/252-day SVD
dashboard, and writes the snapshot.

Commit `data/dashboard_snapshot.json` to Git so Vercel can return it instantly.

Keep CSV caches ignored, but do not ignore `data/dashboard_snapshot.json`.
