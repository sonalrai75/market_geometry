# Market Geometry Monitor — Vercel App Files

These files are intended to live at the root of the existing
`market_geometry` repository alongside the `research/`, `outputs/`, and
`data/` directories.

## Production layout

```text
market_geometry/
├── app.py
├── market_geometry/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   └── engine.py
├── static/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── research/
├── outputs/
├── data/
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

## Local run

From Git Bash:

```bash
cd /c/projects/market_geometry
source .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Vercel

Current Vercel FastAPI support recognizes a root FastAPI app automatically,
so no `vercel.json` routing file is required.

The FastAPI instance must remain named:

```python
app = FastAPI(...)
```

Deployment can then be connected to this GitHub repository in Vercel.

## Important persistence note

Vercel serverless filesystem persistence is ephemeral. V1 therefore treats
the public-source dataset as a regenerable cache. For a production daily
history of SVD states and alert events, add a durable database in a later
version rather than relying on `/tmp`.

## Endpoints

- `/` — dashboard
- `/health` — health check
- `/api/dashboard` — current full analysis
- `/api/refresh` — force a new public-data refresh
- `/api/history?window=126` — recent SVD history for one time scale
