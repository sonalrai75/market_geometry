# Market Geometry Prototype

This prototype extends the evolving-SVD idea from the robotic design app to financial-market regime geometry.

## Files
- `build_market_geometry_dataset.py` — downloads and merges public daily data.
- `analyze_market_geometry_svd.py` — computes rolling local Jacobians and SVD diagnostics.
- `requirements_market_geometry.txt` — Python dependencies.

## Public inputs

FRED:
- VIXCLS — CBOE VIX
- DGS10 — 10-year Treasury yield
- T10Y2Y — 10y minus 2y Treasury spread
- BAA10Y — Moody's Baa corporate yield minus 10-year Treasury yield
- DCOILWTICO — WTI crude oil
- DTWEXBGS — broad trade-weighted U.S. dollar index

Yahoo Finance via `yfinance`:
- SPY
- QQQ
- RSP

## Run

```bash
pip install -r requirements_market_geometry.txt
python build_market_geometry_dataset.py
python analyze_market_geometry_svd.py
```

## Initial catastrophe-style diagnostics

The analysis tracks:
- smallest singular value
- largest singular value
- condition number
- rotation of the weakest right-singular direction
- composition of that weak direction across market-state variables

These are indicators of evolving local geometry. They are **not, by themselves, proof of a Thom catastrophe**. A stronger second stage should test continuation, branch structure, hysteresis, and fold/cusp normal-form behavior around candidate episodes.
