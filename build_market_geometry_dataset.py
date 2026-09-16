#!/usr/bin/env python
"""
Build a daily Market Geometry dataset for evolving-SVD / catastrophe-style analysis.

Public data sources:
- FRED: VIXCLS, DGS10, T10Y2Y, BAA10Y, DCOILWTICO, DTWEXBGS
- Yahoo Finance via yfinance: SPY, QQQ, RSP

Outputs:
- market_geometry_dataset.csv
"""

from __future__ import annotations
import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

START = "2005-01-01"
END = None  # None => latest available
OUT = Path("market_geometry_dataset.csv")

FRED_SERIES = {
    "VIX": "VIXCLS",
    "Treasury10Y": "DGS10",
    "Curve10Y2Y": "T10Y2Y",
    "CreditSpread_BAA10Y": "BAA10Y",
    "WTI": "DCOILWTICO",
    "BroadUSD": "DTWEXBGS",
}

def fetch_fred(series_id: str, start: str = START) -> pd.Series:
    # FRED's fredgraph CSV endpoint needs no API key.
    params = urllib.parse.urlencode({"id": series_id, "cosd": start})
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{params}"
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read()
    df = pd.read_csv(io.BytesIO(raw))
    date_col = df.columns[0]
    value_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.set_index(date_col)[value_col].sort_index()

def fetch_market_prices():
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError(
            "yfinance is required. Install with: pip install yfinance"
        ) from e

    tickers = ["SPY", "QQQ", "RSP"]
    raw = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")

    out = pd.DataFrame(index=raw.index)
    # yfinance may return MultiIndex columns.
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            out[f"{ticker}_Close"] = raw[("Close", ticker)]
            out[f"{ticker}_AdjClose"] = raw[("Adj Close", ticker)]
            out[f"{ticker}_Volume"] = raw[("Volume", ticker)]
    else:
        # Defensive fallback if only one ticker is returned.
        raise RuntimeError("Unexpected yfinance column format.")
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()

def annualized_realized_vol(ret: pd.Series, window: int = 20) -> pd.Series:
    return ret.rolling(window).std() * np.sqrt(252)

def rolling_drawdown(px: pd.Series, window: int = 63) -> pd.Series:
    rolling_peak = px.rolling(window, min_periods=1).max()
    return px / rolling_peak - 1.0

def main():
    print("Downloading FRED series...")
    fred = pd.DataFrame()
    for name, sid in FRED_SERIES.items():
        print(f"  {name} <- {sid}")
        fred[name] = fetch_fred(sid)

    print("Downloading ETF market data...")
    px = fetch_market_prices()

    # Use SPY trading dates as the master calendar.
    df = px.join(fred, how="left")
    df = df.loc[df.index >= pd.Timestamp(START)].copy()

    # Macro/market indicators may skip holidays or release days.
    # Forward-fill only explanatory state variables, not returns.
    fred_cols = list(FRED_SERIES.keys())
    df[fred_cols] = df[fred_cols].ffill(limit=5)

    # Derived explanatory variables.
    df["RSP_SPY_Ratio"] = df["RSP_AdjClose"] / df["SPY_AdjClose"]
    df["SPY_Volume_Log"] = np.log(df["SPY_Volume"].replace(0, np.nan))
    df["SPY_Volume_Z20"] = (
        (df["SPY_Volume_Log"] - df["SPY_Volume_Log"].rolling(20).mean())
        / df["SPY_Volume_Log"].rolling(20).std()
    )

    # Outcomes.
    df["SPY_Return1D"] = df["SPY_AdjClose"].pct_change()
    df["QQQ_Return1D"] = df["QQQ_AdjClose"].pct_change()
    df["SPY_RVol20"] = annualized_realized_vol(df["SPY_Return1D"], 20)
    df["QQQ_RVol20"] = annualized_realized_vol(df["QQQ_Return1D"], 20)
    df["SPY_Drawdown63"] = rolling_drawdown(df["SPY_AdjClose"], 63)
    df["SPY_QQQ_Corr60"] = (
        df["SPY_Return1D"].rolling(60).corr(df["QQQ_Return1D"])
    )

    # Forward outcomes are useful if the local map is framed as
    # current state -> subsequent market response.
    df["SPY_Fwd5D_Return"] = df["SPY_AdjClose"].shift(-5) / df["SPY_AdjClose"] - 1
    df["QQQ_Fwd5D_Return"] = df["QQQ_AdjClose"].shift(-5) / df["QQQ_AdjClose"] - 1
    df["SPY_Fwd20D_Return"] = df["SPY_AdjClose"].shift(-20) / df["SPY_AdjClose"] - 1

    df.index.name = "Date"
    df.to_csv(OUT, float_format="%.10g")
    print(f"\nSaved {len(df):,} rows to {OUT.resolve()}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

    # Quick coverage check for major stress windows.
    checks = ["2008-09-15", "2020-03-16", "2022-06-13"]
    for d in checks:
        t = pd.Timestamp(d)
        nearest = df.index[df.index.get_indexer([t], method="nearest")[0]]
        print(f"Nearest row to {d}: {nearest.date()}")

if __name__ == "__main__":
    main()
