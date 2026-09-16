from __future__ import annotations

import io
import os
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


FRED_SERIES = {
    "VIX": "VIXCLS",
    "Treasury10Y": "DGS10",
    "Curve10Y2Y": "T10Y2Y",
    "CreditSpread_BAA10Y": "BAA10Y",
    "WTI": "DCOILWTICO",
    "BroadUSD": "DTWEXBGS",
}


def _start_date() -> str:
    # The live monitor needs enough history for a 252-day model plus a
    # trailing 252-day reference distribution. Five years provides margin
    # while keeping daily refreshes much lighter than the full research set.
    return (date.today() - timedelta(days=365 * 5 + 30)).isoformat()


def _cache_dir() -> Path:
    # Vercel serverless storage is ephemeral. /tmp is writable during an
    # invocation; locally we use the repository's data/ folder.
    base = Path("/tmp/market_geometry") if os.getenv("VERCEL") else Path("data")
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path() -> Path:
    return _cache_dir() / "market_geometry_live.csv"


def fetch_fred(series_id: str, start: str) -> pd.Series:
    params = urllib.parse.urlencode({"id": series_id, "cosd": start})
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 MarketGeometryMonitor/0.2"},
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        raw = response.read()

    frame = pd.read_csv(io.BytesIO(raw))
    date_col, value_col = frame.columns[:2]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    return frame.set_index(date_col)[value_col].sort_index()


def _yf_field(raw: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    if isinstance(raw.columns, pd.MultiIndex):
        key = (field, ticker)
        if key in raw.columns:
            return raw[key]
        if field == "Adj Close" and ("Close", ticker) in raw.columns:
            return raw[("Close", ticker)]
    raise KeyError(f"Could not locate {field} for {ticker} in yfinance response.")


def fetch_etfs(start: str) -> pd.DataFrame:
    tickers = ["SPY", "QQQ", "RSP"]
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("No ETF data returned by market-data provider.")

    idx = pd.to_datetime(raw.index)
    try:
        idx = idx.tz_localize(None)
    except TypeError:
        pass

    out = pd.DataFrame(index=idx)
    for ticker in tickers:
        out[f"{ticker}_AdjClose"] = _yf_field(raw, "Adj Close", ticker)
        out[f"{ticker}_Volume"] = _yf_field(raw, "Volume", ticker)
    return out.sort_index()


def _annualized_vol(ret: pd.Series, window: int = 20) -> pd.Series:
    return ret.rolling(window).std() * np.sqrt(252.0)


def _rolling_drawdown(px: pd.Series, window: int = 63) -> pd.Series:
    peak = px.rolling(window, min_periods=1).max()
    return px / peak - 1.0


def build_dataset(save: bool = True) -> pd.DataFrame:
    start = _start_date()

    macro = pd.DataFrame()
    for name, series_id in FRED_SERIES.items():
        macro[name] = fetch_fred(series_id, start)

    market = fetch_etfs(start)
    df = market.join(macro, how="left").sort_index()

    # FRED series do not always update on every trading date.
    df[list(FRED_SERIES)] = df[list(FRED_SERIES)].ffill(limit=5)

    df["RSP_SPY_Ratio"] = df["RSP_AdjClose"] / df["SPY_AdjClose"]

    log_volume = np.log(df["SPY_Volume"].replace(0, np.nan))
    df["SPY_Volume_Z20"] = (
        (log_volume - log_volume.rolling(20).mean())
        / log_volume.rolling(20).std()
    )

    df["SPY_Return1D"] = df["SPY_AdjClose"].pct_change()
    df["QQQ_Return1D"] = df["QQQ_AdjClose"].pct_change()

    df["SPY_RVol20"] = _annualized_vol(df["SPY_Return1D"], 20)
    df["SPY_Drawdown63"] = _rolling_drawdown(df["SPY_AdjClose"], 63)

    # The research model uses leakage-safe 5-day forward responses. The most
    # recent HORIZON days therefore cannot themselves be training observations.
    df["SPY_Fwd5D_Return"] = df["SPY_AdjClose"].shift(-5) / df["SPY_AdjClose"] - 1
    df["QQQ_Fwd5D_Return"] = df["QQQ_AdjClose"].shift(-5) / df["QQQ_AdjClose"] - 1

    df.index.name = "Date"

    if save:
        df.to_csv(cache_path(), float_format="%.10g")

    return df


def load_dataset(refresh: bool = False) -> pd.DataFrame:
    path = cache_path()

    if refresh or not path.exists():
        return build_dataset(save=True)

    # Locally, don't keep stale data across days. Vercel /tmp may vanish anyway.
    try:
        mtime_date = date.fromtimestamp(path.stat().st_mtime)
        if mtime_date < date.today():
            return build_dataset(save=True)
    except OSError:
        pass

    return pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
