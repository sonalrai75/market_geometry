from __future__ import annotations

import io
import os
import time
import urllib.error
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

RAW_MARKET_COLS = [
    "SPY_AdjClose",
    "SPY_Volume",
    "QQQ_AdjClose",
    "QQQ_Volume",
    "RSP_AdjClose",
    "RSP_Volume",
]
RAW_COLS = RAW_MARKET_COLS + list(FRED_SERIES.keys())


HISTORY_START = "2005-01-01"

def _start_date() -> str:
    return HISTORY_START


def _cache_dir() -> Path:
    base = Path("/tmp/market_geometry") if os.getenv("VERCEL") else Path("data")
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_path() -> Path:
    return _cache_dir() / "market_geometry_live.csv"


def fetch_fred(
    series_id: str,
    start: str,
    attempts: int = 4,
    timeout: int = 45,
) -> pd.Series:
    """
    Fetch one FRED series with retries/backoff.

    GitHub-hosted runners occasionally see transient read timeouts from FRED.
    This keeps one slow upstream response from immediately failing the whole
    daily workflow.
    """
    params = urllib.parse.urlencode({"id": series_id, "cosd": start})
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{params}"

    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 MarketGeometryMonitor/1.1",
                    "Accept": "text/csv,*/*;q=0.8",
                    "Connection": "close",
                },
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read()

            frame = pd.read_csv(io.BytesIO(raw))
            date_col, value_col = frame.columns[:2]
            frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
            frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")

            return (
                frame.dropna(subset=[date_col])
                .set_index(date_col)[value_col]
                .sort_index()
            )

        except (
            TimeoutError,
            urllib.error.URLError,
            urllib.error.HTTPError,
            ConnectionError,
            OSError,
        ) as exc:
            last_error = exc

            if attempt == attempts:
                break

            wait_seconds = min(5 * (2 ** (attempt - 1)), 30)
            print(
                f"FRED {series_id} attempt {attempt}/{attempts} failed: "
                f"{exc}. Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"FRED series {series_id} failed after {attempts} attempts: {last_error}"
    )


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

    last_error = None
    raw = None

    for attempt in range(1, 4):
        try:
            raw = yf.download(
                tickers,
                start=start,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
                timeout=30,
            )

            if raw is not None and not raw.empty:
                break

            raise RuntimeError("No ETF data returned by market-data provider.")

        except Exception as exc:
            last_error = exc
            if attempt == 3:
                raise
            wait_seconds = 5 * attempt
            print(
                f"ETF download attempt {attempt}/3 failed: {exc}. "
                f"Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    if raw is None or raw.empty:
        raise RuntimeError(f"No ETF data returned: {last_error}")

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


def fetch_raw_dataset(start: str) -> pd.DataFrame:
    macro = pd.DataFrame()

    for name, series_id in FRED_SERIES.items():
        print(f"Fetching FRED {series_id} -> {name}")
        macro[name] = fetch_fred(series_id, start)

    market = fetch_etfs(start)
    df = market.join(macro, how="left").sort_index()
    df[list(FRED_SERIES)] = df[list(FRED_SERIES)].ffill(limit=7)

    return df


def _annualized_vol(ret: pd.Series, window: int = 20) -> pd.Series:
    return ret.rolling(window).std() * np.sqrt(252.0)


def _rolling_drawdown(px: pd.Series, window: int = 63) -> pd.Series:
    peak = px.rolling(window, min_periods=1).max()
    return px / peak - 1.0


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

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

    df["SPY_Fwd5D_Return"] = (
        df["SPY_AdjClose"].shift(-5) / df["SPY_AdjClose"] - 1
    )
    df["QQQ_Fwd5D_Return"] = (
        df["QQQ_AdjClose"].shift(-5) / df["QQQ_AdjClose"] - 1
    )

    df.index.name = "Date"
    return df


def build_dataset(save: bool = True) -> pd.DataFrame:
    df = add_features(fetch_raw_dataset(_start_date()))

    if save:
        df.to_csv(cache_path(), float_format="%.10g")

    return df


def refresh_dataset(
    save: bool = True,
    overlap_days: int = 45,
) -> pd.DataFrame:
    path = cache_path()

    if not path.exists():
        return build_dataset(save=save)

    existing = (
        pd.read_csv(path, parse_dates=["Date"])
        .set_index("Date")
        .sort_index()
    )

    if existing.empty:
        return build_dataset(save=save)

    latest = existing.index.max()
    refresh_start = (
        latest - pd.Timedelta(days=overlap_days)
    ).date().isoformat()

    recent_raw = fetch_raw_dataset(refresh_start)

    available_raw = [c for c in RAW_COLS if c in existing.columns]
    old_raw = existing[available_raw].copy()

    cutoff = pd.Timestamp(refresh_start)
    old_raw = old_raw[old_raw.index < cutoff]

    merged = pd.concat([old_raw, recent_raw], axis=0).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    for col in RAW_COLS:
        if col not in merged.columns:
            merged[col] = np.nan

    merged[list(FRED_SERIES)] = merged[list(FRED_SERIES)].ffill(limit=7)

    df = add_features(merged)

    if save:
        df.to_csv(path, float_format="%.10g")

    return df


def load_dataset(refresh: bool = False) -> pd.DataFrame:
    path = cache_path()

    if refresh:
        return refresh_dataset(save=True)

    if not path.exists():
        return build_dataset(save=True)

    return (
        pd.read_csv(path, parse_dates=["Date"])
        .set_index("Date")
        .sort_index()
    )
