"""
XFeatures — download, cache, and load daily external series for crypto.

Cached CSVs are stored in the same directory as this file so that both
Fincast and Kronos pipelines share a single source of truth.

Each loader returns a *daily* pd.Series indexed by date (tz-naive, floor'd
to midnight).  ``load_xfeatures`` merges all available series into a single
DataFrame.

DVOL handling:
  - BTC_DVOL.csv and ETH_DVOL.csv are loaded directly for BTC/ETH assets.
  - For other altcoins, a Volatility Beta proxy is computed:
        IV_i,t = DVOL_BTC,t * (σ_i,t / σ_BTC,t)
    where σ is the rolling 30-day realized volatility.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ┏━━━━━━━━━━ Cache directory (same dir as this file) ━━━━━━━━━━┓
XFEATURES_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_ts(d: str | pd.Timestamp | datetime) -> pd.Timestamp:
    return pd.Timestamp(d)


def _normalise_index(s: pd.Series, name: str) -> pd.Series:
    """Ensure tz-naive daily DatetimeIndex named 'date'."""
    idx = pd.to_datetime(s.index, utc=True, errors="coerce")
    try:
        idx = idx.tz_convert(None)
    except Exception:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass
    s.index = idx.floor("D")
    s.index.name = "date"
    s.name = name
    return s


def _read_cryptodatadownload_csv(path: Path) -> pd.DataFrame:
    """Read a CSV that may have a CryptoDataDownload header line."""
    with open(path) as f:
        first_line = f.readline()
    skip = 1 if "cryptodatadownload" in first_line.lower() else 0
    return pd.read_csv(path, skiprows=skip)


# ---------------------------------------------------------------------------
# 1.  Fear & Greed Index
# ---------------------------------------------------------------------------

def _fetch_fear_greed(date_start: str, date_end: str) -> pd.Series:
    """Load Fear & Greed Index from local CSV or Alternative.me API."""
    # Try local file first (user-placed)
    local = XFEATURES_DIR / "Fear_Greed_Index.csv"
    if local.exists():
        df = pd.read_csv(local)
        dt_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
        val_col = next((c for c in df.columns if "fear" in c.lower() or "greed" in c.lower()), None)
        if val_col is None:
            val_col = [c for c in df.columns if c != dt_col][0]
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col])
        df.index = df[dt_col].dt.floor("D")
        df.index.name = "date"
        s = pd.to_numeric(df[val_col], errors="coerce").rename("fear_greed_idx")
        s = s.groupby(s.index).last()
        ts_start, ts_end = _to_ts(date_start), _to_ts(date_end)
        s = s.loc[ts_start:ts_end]
        return _normalise_index(s, "fear_greed_idx")

    # Fallback: API download + cache
    cache_path = XFEATURES_DIR / "fear_greed_index_cache.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
        ts_start, ts_end = _to_ts(date_start), _to_ts(date_end)
        if not cached.empty:
            if cached.index.min() <= ts_start and cached.index.max() >= ts_end - timedelta(days=2):
                s = cached["fear_greed_idx"].loc[ts_start:ts_end]
                return _normalise_index(s, "fear_greed_idx")

    try:
        import urllib.request
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = json.loads(resp.read().decode())
    except Exception as exc:
        warnings.warn(f"[XFeatures] Fear & Greed download failed: {exc}")
        return pd.Series(dtype=float, name="fear_greed_idx")

    records = raw.get("data", [])
    rows = []
    for r in records:
        try:
            ts = pd.Timestamp(int(r["timestamp"]), unit="s")
            val = float(r["value"])
            rows.append({"date": ts, "fear_greed_idx": val})
        except Exception:
            continue

    if not rows:
        return pd.Series(dtype=float, name="fear_greed_idx")

    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
    df["date"] = pd.to_datetime(df["date"]).dt.floor("D")
    df = df.set_index("date")
    df.to_csv(cache_path)

    ts_start, ts_end = _to_ts(date_start), _to_ts(date_end)
    s = df["fear_greed_idx"].loc[ts_start:ts_end]
    return _normalise_index(s, "fear_greed_idx")


# ---------------------------------------------------------------------------
# 2.  DVOL  (per-symbol: BTC_DVOL.csv, ETH_DVOL.csv)
# ---------------------------------------------------------------------------

def _load_dvol_file(symbol: str) -> pd.Series:
    """Load a single DVOL CSV file for a given symbol (e.g. 'BTC', 'ETH').

    Looks for files matching {symbol}_DVOL.csv or {symbol}_dvol.csv.
    Handles CryptoDataDownload header lines.

    Returns a daily Series named 'dvol', or empty if not found.
    """
    candidates = list(XFEATURES_DIR.glob(f"{symbol}_DVOL*.csv")) + \
                 list(XFEATURES_DIR.glob(f"{symbol}_dvol*.csv")) + \
                 list(XFEATURES_DIR.glob(f"{symbol.upper()}_DVOL*.csv"))
    # Deduplicate
    seen = set()
    unique = []
    for p in candidates:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)

    for p in unique:
        try:
            raw_df = _read_cryptodatadownload_csv(p)
            dt_col = next((c for c in raw_df.columns if "date" in c.lower()), raw_df.columns[0])
            val_col = next((c for c in raw_df.columns if c.lower() == "close"), None)
            if val_col is None:
                val_col = next((c for c in raw_df.columns if "dvol" in c.lower()), None)
            if val_col is None:
                val_col = [c for c in raw_df.columns if c not in (dt_col, "symbol")][-1]
            raw_df[dt_col] = pd.to_datetime(raw_df[dt_col], errors="coerce")
            raw_df = raw_df.dropna(subset=[dt_col])
            try:
                raw_df[dt_col] = raw_df[dt_col].dt.tz_convert(None)
            except Exception:
                pass
            raw_df.index = raw_df[dt_col].dt.floor("D")
            raw_df.index.name = "date"
            s = pd.to_numeric(raw_df[val_col], errors="coerce").rename("dvol")
            s = s.groupby(s.index).last().sort_index()
            return s
        except Exception:
            continue

    return pd.Series(dtype=float, name="dvol")


def load_dvol_for_asset(asset: str, date_start: str, date_end: str,
                        asset_close: Optional[pd.Series] = None,
                        btc_close: Optional[pd.Series] = None) -> pd.Series:
    """Load DVOL for any crypto asset, using vol-beta proxy if needed.

    For BTC/ETH: loads directly from {symbol}_DVOL.csv.
    For altcoins: computes IV_i = DVOL_BTC * (σ_i / σ_BTC) where σ is
    rolling 30-day realized volatility.

    Parameters
    ----------
    asset : str
        Asset symbol (e.g. 'BTCUSDT', 'ETHUSDT', 'SOLUSDT').
    date_start, date_end : str
        Date range.
    asset_close : pd.Series, optional
        Daily close prices for the asset (needed for altcoin proxy).
    btc_close : pd.Series, optional
        Daily close prices for BTC (needed for altcoin proxy).

    Returns
    -------
    pd.Series : daily DVOL (or proxy), indexed by date.
    """
    ts_start, ts_end = _to_ts(date_start), _to_ts(date_end)

    # Normalise asset name: "BTCUSDT" → "BTC", "ETHUSDT" → "ETH"
    sym = asset.upper().replace("USDT", "").replace("USD", "")

    # Direct DVOL file for this asset?
    dvol = _load_dvol_file(sym)
    if len(dvol) > 0:
        dvol = dvol.loc[ts_start:ts_end]
        return _normalise_index(dvol, "dvol")

    # Volatility Beta Proxy: IV_i = DVOL_BTC * (σ_i / σ_BTC)
    dvol_btc = _load_dvol_file("BTC")
    if len(dvol_btc) == 0:
        return pd.Series(dtype=float, name="dvol")

    if asset_close is None or btc_close is None:
        warnings.warn(f"[XFeatures] Cannot compute DVOL proxy for {asset}: "
                      f"need asset_close and btc_close for vol-beta scaling.")
        return pd.Series(dtype=float, name="dvol")

    # Rolling 30-day realized vol (annualised, 360-day crypto convention)
    window = 30

    def _rvol(prices: pd.Series) -> pd.Series:
        lr = np.log(prices / prices.shift(1))
        return lr.rolling(window, min_periods=window).std() * np.sqrt(360)

    sigma_i   = _rvol(asset_close)
    sigma_btc = _rvol(btc_close)

    # Align DVOL_BTC to the same daily index
    dvol_btc = dvol_btc.reindex(sigma_btc.index, method="ffill")

    # IV_i = DVOL_BTC * (σ_i / σ_BTC)
    proxy = dvol_btc * (sigma_i / (sigma_btc + 1e-10))
    proxy = proxy.rename("dvol")
    proxy = proxy.loc[ts_start:ts_end]

    return _normalise_index(proxy, "dvol")


# ---------------------------------------------------------------------------
# 3.  News Sentiment  (from local CSV or Excel)
# ---------------------------------------------------------------------------

def _fetch_news_sentiment(date_start: str, date_end: str) -> pd.Series:
    """Load News Sentiment from a local CSV or Excel file in XFeatures/."""
    ts_start, ts_end = _to_ts(date_start), _to_ts(date_end)

    # Try CSV files with "sentiment" in the name
    for p in sorted(XFEATURES_DIR.glob("*[Ss]entiment*.csv")):
        try:
            raw = pd.read_csv(p)
            dt_col = next((c for c in raw.columns if "date" in c.lower()), raw.columns[0])
            val_col = next((c for c in raw.columns if "sentiment" in c.lower()), None)
            if val_col is None:
                val_col = [c for c in raw.columns if c != dt_col][0]
            raw[dt_col] = pd.to_datetime(raw[dt_col], errors="coerce")
            raw = raw.dropna(subset=[dt_col])
            try:
                raw[dt_col] = raw[dt_col].dt.tz_convert(None)
            except Exception:
                pass
            raw.index = raw[dt_col].dt.floor("D")
            raw.index.name = "date"
            s = pd.to_numeric(raw[val_col], errors="coerce").rename("news_sentiment")
            s = s.groupby(s.index).last()
            s = s.loc[ts_start:ts_end]
            return _normalise_index(s, "news_sentiment")
        except Exception:
            continue

    # Try Excel files
    for xlsx in sorted(XFEATURES_DIR.glob("*[Ss]entiment*.xlsx")):
        try:
            raw = pd.read_excel(xlsx)
            dt_col = next((c for c in raw.columns if "date" in c.lower()), raw.columns[0])
            val_col = next((c for c in raw.columns if "sentiment" in c.lower()), None)
            if val_col is None:
                val_col = [c for c in raw.columns if c != dt_col][0]
            raw[dt_col] = pd.to_datetime(raw[dt_col], errors="coerce")
            raw = raw.dropna(subset=[dt_col])
            try:
                raw[dt_col] = raw[dt_col].dt.tz_convert(None)
            except Exception:
                pass
            raw.index = raw[dt_col].dt.floor("D")
            raw.index.name = "date"
            s = pd.to_numeric(raw[val_col], errors="coerce").rename("news_sentiment")
            s = s.groupby(s.index).last()
            s = s.loc[ts_start:ts_end]
            return _normalise_index(s, "news_sentiment")
        except Exception:
            continue

    warnings.warn("[XFeatures] News Sentiment: no source file found in XFeatures/.")
    return pd.Series(dtype=float, name="news_sentiment")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_xfeatures(date_start: str,
                   date_end: str,
                   force_refresh: bool = False) -> pd.DataFrame:
    """Load non-asset-specific crypto external features as a daily DataFrame.

    Returns fear_greed_idx and news_sentiment (market-wide, same for all assets).
    DVOL is loaded separately per-asset via ``load_dvol_for_asset``.

    Parameters
    ----------
    date_start, date_end : str
        Date range (inclusive-ish) to load.
    force_refresh : bool
        If True, re-download Fear & Greed even if cached.

    Returns
    -------
    pd.DataFrame
        Columns: fear_greed_idx, news_sentiment
        Index: DatetimeIndex named 'date' (daily, tz-naive).
    """
    if force_refresh:
        cache = XFEATURES_DIR / "fear_greed_index_cache.csv"
        if cache.exists():
            cache.unlink()

    fng  = _fetch_fear_greed(date_start, date_end)
    sent = _fetch_news_sentiment(date_start, date_end)

    full_idx = pd.date_range(_to_ts(date_start), _to_ts(date_end), freq="D", name="date")
    df = pd.DataFrame(index=full_idx)

    for s in (fng, sent):
        if len(s) > 0:
            df = df.join(s, how="left")
        else:
            df[s.name] = np.nan

    return df