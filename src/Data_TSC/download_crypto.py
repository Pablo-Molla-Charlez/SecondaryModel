"""Standalone script to download crypto OHLCV data with technical indicators."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BINANCE_LIMIT = 1000
SLEEP_SEC = 0.35

CRYPTO_SYMBOLS: List[str] = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT",
    "SUIUSDT", "BNBUSDT", "TRXUSDT", "ADAUSDT", "LINKUSDT",
    "AVAXUSDT", "XLMUSDT", "SHIBUSDT", "HBARUSDT", "TONUSDT",
    "NEARUSDT", "BCHUSDT", "AAVEUSDT", "DOTUSDT", "LTCUSDT",
]

GRANULARITY_CONFIG = {
    "1m":  {"binance": "1m",  "freq": "1min", "step_ms": 60_000},
    "5m":  {"binance": "5m",  "freq": "5min", "step_ms": 300_000},
    "15m": {"binance": "15m", "freq": "15min", "step_ms": 900_000},
    "1h":  {"binance": "1h",  "freq": "1h",   "step_ms": 3_600_000},
    "4h":  {"binance": "4h",  "freq": "4h",   "step_ms": 14_400_000},
    "1d":  {"binance": "1d",  "freq": "1d",   "step_ms": 86_400_000},
}


# ─────────────────────────────────────────────────────────────────────────────
# Technical Indicators (imported from shared module)
# ─────────────────────────────────────────────────────────────────────────────
from indicators import add_all_indicators


# ─────────────────────────────────────────────────────────────────────────────
# Binance Data Fetching
# ─────────────────────────────────────────────────────────────────────────────
def fetch_binance_klines(
    symbol: str,
    interval: str,
    step_ms: int,
    freq_alias: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Fetch klines from Binance API."""
    url = "https://api.binance.com/api/v3/klines"
    rows: List[list] = []
    cur = start_ms

    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": BINANCE_LIMIT,
            "startTime": cur,
            "endTime": end_ms,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            chunk = resp.json()
        except Exception as exc:
            print(f"  ⚠️  {symbol}: {exc}")
            break
        if not chunk:
            cur += step_ms
            continue
        rows.extend(chunk)
        cur = chunk[-1][0] + 1
        time.sleep(SLEEP_SEC)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "openTime", "open", "high", "low", "close", "volume",
            "closeTime", "quoteVol", "numTrades", "takerBaseVol", "takerQuoteVol", "ignore",
        ],
    )
    df["date"] = pd.to_datetime(df["openTime"], unit="ms")
    df = (
        df.set_index("date")[["open", "high", "low", "close", "volume", "quoteVol"]]
        .astype(float)
        .rename(columns={"quoteVol": "amount"})
    )
    df.sort_index(inplace=True)

    if df.empty:
        return df

    full_index = pd.date_range(df.index[0], df.index[-1], freq=freq_alias)
    df = df.reindex(full_index).ffill()
    df.index.name = "date"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Download crypto data with technical indicators")
    parser.add_argument("--granularity", "-g", default="1h", choices=list(GRANULARITY_CONFIG.keys()),
                        help="Data granularity (default: 1h)")
    parser.add_argument("--start", "-s", default="2018-02-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", default="2026-02-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing files")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "Data_TSC" / args.granularity
    output_dir.mkdir(parents=True, exist_ok=True)

    # Granularity config
    gran_cfg = GRANULARITY_CONFIG[args.granularity]
    interval = gran_cfg["binance"]
    freq_alias = gran_cfg["freq"]
    step_ms = gran_cfg["step_ms"]

    # Time range
    start_ms = int(pd.Timestamp(args.start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(args.end).timestamp() * 1000)

    print(f"\n{'='*60}")
    print(f"Crypto Download with Indicators")
    print(f"{'='*60}")
    print(f"  Granularity : {args.granularity}")
    print(f"  Start       : {args.start}")
    print(f"  End         : {args.end}")
    print(f"  Output      : {output_dir}")
    print(f"{'='*60}\n")

    for symbol in CRYPTO_SYMBOLS:
        # Remove USDT suffix for filename
        asset_name = symbol.replace("USDT", "")
        csv_path = output_dir / f"{asset_name}.csv"

        if csv_path.exists() and not args.force:
            print(f"  ⏭️  {asset_name}: already exists (use --force to overwrite)")
            continue

        print(f"  📥 {asset_name}...", end=" ", flush=True)

        # Fetch data
        df = fetch_binance_klines(symbol, interval, step_ms, freq_alias, start_ms, end_ms)
        if df.empty:
            print("No data")
            continue

        # Add indicators
        df = add_all_indicators(df)

        # Save
        df.to_csv(csv_path)
        print(f"✓ {len(df):,} rows")

    print(f"\n{'='*60}")
    print(f"Done! Files saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
