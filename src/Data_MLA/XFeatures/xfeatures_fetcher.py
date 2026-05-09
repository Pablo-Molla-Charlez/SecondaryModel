"""
XFeatures Live Fetcher — downloads and caches external features for live trading.

Sources:
  1. DVOL (BTC & ETH): CryptoDataDownload CSVs
  2. Fear & Greed Index: alternative.me JSON API
  3. News Sentiment: FRBSF xlsx (weekly updates, daily data)

Run this as a cron job (e.g. daily at 00:30 UTC) or call fetch_all() programmatically.

Cache files written to XFEATURES_DIR:
  - xfeatures_cache.json   (latest values for live consumption)
  - xfeatures_status.json  (health/staleness info)
  - BTC_DVOL.csv, ETH_DVOL.csv (full history, overwritten on success)
  - Fear_Greed_Index.csv   (full history from API)
  - News_Sentiment_Data.csv (full history from FRBSF)
"""

from __future__ import annotations

import json
import requests
import warnings
import numpy as np
import pandas as pd

from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path

# ┏━━━━━━━━━━ Cache directory ━━━━━━━━━━┓
XFEATURES_DIR = Path(__file__).resolve().parent
CACHE_FILE = XFEATURES_DIR / "xfeatures_cache.json"
STATUS_FILE = XFEATURES_DIR / "xfeatures_status.json"

TIMEOUT = 30


# ---------------------------------------------------------------------------
# Individual Fetchers
# ---------------------------------------------------------------------------

# ┏━━━━━━━━━━ Fetcher for the Deribit Volatility Index (DVOL) [BTC and ETH] ━━━━━━━━━━┓
def fetch_dvol() -> dict:
    """Fetch BTC and ETH DVOL from CryptoDataDownload. Returns latest close values."""
    results = {}
    for symbol in ("BTC", "ETH"):
        url = f"https://www.cryptodatadownload.com/cdd/DeriBit_volatility_OHLC_{symbol}.csv"
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            lines = resp.text.split("\n")
            skip = 1 if "cryptodatadownload" in lines[0].lower() else 0
            df = pd.read_csv(StringIO("\n".join(lines[skip:])))
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            # ┏━━━━━━━━━━ Save full CSV for offline use ━━━━━━━━━━┓
            csv_path = XFEATURES_DIR / f"{symbol}_DVOL.csv"
            df.to_csv(csv_path, index=False)

            # ┏━━━━━━━━━━ Latest value (yesterday's close since data is lagged by 1 day in pipeline) ━━━━━━━━━━┓
            latest = df.iloc[-1]
            results[f"dvol_{symbol.lower()}"] = {"value": float(latest["close"]),
                                                 "date": str(latest["date"].date())}
        except Exception as exc:
            warnings.warn(f"[XFeatures Fetcher] DVOL {symbol} failed: {exc}")
            results[f"dvol_{symbol.lower()}"] = None

    return results


# ┏━━━━━━━━━━ Fetcher for the Fear & Greed Index ━━━━━━━━━━┓
def fetch_fear_greed() -> dict | None:
    """Fetch Fear & Greed Index from alternative.me API."""
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        records = data.get("data", [])

        if not records:
            return None

        # ┏━━━━━━━━━━ Build DataFrame and save full history ━━━━━━━━━━┓
        rows = []
        for r in records:
            try:
                ts = pd.Timestamp(int(r["timestamp"]), unit="s")
                rows.append({"date": ts.strftime("%Y-%m-%d"), "fear_greed_idx": int(r["value"])})
            except Exception:
                continue

        # ┏━━━━━━━━━━ Dataframe and save full history ━━━━━━━━━━┓
        df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
        df.to_csv(XFEATURES_DIR / "Fear_Greed_Index.csv", index=False)

        # Latest value
        latest = df.iloc[-1]
        return {"value": int(latest["fear_greed_idx"]), "date": latest["date"]}

    except Exception as exc:
        warnings.warn(f"[XFeatures Fetcher] Fear & Greed failed: {exc}")
        return None


# ┏━━━━━━━━━━ Fetcher for the News Sentiment ━━━━━━━━━━┓
def fetch_news_sentiment() -> dict | None:
    """Fetch News Sentiment from FRBSF xlsx (Data sheet)."""
    url = "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx"
    try:
        # ┏━━━━━━━━━━ Fetch data from FRBSF ━━━━━━━━━━┓
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()

        # ┏━━━━━━━━━━ Read Excel file ━━━━━━━━━━┓
        df = pd.read_excel(BytesIO(resp.content), sheet_name="Data")

        # ┏━━━━━━━━━━ Identify columns ━━━━━━━━━━┓
        dt_col = df.columns[0]
        val_col = next((c for c in df.columns if "sentiment" in c.lower()), df.columns[1])
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col])
        df = df.rename(columns={dt_col: "date", val_col: "News Sentiment"})
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df = df[["date", "News Sentiment"]].sort_values("date")

        # ┏━━━━━━━━━━ Save as CSV ━━━━━━━━━━┓
        df.to_csv(XFEATURES_DIR / "News_Sentiment_Data.csv", index=False)

        # ┏━━━━━━━━━━ Latest value ━━━━━━━━━━┓
        latest = df.iloc[-1]
        return {"value": float(latest["News Sentiment"]), "date": latest["date"]}

    except Exception as exc:
        warnings.warn(f"[XFeatures Fetcher] News Sentiment failed: {exc}")
        return None


# ┏━━━━━━━━━━ Fetch all external features ━━━━━━━━━━┓
def fetch_all() -> dict:
    """Fetch all external features, update cache and status files.

    Returns the cache dict for immediate use.
    """
    now = datetime.now(timezone.utc).isoformat()

    # ┏━━━━━━━━━━ Load previous cache for fallback ━━━━━━━━━━┓
    prev_cache = {}
    if CACHE_FILE.exists():
        try:
            prev_cache = json.loads(CACHE_FILE.read_text())
        except Exception:
            pass

    # ┏━━━━━━━━━━ Load previous status ━━━━━━━━━━┓
    prev_status = {}
    if STATUS_FILE.exists():
        try:
            prev_status = json.loads(STATUS_FILE.read_text())
        except Exception:
            pass

    # ┏━━━━━━━━━━ Fetch each source ━━━━━━━━━━┓
    dvol_result = fetch_dvol()
    fng_result = fetch_fear_greed()
    sent_result = fetch_news_sentiment()

    # ┏━━━━━━━━━━ Build cache (use previous value as fallback) ━━━━━━━━━━┓
    cache = {}
    status = {}

    # ┏━━━━━━━━━━ DVOL BTC ━━━━━━━━━━┓
    if dvol_result.get("dvol_btc"):
        cache["dvol_btc"] = dvol_result["dvol_btc"]["value"]
        cache["dvol_btc_date"] = dvol_result["dvol_btc"]["date"]
        status["dvol_btc"] = {"last_success": now, "stale": False}
    else:
        cache["dvol_btc"] = prev_cache.get("dvol_btc")
        cache["dvol_btc_date"] = prev_cache.get("dvol_btc_date")
        status["dvol_btc"] = {"last_success": prev_status.get("dvol_btc", {}).get("last_success"),
                              "stale": True,
                              "error_at": now}

    # ┏━━━━━━━━━━ DVOL ETH ━━━━━━━━━━┓
    if dvol_result.get("dvol_eth"):
        cache["dvol_eth"] = dvol_result["dvol_eth"]["value"]
        cache["dvol_eth_date"] = dvol_result["dvol_eth"]["date"]
        status["dvol_eth"] = {"last_success": now, "stale": False}
    else:
        cache["dvol_eth"] = prev_cache.get("dvol_eth")
        cache["dvol_eth_date"] = prev_cache.get("dvol_eth_date")
        status["dvol_eth"] = {"last_success": prev_status.get("dvol_eth", {}).get("last_success"),
                              "stale": True,
                              "error_at": now}

    # ┏━━━━━━━━━━ Fear & Greed ━━━━━━━━━━┓
    if fng_result:
        cache["fear_greed_idx"] = fng_result["value"]
        cache["fear_greed_date"] = fng_result["date"]
        status["fear_greed"] = {"last_success": now, "stale": False}
    else:
        cache["fear_greed_idx"] = prev_cache.get("fear_greed_idx")
        cache["fear_greed_date"] = prev_cache.get("fear_greed_date")
        status["fear_greed"] = {"last_success": prev_status.get("fear_greed", {}).get("last_success"),
                                "stale": True,
                                "error_at": now}

    # ┏━━━━━━━━━━ News Sentiment ━━━━━━━━━━┓
    if sent_result:
        cache["news_sentiment"] = sent_result["value"]
        cache["news_sentiment_date"] = sent_result["date"]
        status["news_sentiment"] = {"last_success": now, "stale": False}
    else:
        cache["news_sentiment"] = prev_cache.get("news_sentiment")
        cache["news_sentiment_date"] = prev_cache.get("news_sentiment_date")
        status["news_sentiment"] = {"last_success": prev_status.get("news_sentiment", {}).get("last_success"),
                                    "stale": True,
                                    "error_at": now}

    # ┏━━━━━━━━━━ Write cache and status ━━━━━━━━━━┓
    cache["last_fetched"] = now
    CACHE_FILE.write_text(json.dumps(cache, indent=2))
    STATUS_FILE.write_text(json.dumps(status, indent=2))

    print(f"[XFeatures Fetcher] Done at {now}")
    for key in ("dvol_btc", "dvol_eth", "fear_greed", "news_sentiment"):
        s = status[key]
        if s.get("stale"):
            print(f"  WARNING: {key} is STALE (last success: {s.get('last_success', 'never')})")
        else:
            print(f"  OK: {key}")

    return cache


# ┏━━━━━━━━━━ Load cached xfeatures values ━━━━━━━━━━┓
def load_cache() -> dict:
    """Load the cached xfeatures values. Returns empty dict if no cache exists."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


# ┏━━━━━━━━━━ Load status file ━━━━━━━━━━┓
def load_status() -> dict:
    """Load the status file. Returns empty dict if no status exists."""
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except Exception:
            return {}
    return {}


if __name__ == "__main__":
    fetch_all()
