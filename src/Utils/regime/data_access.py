"""Raw OHLCV(A) access for the regime package (Stage 0).

A single reader for ``Data_MLA/Raw/<granularity>/<ASSET>.csv`` so the entrypoint
loader and the cross-asset feature builders share one access contract (path
layout, date parsing, sort order, column validation) instead of re-deriving it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from Utils.regime.paths import RAW_ROOT


# ┏━━━━━━━━━━ Cached CSV read (parse once, reuse across paths) ━━━━━━━━━━┓
# The cross-asset cortese builder re-reads the same 4 peer CSVs (BTC/ETH/LTC/BCH) once per
# synthetic path — 4×N reads of immutable files (N=1000 Kronos, ~891k merton) dominated the
# regime cache build (~75% of wall-clock, measured). The raw CSVs don't change during a build,
# so parse each exactly once. The cache key includes the file mtime, so editing/replacing a CSV
# transparently invalidates its entry. The DataFrame is cached and a COPY returned, so callers
# that mutate (set_index, dropna, column adds) can't corrupt the shared cached frame.
@lru_cache(maxsize=64)
def _read_ohlcv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    return (pd.read_csv(path_str, parse_dates=["date"])
            .set_index("date").sort_index())


def clear_ohlcv_cache() -> None:
    """Drop the in-process raw-OHLCV read cache (e.g. after regenerating a CSV mid-session)."""
    _read_ohlcv_cached.cache_clear()


def load_ohlcv(asset: str,
               granularity: str,
               root: str | Path | None = None,
               required_columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Load an asset's raw OHLCV(A) history, date-indexed and sorted.

    Reads are memoized by (resolved path, mtime) so the same CSV is parsed at most once per
    process while a build runs; a copy is returned so callers may mutate freely.

    Args:
        asset: symbol whose ``<asset>.csv`` to read (e.g. "BTCUSDT").
        granularity: bar granularity subfolder (e.g. "4h").
        root: raw-data root override; defaults to ``Data_MLA/Raw``.
        required_columns: if given, columns that must be present — a missing one
            raises ``ValueError`` rather than failing later inside a builder.
    """
    base = Path(root) if root is not None else RAW_ROOT
    path = base / granularity / f"{asset}.csv"
    if not path.exists():
        # The raw tree spells sub-hour folders "30min"/"15min" while BARS_PER_DAY and the
        # rest of the pipeline key on "30m"/"15m". Accept either spelling here rather than
        # renaming the data or symlinking a shim root.
        alt = None
        if granularity.endswith("min"):
            alt = granularity[:-2]              # 30min -> 30m
        elif granularity.endswith("m") and not granularity.endswith("min"):
            alt = f"{granularity}in"            # 30m -> 30min
        if alt is not None:
            alt_path = base / alt / f"{asset}.csv"
            if alt_path.exists():
                path = alt_path
    if not path.exists():
        raise FileNotFoundError(f"No raw OHLCV at {path}")
    df = _read_ohlcv_cached(str(path), path.stat().st_mtime).copy()
    if required_columns:
        missing = sorted(set(required_columns) - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
    return df


__all__ = ["load_ohlcv", "clear_ohlcv_cache"]
