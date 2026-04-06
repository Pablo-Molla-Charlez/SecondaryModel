"""
Technical Indicators Module.

Provides reusable functions for computing technical indicators (RSI, MACD, Bollinger Bands,
ATR, ADX, Rate of Change) that can be used both:
1. When downloading data - compute indicators before saving
2. When data already exists - add indicators to existing CSVs

Also provides volatility/sentiment features for crypto:
  - Multi-window realized vol, ATR, RVI, MASSI
  - GARCH(1,1) volatility
  - Log return, DCR (Deribit Crypto Risk premium)
  - External features: DVOL, Fear & Greed Index, News Sentiment

Usage:
    from indicators import add_all_indicators, add_indicators_to_csv
    from indicators import add_crypto_xfeatures

    # For DataFrames:
    df = add_all_indicators(df)

    # Add crypto external + computed features:
    df = add_crypto_xfeatures(df, date_start="2023-01-01", date_end="2026-01-01")

    # For existing CSV files:
    add_indicators_to_csv(csv_path)
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# ┏━━━━━━━━━━ Base technical indicator columns (original 9) ━━━━━━━━━━┓
BASE_INDICATOR_COLUMNS = ['rsi_14', 'macd_histogram', 'bollinger_pct_b', 'bollinger_bandwidth',
                          'atr_14', 'atr_norm', 'adx_14', 'roc_5', 'roc_20']

# ┏━━━━━━━━━━ Crypto extra columns (trimmed: single 30-day window, no garch) ━━━━━━━━━━┓
CRYPTO_VOL_WINDOW = 30  # single representative window to avoid redundancy

CRYPTO_XFEATURE_COLUMNS: List[str] = [
    # External daily series (downloaded / cached)
    'dvol', 'fear_greed_idx', 'news_sentiment',
    # 30-day window vol features (one per indicator — avoids multi-window redundancy)
    'r_vol_30', 'r_vol_30_ann',
    'atr_w30', 'rvi_30', 'massi_30',
    # Scalar features
    'log_return', 'dcr',
]

# ┏━━━━━━━━━━ Full indicator columns list (base + crypto extras) ━━━━━━━━━━┓
INDICATOR_COLUMNS = BASE_INDICATOR_COLUMNS + CRYPTO_XFEATURE_COLUMNS

# ┏━━━━━━━━━━ Compute RSI ━━━━━━━━━━┓
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    RSI = 100 - [100 / (1 + RS)]
    RS = avg_gain / avg_loss

    Args:
        close: Series of closing prices
        period: Lookback period (default: 14)

    Returns:
        RSI values (0-100 scale)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # ┏━━━━━━━━━━ Exponential moving average ━━━━━━━━━━┓
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    # ┏━━━━━━━━━━ Avoid division by zero ━━━━━━━━━━┓
    rs = avg_gain / (avg_loss + 1e-10)  
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ┏━━━━━━━━━━ Compute MACD Histogram ━━━━━━━━━━┓
def compute_macd_histogram(close: pd.Series,
                           fast: int = 12,
                           slow: int = 26,
                           signal: int = 9) -> pd.Series:
    """
    Compute MACD Histogram.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line

    Args:
        close: Series of closing prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)

    Returns:
        MACD histogram values
    """
    # ┏━━━━━━━━━━ Exponential moving average ━━━━━━━━━━┓
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    # ┏━━━━━━━━━━ MACD Line ━━━━━━━━━━┓
    macd_line = ema_fast - ema_slow
    
    # ┏━━━━━━━━━━ Signal Line ━━━━━━━━━━┓
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # ┏━━━━━━━━━━ Histogram ━━━━━━━━━━┓
    histogram = macd_line - signal_line

    return histogram


# ┏━━━━━━━━━━ Compute Bollinger Bands ━━━━━━━━━━┓
def compute_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """
    Compute Bollinger Band indicators.

    Args:
        close: Series of closing prices
        period: Moving average period (default: 20)
        std_dev: Number of standard deviations (default: 2.0)

    Returns:
        Tuple of (pct_b, bandwidth):
        - pct_b: (close - lower) / (upper - lower), indicates where price is in band
        - bandwidth: (upper - lower) / middle, indicates band width
    """
    # ┏━━━━━━━━━━ Moving average ━━━━━━━━━━┓
    middle = close.rolling(window=period).mean()
    
    # ┏━━━━━━━━━━ Standard deviation ━━━━━━━━━━┓
    std = close.rolling(window=period).std()

    # ┏━━━━━━━━━━ Upper and lower bands ━━━━━━━━━━┓
    upper = middle + std_dev * std
    lower = middle - std_dev * std

    # ┏━━━━━━━━━━ %B: Where is price in the band? (0 = lower, 1 = upper, can exceed) ━━━━━━━━━━┓
    band_width = upper - lower
    pct_b = (close - lower) / (band_width + 1e-10)

    # ┏━━━━━━━━━━ Bandwidth: How wide is the band relative to price? ━━━━━━━━━━┓
    bandwidth = band_width / (middle + 1e-10)

    return pct_b, bandwidth


# ┏━━━━━━━━━━ Compute Average True Range (ATR) ━━━━━━━━━━┓
def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).

    TR = max(H-L, |H-prevC|, |L-prevC|)
    ATR = EMA(TR, period)

    Args:
        high, low, close: OHLC price series.
        period: Smoothing period (default: 14).

    Returns:
        ATR values per bar.
    """
    # ┏━━━━━━━━━━ True Range (TR) ━━━━━━━━━━┓
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                   (high - prev_close).abs(),
                   (low - prev_close).abs(),
                   ], axis=1).max(axis=1)

    # ┏━━━━━━━━━━ Exponential moving average ━━━━━━━━━━┓
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


# ┏━━━━━━━━━━ Compute Average Directional Index (ADX) ━━━━━━━━━━┓
def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX) — full standard Wilder method.

    Steps:
      1. +DM / -DM with mutual exclusion (larger wins, other is 0)
      2. +DI = 100 * EMA(+DM) / ATR
      3. -DI = 100 * EMA(-DM) / ATR
      4. DX  = 100 * |+DI - -DI| / (+DI + -DI)
      5. ADX = EMA(DX, period)

    Args:
        high, low, close: OHLC price series.
        period: Smoothing period (default: 14).

    Returns:
        ADX values (0-100 scale).
    """
    # ┏━━━━━━━━━━ Directional Movement (DM) ━━━━━━━━━━┓
    up_move = high.diff()
    down_move = -low.diff()

    # ┏━━━━━━━━━━ Mutual exclusion: whichever is larger and positive wins ━━━━━━━━━━┓
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),  index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    # ┏━━━━━━━━━━ Average True Range (ATR) ━━━━━━━━━━┓
    atr = compute_atr(high, low, close, period)

    # ┏━━━━━━━━━━ Exponential moving average ━━━━━━━━━━┓
    smooth_plus_dm = plus_dm.ewm(span=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(span=period, adjust=False).mean()

    # ┏━━━━━━━━━━ Directional Index (DI) ━━━━━━━━━━┓
    plus_di = 100.0 * smooth_plus_dm / (atr + 1e-10)
    minus_di = 100.0 * smooth_minus_dm / (atr + 1e-10)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx


# ┏━━━━━━━━━━ Compute Rate of Change (RoC) ━━━━━━━━━━┓
def compute_roc(close: pd.Series, period: int) -> pd.Series:
    """
    Compute Rate of Change.

    RoC = close / close.shift(period) - 1

    Args:
        close: Series of closing prices.
        period: Lookback bars.

    Returns:
        Rate of change values.
    """
    return close / close.shift(period) - 1.0


# ┏━━━━━━━━━━ Add all technical indicators to a DataFrame ━━━━━━━━━━┓
def add_all_indicators(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.

    Adds the following columns:
    - rsi_14: Relative Strength Index (14-period)
    - macd_histogram: MACD Histogram (12,26,9)
    - bollinger_pct_b: Bollinger %B (20-period, 2 std)
    - bollinger_bandwidth: Bollinger Bandwidth
    - atr_14: Average True Range (14-period)
    - atr_norm: ATR normalized by close price
    - adx_14: Average Directional Index (14-period)
    - roc_5: Rate of Change (5-bar)
    - roc_20: Rate of Change (20-bar)

    Args:
        df: DataFrame with OHLCV data (requires open, high, low, close, volume)
        close_col: Name of close price column

    Returns:
        DataFrame with indicator columns added
    """
    df = df.copy()
    close = df[close_col]
    high = df['high']
    low = df['low']

    df['rsi_14']         = compute_rsi(close, period=14)
    df['macd_histogram'] = compute_macd_histogram(close, fast=12, slow=26, signal=9)
    df['bollinger_pct_b'], df['bollinger_bandwidth'] = compute_bollinger_bands(close, period=20)
    df['atr_14']         = compute_atr(high, low, close, period=14)
    df['atr_norm']       = df['atr_14'] / (close + 1e-10)
    df['adx_14']         = compute_adx(high, low, close, period=14)
    df['roc_5']          = compute_roc(close, period=5)
    df['roc_20']         = compute_roc(close, period=20)

    return df


def has_indicators(df: pd.DataFrame) -> bool:
    """Check if DataFrame already has all indicator columns."""
    return all(col in df.columns for col in INDICATOR_COLUMNS)


# ┏━━━━━━━━━━ Add technical indicators to a CSV file ━━━━━━━━━━┓
def add_indicators_to_csv(csv_path: Union[str, Path], overwrite: bool = True, save: bool = True) -> pd.DataFrame:
    """
    Add technical indicators to an existing CSV file.

    Args:
        csv_path: Path to CSV file
        overwrite: If True, recompute even if indicators exist
        save: If True, save back to the same file

    Returns:
        DataFrame with indicators added
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, parse_dates=['date'] if 'date' in pd.read_csv(csv_path, nrows=1).columns else None)

    # Skip if already has indicators and not forcing overwrite
    if has_indicators(df) and not overwrite:
        return df

    # Add indicators
    df = add_all_indicators(df)

    # Save back
    if save:
        df.to_csv(csv_path, index=False)

    return df


# ┏━━━━━━━━━━ Process a directory of CSV files ━━━━━━━━━━┓
def process_directory(data_dir: Union[str, Path],
                      pattern: str = "*.csv",
                      overwrite: bool = True) -> None:
    """
    Add indicators to all CSVs in a directory.

    Args:
        data_dir: Directory containing CSV files
        pattern: Glob pattern for files (default: *.csv)
        overwrite: Whether to recompute existing indicators
    """
    # ┏━━━━━━━━━━ Convert to Path object ━━━━━━━━━━┓
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob(pattern))

    # ┏━━━━━━━━━━ Check if there are any CSV files ━━━━━━━━━━┓
    if not csv_files:
        print(f"No {pattern} files found in {data_dir}")
        return

    # ┏━━━━━━━━━━ Print processing information ━━━━━━━━━━┓
    print(f"Processing {len(csv_files)} files in {data_dir}")
    print("-" * 60)

    # ┏━━━━━━━━━━ Process each CSV file ━━━━━━━━━━┓
    for csv_path in csv_files:
        try:
            df = add_indicators_to_csv(csv_path, overwrite=overwrite)
            nan_counts = {col: df[col].isna().sum() for col in INDICATOR_COLUMNS if col in df.columns}
            print(f"  + {csv_path.name}: added indicators (NaN warmup: {nan_counts})")
        except Exception as e:
            print(f"  x {csv_path.name}: ERROR - {e}")

    print("-" * 60)
    print("Done!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRYPTO VOLATILITY & EXTERNAL FEATURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ┏━━━━━━━━━━ Detect granularity ━━━━━━━━━━┓
def _detect_granularity(date_series: pd.Series) -> str:
    """Detect if data is daily or intraday from timestamp spacing."""
    s = pd.to_datetime(date_series, errors="coerce").dropna().sort_values()
    if len(s) < 3:
        return "unknown"
    med = s.diff().dropna().median()
    return "daily" if med >= pd.Timedelta(days=1) else "intraday"

# ┏━━━━━━━━━━ Inference of bars per day ━━━━━━━━━━┓
def _infer_bars_per_day(date_series: pd.Series) -> int:
    """Approximate number of bars per calendar day (crypto = 24h market)."""
    s = pd.to_datetime(date_series, errors="coerce").dropna().sort_values()
    if len(s) < 3:
        return 1
    med = s.diff().dropna().median()
    if med is pd.NaT or med >= pd.Timedelta(days=1):
        return 1
    bars = int(round(pd.Timedelta(days=1) / med))
    return max(1, min(bars, 24 * 60))

# ┏━━━━━━━━━━ Compute Average True Range (ATR) simple ━━━━━━━━━━┓
def _compute_atr_simple(high: pd.Series, low: pd.Series,
                         close: pd.Series, length: int) -> pd.Series:
    """Simple rolling-mean ATR (no EMA smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

# ┏━━━━━━━━━━ Compute Relative Volatility Index (RVI) ━━━━━━━━━━┓
def _compute_rvi(close: pd.Series, length: int) -> pd.Series:
    """Relative Volatility Index (simplified: std of returns over window)."""
    returns = close.pct_change()
    std_up = returns.where(returns > 0, 0.0).rolling(length, min_periods=length).std()
    std_dn = returns.where(returns < 0, 0.0).abs().rolling(length, min_periods=length).std()
    rvi = 100.0 * std_up / (std_up + std_dn + 1e-10)
    return rvi

# ┏━━━━━━━━━━ Compute Mass Index ━━━━━━━━━━┓
def _compute_massi(high: pd.Series, low: pd.Series,
                    fast: int, slow: int) -> pd.Series:
    """Mass Index: sum of EMA(H-L, fast) / EMA(EMA(H-L, fast), slow)."""
    hl = high - low
    ema_fast = hl.ewm(span=fast, adjust=False).mean()
    ema_slow = ema_fast.ewm(span=slow, adjust=False).mean()
    ratio = ema_fast / (ema_slow + 1e-10)
    massi = ratio.rolling(25, min_periods=1).sum()
    return massi

# ┏━━━━━━━━━━ Recent volatility ━━━━━━━━━━┓
def _recent_volatility(price: pd.Series, nb_day: int = 360) -> float:
    """Annualised realized vol from log-returns (scalar)."""
    lr = np.log(price / price.shift(1))
    return float(np.sqrt((nb_day / len(price)) * np.nansum(lr.iloc[1:].to_numpy() ** 2)))

# ┏━━━━━━━━━━ Mean-reversion volatility path ━━━━━━━━━━┓
def _mr_vol(price: pd.Series, days: int = 30, nb_day: int = 360):
    """Mean-reversion volatility path (causal, no future data).

    Uses lagged vol pairs (vol[t-days] -> vol[t]) instead of the original
    forward-looking shift(-days), so no future prices are used.

    Returns (mr_vol_series, mr_adj_series, S_last, M_last).
    """
    data = pd.DataFrame({"Price": price})
    data["Recent_Vol"] = data["Price"].rolling(days, min_periods=days).apply(
        lambda x: _recent_volatility(pd.Series(x), nb_day=nb_day), raw=False)

    # ┏━━━━━━━━━━ Causal proxy: use (vol[t-days], vol[t]) pairs instead of (vol[t], vol[t+days]) ━━━━━━━━━━┓
    data["Lagged_Vol"] = data["Recent_Vol"].shift(days)

    # ┏━━━━━━━━━━ Drop rows with NaN values ━━━━━━━━━━┓
    vol_data = data.dropna(subset=["Recent_Vol", "Lagged_Vol"])
    if len(vol_data) < 20:
        empty = pd.Series(np.nan, index=data.index, dtype=float)
        return empty, empty, float("nan"), float("nan")

    # ┏━━━━━━━━━━ Create volatility buckets ━━━━━━━━━━┓
    buckets = pd.qcut(vol_data["Lagged_Vol"], q=min(20, len(vol_data) // 3), labels=False, duplicates="drop")
    avg = (vol_data.groupby(buckets).agg({"Lagged_Vol": "mean", "Recent_Vol": "mean"})
           .rename(columns={"Lagged_Vol": "Avg_Lagged", "Recent_Vol": "Avg_Recent"})
           .reset_index(drop=True))

    # ┏━━━━━━━━━━ Get the values of the buckets ━━━━━━━━━━┓
    x, y = avg["Avg_Lagged"].values, avg["Avg_Recent"].values
    mask = np.isfinite(x) & np.isfinite(y)

    # ┏━━━━━━━━━━ Check if there are enough data points ━━━━━━━━━━┓
    if mask.sum() < 2:
        empty = pd.Series(np.nan, index=data.index, dtype=float)
        return empty, empty, float("nan"), float("nan")

    # ┏━━━━━━━━━━ Compute slope and intercept ━━━━━━━━━━┓
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    if abs(slope - 1.0) < 1e-8:
        empty = pd.Series(np.nan, index=data.index, dtype=float)
        return empty, empty, float("nan"), float("nan")

    # ┏━━━━━━━━━━ Compute S and M ━━━━━━━━━━┓
    S = 1 - slope
    M = intercept / (1 - slope)
    mr = data["Recent_Vol"] + S * (M - data["Recent_Vol"])
    mr_adj = mr - data["Recent_Vol"]
    return mr, mr_adj, S, M

# ┏━━━━━━━━━━ Compute Variance Premium (VP) and EVIX ━━━━━━━━━━┓
def _vp_evix(price: pd.Series, implied_vol: pd.Series,
              days: int = 30, nb_day: int = 360):
    """Variance premium / EVIX.  Returns (EVIX, VP, vol_prem)."""
    # ┏━━━━━━━━━━ Create DataFrame with Price and Implied Volatility ━━━━━━━━━━┓
    data = pd.DataFrame({"Price": price, "IV": implied_vol}).dropna()
    
    # ┏━━━━━━━━━━ Check if there are enough data points ━━━━━━━━━━┓
    if len(data) < 2 * days:
        empty = pd.Series(np.nan, index=price.index, dtype=float)
        return empty, empty, empty

    # ┏━━━━━━━━━━ Compute Mean-Reversion Volatility (MR_Vol) ━━━━━━━━━━┓
    mr, _, _, _ = _mr_vol(data["Price"], days, nb_day)
    data["MR_Vol"] = mr
    
    # ┏━━━━━━━━━━ Compute Variance Premium (VP) ━━━━━━━━━━┓
    data["VP"] = (data["IV"] ** 2) - (data["MR_Vol"] ** 2)
    data = data.dropna(subset=["MR_Vol", "VP"])

    # ┏━━━━━━━━━━ Check if there are enough data points ━━━━━━━━━━┓
    if len(data) < 20:
        empty = pd.Series(np.nan, index=price.index, dtype=float)
        return empty, empty, empty

    # ┏━━━━━━━━━━ Create volatility buckets ━━━━━━━━━━┓
    buckets = pd.qcut(data["MR_Vol"], q=20, labels=False, duplicates="drop")
    avg = (data.groupby(buckets)
           .agg({"MR_Vol": "mean", "VP": "mean"})
           .rename(columns={"MR_Vol": "Avg_MR", "VP": "Avg_VP"})
           .reset_index(drop=True))

    # ┏━━━━━━━━━━ Get the values of the buckets ━━━━━━━━━━┓
    avg["MR_sq"] = avg["Avg_MR"] ** 2
    x, y = avg["MR_sq"].values, avg["Avg_VP"].values
    mask = np.isfinite(x) & np.isfinite(y)
    
    # ┏━━━━━━━━━━ Check if there are enough data points ━━━━━━━━━━┓
    if mask.sum() < 2:
        empty = pd.Series(np.nan, index=price.index, dtype=float)
        return empty, empty, empty

    # ┏━━━━━━━━━━ Compute slope and intercept ━━━━━━━━━━┓
    slope_c, intercept_d = np.polyfit(x[mask], y[mask], 1)
    
    # ┏━━━━━━━━━━ Compute EVIX and Variance Premium (VP) ━━━━━━━━━━┓
    VP = slope_c * data["MR_Vol"] ** 2 + intercept_d
    EVIX = np.sqrt(data["MR_Vol"] ** 2 + VP)
    vol_prem = EVIX - data["MR_Vol"]
    return EVIX, VP, vol_prem

# ┏━━━━━━━━━━ Compute Deribit Crypto Risk premium (DCR) ━━━━━━━━━━┓
def _compute_dcr(price: pd.Series, dvol: pd.Series,
                  days: int = 30, nb_day: int = 360) -> pd.Series:
    """Deribit Crypto Risk premium (DCR = MR_adj + DTM).

    Where DTM = implied_vol - EVIX.
    """
    data = pd.DataFrame({"Price": price, "DVOL": dvol}).dropna()
    if len(data) < 2 * days:
        return pd.Series(np.nan, index=price.index, dtype=float, name="dcr")

    _, mr_adj, _, _ = _mr_vol(data["Price"], days, nb_day)
    evix, _, _ = _vp_evix(data["Price"], data["DVOL"], days, nb_day)
    dtm = data["DVOL"] - evix
    dcr = mr_adj + dtm
    dcr.name = "dcr"
    return dcr


# ┏━━━━━━━━━━ Main enrichment function ━━━━━━━━━━┓
def add_crypto_xfeatures(df: pd.DataFrame,
                          asset: str = "BTC",
                          btc_close_daily: Optional[pd.Series] = None,
                          date_start: Optional[str] = None,
                          date_end: Optional[str] = None,
                          lag_days: int = 1,
                          force_refresh: bool = False,
                          verbose: bool = False) -> pd.DataFrame:
    """Add crypto volatility + external features to a per-asset DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: date, open, high, low, close, volume.
    asset : str
        Asset symbol (e.g. 'BTCUSDT', 'SOLUSDT'). Used for per-symbol DVOL
        loading and vol-beta proxy computation.
    btc_close_daily : pd.Series, optional
        Daily BTC close prices (DatetimeIndex). Needed for vol-beta DVOL proxy
        on non-BTC/ETH assets. If None, proxy is skipped for altcoins.
    date_start, date_end : str, optional
        Date range for external feature download.  Auto-detected from df if not given.
    lag_days : int
        Conservative lag for daily external features on intraday bars (default 1).
    force_refresh : bool
        Re-download external features even if cached.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Original df with CRYPTO_XFEATURE_COLUMNS appended (in-place safe).
    """
    df = df.copy()

    # ┏━━━━━━━━━━ Parse dates ━━━━━━━━━━┓
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    close = pd.to_numeric(df["close"], errors="coerce")
    high  = pd.to_numeric(df["high"],  errors="coerce")
    low   = pd.to_numeric(df["low"],   errors="coerce")

    # ┏━━━━━━━━━━ Detect granularity ━━━━━━━━━━┓
    bars_per_day = _infer_bars_per_day(df["date"])
    nb_days_in_year = 360  # crypto convention
    periods_per_year = nb_days_in_year * max(1, bars_per_day)

    if verbose:
        print(f"[add_crypto_xfeatures] asset={asset}, bars_per_day={bars_per_day}")

    # ┏━━━━━━━━━━ 1. Log return ━━━━━━━━━━┓
    df["log_return"] = np.log(close / close.shift(1))

    # ┏━━━━━━━━━━ 2. Single 30-day window vol features ━━━━━━━━━━┓
    d = CRYPTO_VOL_WINDOW
    w = max(1, int(d * max(1, bars_per_day)))

    rvol = df["log_return"].rolling(w, min_periods=w).apply(lambda x: float(np.sqrt(np.nansum(np.asarray(x) ** 2))), raw=True)
    df["r_vol_30"]     = rvol
    df["r_vol_30_ann"] = rvol * np.sqrt(periods_per_year)
    df["atr_w30"]      = _compute_atr_simple(high, low, close, length=w)
    df["rvi_30"]       = _compute_rvi(close, length=w)
    df["massi_30"]     = _compute_massi(high, low, fast=w, slow=w + 16)

    # ┏━━━━━━━━━━ 3. External features (daily, lagged to avoid leakage) ━━━━━━━━━━┓
    if date_start is None:
        date_start = str(df["date"].min().date() - pd.Timedelta(days=lag_days + 5))
    if date_end is None:
        date_end = str(df["date"].max().date() + pd.Timedelta(days=1))

    # ┏━━━━━━━━━━ 3a. Market-wide features (fear & greed, news sentiment) ━━━━━━━━━━┓
    try:
        from Data_MLA.XFeatures.xfeatures import load_xfeatures
        xf = load_xfeatures(date_start, date_end, force_refresh=force_refresh)
    except Exception as exc:
        warnings.warn(f"[add_crypto_xfeatures] XFeatures load failed: {exc}")
        xf = pd.DataFrame()

    # ┏━━━━━━━━━━ 3b. Per-asset DVOL (or vol-beta proxy) ━━━━━━━━━━┓
    try:
        from Data_MLA.XFeatures.xfeatures import load_dvol_for_asset
        # ┏━━━━━━━━━━ Build a daily close series for the asset (for vol-beta proxy) ━━━━━━━━━━┓
        daily_close = close.copy()
        daily_close.index = df["date"]
        daily_close = daily_close.resample("D").last().dropna()

        # ┏━━━━━━━━━━ Load DVOL for the asset ━━━━━━━━━━┓
        dvol_series = load_dvol_for_asset(asset      = asset,
                                          date_start = date_start,
                                          date_end   = date_end,
                                          asset_close= daily_close,
                                          btc_close  = btc_close_daily)
        
        # ┏━━━━━━━━━━ Add to xf DataFrame ━━━━━━━━━━┓
        if len(dvol_series) > 0:
            if xf.empty:
                xf = dvol_series.to_frame()
            else:
                xf = xf.join(dvol_series, how="left")
    except Exception as exc:
        warnings.warn(f"[add_crypto_xfeatures] DVOL load failed for {asset}: {exc}")

    # ┏━━━━━━━━━━ Merge external features with conservative lag ━━━━━━━━━━┓
    if not xf.empty:
        xf_shifted = xf.copy()
        xf_shifted.index = xf_shifted.index + pd.Timedelta(days=lag_days)

        df["_merge_day"] = df["date"].dt.floor("D")
        xf_shifted = xf_shifted.reset_index().rename(columns={"date": "_merge_day"})
        xf_shifted["_merge_day"] = pd.to_datetime(xf_shifted["_merge_day"])

        for col in ["fear_greed_idx", "dvol", "news_sentiment"]:
            if col in xf_shifted.columns:
                mapping = xf_shifted[["_merge_day", col]].drop_duplicates("_merge_day", keep="last")
                mapping = mapping.set_index("_merge_day")[col]
                df[col] = df["_merge_day"].map(mapping)
            else:
                df[col] = np.nan

        df = df.drop(columns=["_merge_day"])
    else:
        for col in ["fear_greed_idx", "dvol", "news_sentiment"]:
            df[col] = np.nan

    # ┏━━━━━━━━━━ 4. DCR (requires DVOL) ━━━━━━━━━━┓
    if df["dvol"].notna().sum() > 60:
        df["dcr"] = _compute_dcr(close, df["dvol"], days=30, nb_day=nb_days_in_year)
    else:
        df["dcr"] = np.nan

    # ┏━━━━━━━━━━ Ensure all expected columns exist ━━━━━━━━━━┓
    for col in CRYPTO_XFEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    if verbose:
        n_feats = len(CRYPTO_XFEATURE_COLUMNS)
        na_pct = {c: f"{df[c].isna().mean():.0%}" for c in CRYPTO_XFEATURE_COLUMNS}
        print(f"[add_crypto_xfeatures] Added {n_feats} features. NaN%: {na_pct}")

    return df


def main():
    """Command-line interface for adding indicators to existing CSVs."""
    import argparse

    parser = argparse.ArgumentParser(description="Add technical indicators to existing CSV files.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing CSV files to process")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for files (default: *.csv)")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip files that already have indicator columns")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    process_directory(data_dir  = data_dir,
                      pattern   = args.pattern,
                      overwrite = not args.no_overwrite)


if __name__ == "__main__":
    main()
