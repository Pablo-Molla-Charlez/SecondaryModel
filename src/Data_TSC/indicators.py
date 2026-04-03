"""
Technical Indicators Module.

Provides reusable functions for computing technical indicators (RSI, MACD, Bollinger Bands,
ATR, ADX, Rate of Change) that can be used both:
1. When downloading data - compute indicators before saving
2. When data already exists - add indicators to existing CSVs

Usage:
    from indicators import add_all_indicators, add_indicators_to_csv

    # For DataFrames:
    df = add_all_indicators(df)

    # For existing CSV files:
    add_indicators_to_csv(csv_path)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence, Union

INDICATOR_COLUMNS = ['rsi_14', 'macd_histogram', 'bollinger_pct_b', 'bollinger_bandwidth',
                     'atr_14', 'atr_norm', 'adx_14', 'roc_5', 'roc_20']

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
