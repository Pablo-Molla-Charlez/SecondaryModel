import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.utils.data
from pathlib import Path
from typing import List, Tuple, Union, Sequence, Optional, Dict, Any


# ┏━━━━━━━━━━ Dataset Access Helpers ━━━━━━━━━━┓
def _get_from_dataset(dataset: Any, key: str) -> Any:
    """Helper to access keys from both dicts and MultiGranDataset objects."""
    if isinstance(dataset, dict):
        return dataset.get(key)
    return getattr(dataset, key, None)

# ┏━━━━━━━━━━ Dynamic Return Limits for Plotting ━━━━━━━━━━┓
def get_dynamic_ret_limits(returns_list: List[np.ndarray], min_buffer: float = 5.0) -> Tuple[float, float]:
    """
    Calculate symmetric x-limits based on data percentiles (1st and 99th),
    ensuring a minimum visibility range of +/- min_buffer%.
    """
    import numpy as np
    # Filter out empty arrays or NaNs
    valid_data = []
    for r in returns_list:
        if r is not None and len(r) > 0:
            v = r[~np.isnan(r)]
            if len(v) > 0:
                valid_data.append(v)
    
    if not valid_data:
        return -min_buffer, min_buffer
    
    all_data = np.concatenate(valid_data)
    p1 = np.percentile(all_data, 1)
    p99 = np.percentile(all_data, 99)
    # Use symmetric limit for balanced view
    limit = max(abs(p1), abs(p99), min_buffer)
    # Round up to nearest integer for cleaner axis
    limit = float(np.ceil(limit))
    return -limit, limit


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MULTI-GRANULARITY SUPPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Granularity → sequence length (context window for M2).
# Must match the seq_len used by M1 when generating tokens / meta-labels.
GRAN_SEQ_LEN = {"1d": 40, "12h": 60, "8h": 65, "6h": 75, "4h": 90,  
                "2h": 60, "1h": 80, "30m": 90, "15m": 160, "5m": 480}

# Canonical ordering (coarsest → finest) and integer IDs
GRAN_ORDER = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m", "15m", "5m"]
GRAN_TO_ID = {g: i for i, g in enumerate(GRAN_ORDER)}


class MultiGranDataset(torch.utils.data.Dataset):
    """
    Flat-indexed dataset that wraps per-granularity sub-datasets.

    Each granularity has its own tensors with different seq_len.  A flat
    index 'i' maps to '(granularity, local_index)' via an internal
    offset table. This guarantees that '__getitem__' returns tensors
    of the correct shape for that granularity.

    Public attributes (flat, used for splitting / balancing):
        labels   : torch.Tensor  (N_total,)
        dates    : list          (N_total,)
        returns  : torch.Tensor  (N_total,)
        gran_ids : torch.Tensor  (N_total,)
        asset_ids: torch.Tensor  (N_total,)
    """
    def __init__(self, per_gran: Dict[str, Dict[str, Any]]):
        """
        Parameters
        ----------
        per_gran : dict
            {granularity_str: dataset_dict} where each
            dataset_dict comes from prepare_multi_asset_dataset.
        """
        print(f"\n┏━━━━━━━━━━ Building MultiGranDataset ━━━━━━━━━━┓")
        self.grans: List[str] = []          # ordered gran names
        self.sub:   Dict[str, Any] = {}     # per-gran dataset dicts
        self._offsets: List[int]   = []     # cumulative offsets

        # ┏━━━━━━━━━━ Build offset table ━━━━━━━━━━┓
        flat_labels, flat_dates, flat_returns = [], [], []
        flat_gran_ids, flat_asset_ids = [], []
        flat_m1_pred_returns, flat_m1_pred_labels, flat_m1_true_labels = [], [], []
        flat_eng_features = []
        offset = 0
        for g in GRAN_ORDER:
            if g not in per_gran:
                continue
            ds = per_gran[g]
            n  = len(ds['labels'])
            self.grans.append(g)
            self.sub[g] = ds
            self._offsets.append(offset)

            flat_labels.append(ds['labels'])
            flat_dates.extend(ds['dates'])
            flat_returns.append(ds['returns'])
            flat_gran_ids.append(torch.full((n,), GRAN_TO_ID[g], dtype=torch.long))
            flat_asset_ids.append(ds['asset_ids'])
            flat_m1_pred_returns.append(ds.get('m1_pred_returns', torch.full((n,), float('nan'))))
            flat_m1_pred_labels.append(ds.get('m1_pred_labels', torch.full((n,), float('nan'))))
            flat_m1_true_labels.append(ds.get('m1_true_labels', torch.full((n,), float('nan'))))
            if 'eng_features' in ds:
                flat_eng_features.append(ds['eng_features'])
            offset += n

        self._offsets.append(offset)  # sentinel
        self._total = offset

        # ┏━━━━━━━━━━ Flat public tensors ━━━━━━━━━━┓
        self.labels    = torch.cat(flat_labels)
        self.dates     = flat_dates
        self.returns   = torch.cat(flat_returns)
        self.gran_ids  = torch.cat(flat_gran_ids)
        self.asset_ids = torch.cat(flat_asset_ids)
        self.m1_pred_returns = torch.cat(flat_m1_pred_returns)
        self.m1_pred_labels  = torch.cat(flat_m1_pred_labels)
        self.m1_true_labels  = torch.cat(flat_m1_true_labels)
        self.eng_features    = torch.cat(flat_eng_features) if flat_eng_features else None

        # ┏━━━━━━━━━━ Build combined asset_map ━━━━━━━━━━┓
        self.asset_map = {}
        for g in self.grans:
            self.asset_map.update(self.sub[g].get('asset_map', {}))

        # ┏━━━━━━━━━━ Print summary ━━━━━━━━━━┓
        for i, g in enumerate(self.grans):
            n = self._offsets[i+1] - self._offsets[i]
            print(f"  [{g}] {n:,} windows  (seq_len={GRAN_SEQ_LEN.get(g, '?')})")
        print(f"  [TOTAL] {self._total:,} windows (including meta_label with False/NaN values) across {len(self.grans)} granularities.")

    # ┏━━━━━━━━━━ Flat → (gran, local) ━━━━━━━━━━┓
    def _resolve(self, flat_idx: int):
        for i, g in enumerate(self.grans):
            if flat_idx < self._offsets[i+1]:
                return g, flat_idx - self._offsets[i]
        raise IndexError(f"Index {flat_idx} out of range (total={self._total})")

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        g, local = self._resolve(idx)
        ds = self.sub[g]

        output = {'ohlcv':     ds['ohlcv'][local],
                  'labels':    ds['labels'][local],
                  'time_ids':  {k: v[local] for k, v in ds['time_ids'].items()},
                  'asset_ids': ds['asset_ids'][local],
                  'returns':   ds['returns'][local],
                  'gran_id':   GRAN_TO_ID[g],
                  'orig_idx':  idx}

        if 's1_ids' in ds and ds['s1_ids'] is not None:
            output['s1_ids'] = ds['s1_ids'][local]
            output['s2_ids'] = ds['s2_ids'][local]

        # RRS (Regime Rarity Score) — attached by token_regime.attach_rrs_to_dataset()
        if hasattr(self, 'rrs') and self.rrs is not None:
            output['rrs'] = self.rrs[idx]

        # Engineered window-level features
        if self.eng_features is not None:
            output['eng_features'] = self.eng_features[idx]

        return output


def prepare_multi_gran_dataset(combined_df: pd.DataFrame,
                               column_features: Sequence[str],
                               target_col: str,
                               forecast_horizon: int,
                               cfg: Dict[str, Any]) -> 'MultiGranDataset':
    """
    Build a MultiGranDataset from a DataFrame containing multiple granularities.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Must have a 'granularity' column (added by 'load_dataset_from_config'
        when 'granularity == "all"').
    column_features, target_col, forecast_horizon, cfg :
        Forwarded to 'prepare_multi_asset_dataset' for each granularity group.

    Returns
    -------
    MultiGranDataset
    """
    if 'granularity' not in combined_df.columns:
        raise ValueError("DataFrame must have a 'granularity' column for multi-gran mode.")

    print(f"\n┏━━━━━━━━━━ Pre-Processing {combined_df['granularity'].unique()} Granularities: ━━━━━━━━━━┓")
    per_gran = {}
    for g, g_df in combined_df.groupby('granularity'):
        seq_len = GRAN_SEQ_LEN.get(g)
        if seq_len is None:
            print(f"  [WARN] Unknown granularity '{g}', skipping.")
            continue
        print(f"[prepare_multi_gran_dataset] Processing {g} (seq_len={seq_len}, rows={len(g_df):,}).")
        ds = prepare_multi_asset_dataset(g_df,
                                         seq_len            = seq_len,
                                         column_features    = column_features,
                                         target_col         = target_col,
                                         forecast_horizon   = forecast_horizon,
                                         cfg                = cfg)
        per_gran[g] = ds
    print(f"┏━━━━━━━━━━ Finished: Pre-Processing Granularities ━━━━━━━━━━┓")
    return MultiGranDataset(per_gran)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MULTI-ASSET DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_multi_asset_dataset(data_dir: Union[str, Path], 
                             direction: str = "up", 
                             assets: Optional[List[str]] = None,) -> pd.DataFrame:
    """
    Load all assets from a directory into a single DataFrame.
    
    Adds an 'asset' column to identify which asset each row belongs to.
    Sorts by date for proper time-based splitting.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing CSVs named {ASSET}_{direction}.csv
    direction : str
        Direction filter: 'up' or 'down'
    assets : List[str], optional
        Specific assets to load. If None, loads all matching files.
        
    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with 'asset' column, sorted by date.
    """
    # ┏━━━━━━━━━━ Convert data_dir to Path ━━━━━━━━━━┓
    data_dir = Path(data_dir)
    pattern = f"*_{direction}.csv"
    
    # ┏━━━━━━━━━━ Find CSV files ━━━━━━━━━━┓
    csv_files = sorted(data_dir.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No {pattern} files found in {data_dir}")
    
    # ┏━━━━━━━━━━ Load and combine data ━━━━━━━━━━┓
    dfs = []
    for csv_path in csv_files:
        # Extract asset name from filename (e.g., BTC_up.csv → BTC)
        asset = csv_path.stem.replace(f"_{direction}", "")
        
        # Skip if not in requested assets
        if assets is not None:
            matched_asset = next((a for a in assets if asset == a or asset.startswith(f"{a}USDT")), None)
            if not matched_asset:
                continue
            # Standardize the asset name back to e.g. "BTC"
            asset = matched_asset
        
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df['asset'] = asset
        dfs.append(df)
    
    # ┏━━━━━━━━━━ Check if any assets were loaded ━━━━━━━━━━┓
    if not dfs:
        raise ValueError(f"No assets loaded from {data_dir}")
    
    # ┏━━━━━━━━━━ Combine and sort ━━━━━━━━━━┓
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('date').reset_index(drop=True)
    
    print(f"[load_multi_asset_dataset] Loaded {len(dfs)} assets from {data_dir.name}, {len(combined):,} total rows.")
    return combined


def compute_dynamic_labels(df: pd.DataFrame, 
                           target_col: str = 'close', 
                           horizon: int = 1, 
                           lookback: int = 24, 
                           percentile: float = 33) -> np.ndarray:
    """
    Compute tri-class labels based on rolling dynamic thresholds.
    0: UP (return > rolling_pos_quantile)
    1: FLAT (between thresholds)
    2: DN (return < -rolling_neg_quantile)
    """
    prices = df[target_col]
    
    # ┏━━━━━━━━━━ 1. Compute Forward Returns (Target) ━━━━━━━━━━┓
    future_returns = (prices.shift(-horizon) / prices) - 1.0
    
    # ┏━━━━━━━━━━ 2. Compute Past Returns for Thresholds (Rolling Distribution) ━━━━━━━━━━┓
    # We use realized returns up to time t to determine the threshold for time t.
    past_returns = prices.pct_change(periods=1) # 1-step returns for volatility measurement
    
    # ┏━━━━━━━━━━ Separate Positive and Negative returns ━━━━━━━━━━┓
    pos_returns = past_returns.copy()
    neg_returns = past_returns.copy()
    pos_returns[pos_returns <= 0] = np.nan
    neg_returns[neg_returns >= 0] = np.nan
    neg_returns = neg_returns.abs()
    
    # ┏━━━━━━━━━━ 3. Rolling Quantiles (min_periods=lookback//2 to handle early data) ━━━━━━━━━━┓
    rho_up   = pos_returns.rolling(window=lookback, min_periods=lookback//2).quantile(percentile / 100.0)
    rho_down = neg_returns.rolling(window=lookback, min_periods=lookback//2).quantile(percentile / 100.0)
    
    # ┏━━━━━━━━━━ Fill early NaNs with first valid or fallback ━━━━━━━━━━┓
    rho_up   = rho_up.bfill().fillna(0.005)   # Fallback 0.5%
    rho_down = rho_down.bfill().fillna(0.005) # Fallback 0.5%
    
    # ┏━━━━━━━━━━ 4. Assign Labels ━━━━━━━━━━┓
    labels = np.ones(len(df), dtype=np.float32) # Default FLAT (1.0)
    
    # ┏━━━━━━━━━━ UP: Future Return > Dynamic UP Threshold (and positive) ━━━━━━━━━━┓
    labels[(future_returns > rho_up) & (future_returns > 0)] = 0.0
    
    # ┏━━━━━━━━━━ DN: Future Return < -Dynamic DOWN Threshold (and negative) ━━━━━━━━━━┓
    labels[(future_returns < -rho_down) & (future_returns < 0)] = 2.0
    
    # ┏━━━━━━━━━━ Handle NaN comparisons (where future_returns is NaN at end of series) ━━━━━━━━━━┓
    labels[np.isnan(future_returns)] = np.nan
    
    return labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENGINEERED WINDOW-LEVEL FEATURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ┏━━━━━━━━━━ Feature group → ordered names (must match compute_window_features output) ━━━━━━━━━━┓
ENG_FEATURE_GROUPS = {
    'window_stats': ['ret_mean', 'ret_std', 'ret_skew', 'ret_kurt', 'trend_slope', 'trend_r2', 'reversal_count', 'max_drawdown'],
    'volatility':   ['atr_last', 'atr_norm_last', 'bb_pctb_last', 'bb_bw_last'],
    'volume':       ['vol_spike_ratio', 'vol_trend_slope', 'vol_cv'],
    'momentum':     ['rsi_last', 'macd_last', 'roc_5_last', 'roc_20_last', 'momentum_align'],
    'regime':       ['adx_last', 'hurst', 'choppiness_idx'],
    # Crypto external + 30-day volatility features (last value per window)
    'xfeatures':    ['dvol_last', 'fear_greed_last', 'news_sentiment_last',
                     'r_vol_30_last', 'r_vol_30_ann_last', 'atr_w30_last', 
                     'rvi_30_last', 'massi_30_last', 'log_return_last', 'dcr_last'],
}

# ┏━━━━━━━━━━ Flat ordered list of all feature names ━━━━━━━━━━┓
ENG_FEATURE_NAMES = []
for _g in ['window_stats', 'volatility', 'volume', 'momentum', 'regime', 'xfeatures']:
    ENG_FEATURE_NAMES.extend(ENG_FEATURE_GROUPS[_g])

# ┏━━━━━━━━━━ List of Features depending on Cache Dataset ━━━━━━━━━━┓
def resolve_feature_names(n_cols: int) -> list:
    """Return ENG_FEATURE_NAMES sliced to match the actual tensor width.

    Old caches may have fewer columns (e.g. 23) than the current full list (33).
    This avoids a shape mismatch when building DataFrames from cached tensors.
    """
    if n_cols == len(ENG_FEATURE_NAMES):
        return list(ENG_FEATURE_NAMES)
    if n_cols < len(ENG_FEATURE_NAMES):
        return list(ENG_FEATURE_NAMES[:n_cols])
    # More columns than expected — pad with generic names
    return list(ENG_FEATURE_NAMES) + [f"feat_{i}" for i in range(len(ENG_FEATURE_NAMES), n_cols)]

# ┏━━━━━━━━━━ Known indicator column names used by compute_window_features ━━━━━━━━━━┓
# Base 9 technical indicators + Crypto extras.  Auto-discovered from CSV columns.
from Data_MLA.indicators import BASE_INDICATOR_COLUMNS, CRYPTO_XFEATURE_COLUMNS
INDICATOR_COLUMNS = BASE_INDICATOR_COLUMNS + CRYPTO_XFEATURE_COLUMNS


def compute_window_features(ohlcv_window: np.ndarray,
                            extrinsic_window: np.ndarray,
                            extrinsic_cols: list) -> np.ndarray:
    """
    Compute engineered scalar features for a single window.

    The output vector has 23 base features + 26 CTTS-style xfeatures = 49 total.

    Parameters
    ----------
    ohlcv_window : np.ndarray, shape (seq_len, 5)
        Columns: open, high, low, close, volume.
    extrinsic_window : np.ndarray, shape (seq_len, F)
        Row-level indicator columns (RSI, MACD, BB, ATR, ADX, RoC,
        plus CTTS crypto xfeature columns when available).
    extrinsic_cols : list of str
        Column names corresponding to extrinsic_window columns.

    Returns
    -------
    np.ndarray, shape (len(ENG_FEATURE_NAMES),)
        Ordered as ENG_FEATURE_NAMES.
    """
    T = ohlcv_window.shape[0]
    close = ohlcv_window[:, 3]
    high = ohlcv_window[:, 1]
    low = ohlcv_window[:, 2]
    volume = ohlcv_window[:, 4]

    # ┏━━━━━━━━━━ Helper to safely get last value of an extrinsic column ━━━━━━━━━━┓
    def _ext_last(col_name: str) -> float:
        if col_name in extrinsic_cols:
            idx = extrinsic_cols.index(col_name)
            return float(extrinsic_window[-1, idx])
        return 0.0

    # ┏━━━━━━━━━━ Group 1: Window Statistics [Mean of Returns, Std of Returns, Skewness of Returns, Kurtosis of Returns, Trend Slope, Trend R2, Reversal Count, Max Drawdown] ━━━━━━━━━━┓
    rets = np.diff(close) / (close[:-1] + 1e-10)  # (T-1,)
    ret_mean = rets.mean()
    ret_std = rets.std() if len(rets) > 1 else 0.0
    
    # ┏━━━━━━━━━━ 1.1 Skewness & Kurtosis ━━━━━━━━━━┓
    if ret_std > 1e-10:
        ret_skew = float(((rets - ret_mean) ** 3).mean() / (ret_std ** 3))
        ret_kurt = float(((rets - ret_mean) ** 4).mean() / (ret_std ** 4) - 3.0)
    else:
        ret_skew = 0.0
        ret_kurt = 0.0
    
    # ┏━━━━━━━━━━ 1.2 Trend slope + R2 (linear regression of close) ━━━━━━━━━━┓
    x = np.arange(T, dtype=np.float64)
    x_mean = x.mean()
    y_mean = close.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    ss_xy = ((x - x_mean) * (close - y_mean)).sum()
    slope = ss_xy / (ss_xx + 1e-10)
    y_hat = slope * x + (y_mean - slope * x_mean)
    ss_res = ((close - y_hat) ** 2).sum()
    ss_tot = ((close - y_mean) ** 2).sum()
    trend_slope = slope / (close[-1] + 1e-10)  # normalized
    trend_r2 = float(1.0 - ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0
    
    # ┏━━━━━━━━━━ 1.3 Reversal count (normalized) ━━━━━━━━━━┓
    signs = np.sign(rets)
    reversal_count = float(np.sum(signs[1:] != signs[:-1])) / max(T - 2, 1)
    
    # ┏━━━━━━━━━━ 1.4 Max drawdown ━━━━━━━━━━┓
    cum_max = np.maximum.accumulate(close)
    drawdowns = (cum_max - close) / (cum_max + 1e-10)
    max_drawdown = float(drawdowns.max())

    # ┏━━━━━━━━━━ Group 2: Volatility [ATR, Normalized ATR, Bollinger %B, Bollinger Bandwidth] ━━━━━━━━━━┓
    # ┏━━━━━━━━━━ 2.1 ATR ━━━━━━━━━━┓
    atr_last = _ext_last('atr_14')
    
    # ┏━━━━━━━━━━ 2.2 Normalized ATR ━━━━━━━━━━┓
    atr_norm_last = _ext_last('atr_norm')
    
    # ┏━━━━━━━━━━ 2.3 Bollinger %B ━━━━━━━━━━┓
    bb_pctb_last = _ext_last('bollinger_pct_b')
    
    # ┏━━━━━━━━━━ 2.4 Bollinger Bandwidth ━━━━━━━━━━┓
    bb_bw_last = _ext_last('bollinger_bandwidth')

    # ┏━━━━━━━━━━ Group 3: Volume [Mean, Spike Ratio, Trend Slope, Coefficient of Variation] ━━━━━━━━━━┓
    # ┏━━━━━━━━━━ 3.1 Mean of Volume ━━━━━━━━━━┓
    vol_mean = volume.mean()
    
    # ┏━━━━━━━━━━ 3.2 Spike Ratio ━━━━━━━━━━┓
    vol_spike_ratio = float(volume[-1] / (vol_mean + 1e-10))
    
    # ┏━━━━━━━━━━ 3.3 Volume Trend Slope ━━━━━━━━━━┓
    v_ss_xy = ((x - x_mean) * (volume - volume.mean())).sum()
    vol_trend_slope = float(v_ss_xy / (ss_xx + 1e-10)) / (vol_mean + 1e-10)
    
    # ┏━━━━━━━━━━ 3.4 Coefficient of Variation ━━━━━━━━━━┓
    vol_cv = float(volume.std() / (vol_mean + 1e-10))

    # ┏━━━━━━━━━━ Group 4: Momentum [RSI, MACD, RoC, Momentum Alignment] ━━━━━━━━━━┓
    # ┏━━━━━━━━━━ 4.1 RSI ━━━━━━━━━━┓
    rsi_last = _ext_last('rsi_14')
    
    # ┏━━━━━━━━━━ 4.2 MACD ━━━━━━━━━━┓
    macd_last = _ext_last('macd_histogram')
    
    # ┏━━━━━━━━━━ 4.3 RoC ━━━━━━━━━━┓
    roc_5_last = _ext_last('roc_5')
    roc_20_last = _ext_last('roc_20')
    
    # ┏━━━━━━━━━━ 4.4 Momentum Alignment ━━━━━━━━━━┓
    momentum_align = 1.0 if np.sign(roc_5_last) == np.sign(roc_20_last) else 0.0

    # ┏━━━━━━━━━━ Group 5: Regime [ADX, Hurst Exponent, Choppiness Index] ━━━━━━━━━━┓
    # ┏━━━━━━━━━━ 5.1 ADX ━━━━━━━━━━┓
    adx_last = _ext_last('adx_14')
    
    # ┏━━━━━━━━━━ 5.2 Hurst Exponent ━━━━━━━━━━┓
    if ret_std > 1e-10 and T > 2:
        cum_dev = np.cumsum(rets - ret_mean)
        R = cum_dev.max() - cum_dev.min()
        S = rets.std(ddof=0)
        hurst = float(np.log(R / (S + 1e-10) + 1e-10) / np.log(len(rets)))
    else:
        hurst = 0.5
    
    # ┏━━━━━━━━━━ 5.3 Choppiness Index ━━━━━━━━━━┓
    prev_close_arr = np.concatenate([[close[0]], close[:-1]])
    tr_arr = np.maximum(high - low, np.maximum(np.abs(high - prev_close_arr), np.abs(low - prev_close_arr)))
    atr_sum = tr_arr.sum()
    price_range = high.max() - low.min()
    if price_range > 1e-10 and T > 1:
        choppiness_idx = float(100.0 * np.log10(atr_sum / price_range + 1e-10) / np.log10(T))
    else:
        choppiness_idx = 50.0

    # ┏━━━━━━━━━━ Group 6: Crypto external + 30-day volatility features ━━━━━━━━━━┓
    dvol_last           = _ext_last('dvol')
    fear_greed_last     = _ext_last('fear_greed_idx')
    news_sentiment_last = _ext_last('news_sentiment')
    r_vol_30_last       = _ext_last('r_vol_30')
    r_vol_30_ann_last   = _ext_last('r_vol_30_ann')
    atr_w30_last        = _ext_last('atr_w30')
    rvi_30_last         = _ext_last('rvi_30')
    massi_30_last       = _ext_last('massi_30')
    log_return_last     = _ext_last('log_return')
    dcr_last            = _ext_last('dcr')

    return np.array([ret_mean, ret_std, ret_skew, ret_kurt,
                     trend_slope, trend_r2, reversal_count, max_drawdown,
                     atr_last, atr_norm_last, bb_pctb_last, bb_bw_last,
                     vol_spike_ratio, vol_trend_slope, vol_cv,
                     rsi_last, macd_last, roc_5_last, roc_20_last, momentum_align,
                     adx_last, hurst, choppiness_idx,
                     # XFeatures Group
                     dvol_last, fear_greed_last, news_sentiment_last,
                     r_vol_30_last, r_vol_30_ann_last,
                     atr_w30_last, rvi_30_last, massi_30_last,
                     log_return_last, dcr_last],
                     dtype=np.float32)


def prepare_multi_asset_dataset(df: pd.DataFrame,
                                seq_len: int = 96,
                                column_features: Sequence[str] = ("open", "high", "low", "close", "volume"),
                                extrinsic_features: Sequence[str] = (),
                                target_col: str = "meta_label",
                                task_type: str = "classification",
                                forecast_horizon: int = 1,
                                cfg: Dict[str, Any] = {}) -> Dict[str, torch.Tensor]:
    """
    Prepare multi-asset dataset for training.

    Creates sliding windows PER ASSET to avoid cross-asset leakage,
    then combines all windows with global time ordering.

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame from load_multi_asset_dataset with 'asset' column.
    seq_len : int
        Sequence length for sliding windows.
    column_features : Sequence[str]
        OHLCV columns for the main branch.
    extrinsic_features : Sequence[str]
        Legacy parameter (ignored). Indicator columns are auto-discovered via INDICATOR_COLUMNS.
    target_col : str
        Target column name (e.g., 'meta_label').

    Returns
    -------
    Dict with tensors: ohlcv, labels, time_ids, asset_ids, dates, eng_features
    """
    # ┏━━━━━━━━━━ Validate input ━━━━━━━━━━┓
    assets = df['asset'].unique()
    C = len(column_features)

    # ┏━━━━━━━━━━ Initialize lists ━━━━━━━━━━┓
    all_ohlcv           = []
    all_labels          = []
    all_returns         = []
    all_m1_pred_returns = []
    all_m1_pred_labels  = []
    all_m1_true_labels  = []
    all_dates           = []
    all_asset_ids       = []
    all_eng_features    = []
    all_time_ids        = {'minute': [], 'hour': [], 'dow': [], 'dom': [], 'month': []}

    # ┏━━━━━━━━━━ Auto-discover indicator columns for engineered features ━━━━━━━━━━┓
    _ext_cols_list = [c for c in INDICATOR_COLUMNS if c in df.columns]

    # ┏━━━━━━━━━━ Skip Counters (for diagnostic logging) ━━━━━━━━━━┓
    total_possible_windows = 0
    skip_nan_ohlcv         = 0
    skip_nan_extrinsic     = 0
    skip_short_asset       = 0
    skip_nan_target        = 0  # only for ground_truth / forecasting
    skip_boundary          = 0  # end >= n or forecasting horizon overflow

    # ┏━━━━━━━━━━ Map assets to IDs ━━━━━━━━━━┓
    asset_to_id = {a: i for i, a in enumerate(sorted(assets))}

    print(f"[prepare_multi_asset_dataset] {len(assets)} Assets, seq_len={seq_len}.")
    
    # ┏━━━━━━━━━━ Enrich with crypto xfeatures if not already present ━━━━━━━━━━┓
    _xfeat_needed = [c for c in CRYPTO_XFEATURE_COLUMNS if c not in df.columns]
    _do_xfeat_enrich = len(_xfeat_needed) > 0

    # ┏━━━━━━━━━━ Pre-load BTC daily close for vol-beta DVOL proxy ━━━━━━━━━━┓
    _btc_close_daily = None
    if _do_xfeat_enrich:
        _btc_assets = [a for a in assets if "BTC" in a.upper()]
        if _btc_assets:
            _btc_df = df[df['asset'] == _btc_assets[0]].copy()
            _btc_df = _btc_df.sort_values('date').reset_index(drop=True)
            if 'close' in _btc_df.columns and 'date' in _btc_df.columns:
                _btc_close_daily = _btc_df.set_index('date')['close'].sort_index()
                _btc_close_daily.index = pd.to_datetime(_btc_close_daily.index)

    # ┏━━━━━━━━━━ Process each asset ━━━━━━━━━━┓
    for asset in assets:
        asset_df = df[df['asset'] == asset].copy()
        asset_df = asset_df.sort_values('date').reset_index(drop=True)

        # ┏━━━━━━━━━━ Add crypto xfeatures per-asset if missing ━━━━━━━━━━┓
        if _do_xfeat_enrich:
            try:
                from Data_MLA.indicators import add_crypto_xfeatures
                asset_df = add_crypto_xfeatures(
                    asset_df,
                    asset=asset,
                    btc_close_daily=_btc_close_daily,
                    lag_days=1,
                )
            except Exception as exc:
                import warnings
                warnings.warn(f"[prepare] xfeature enrichment failed for {asset}: {exc}")
                for c in CRYPTO_XFEATURE_COLUMNS:
                    if c not in asset_df.columns:
                        asset_df[c] = np.nan

        # ┏━━━━━━━━━━ Filter short assets ━━━━━━━━━━┓
        n = len(asset_df)
        if n < seq_len:
            skip_short_asset += 1
            print(f"  [SKIP] {asset}: only {n} rows, need {seq_len}")
            continue

        # ┏━━━━━━━━━━ Create windows ━━━━━━━━━━┓
        N = n - seq_len + 1  # Number of windows for this asset
        total_possible_windows += N
        
        # ┏━━━━━━━━━━ Extract raw values ━━━━━━━━━━┓
        ohlcv_vals = asset_df[list(column_features)].values.astype(np.float32)
        
        # ┏━━━━━━━━━━ Auto-discover indicator columns for this asset ━━━━━━━━━━┓
        # Base indicators: strict NaN check (skip window if incomplete)
        # Xfeature columns: forward-fill + zero-fill (never drop windows)
        base_ext_cols = [c for c in BASE_INDICATOR_COLUMNS if c in asset_df.columns]
        xfeat_ext_cols = [c for c in CRYPTO_XFEATURE_COLUMNS if c in asset_df.columns]
        ext_cols = base_ext_cols + xfeat_ext_cols
        n_base_ext = len(base_ext_cols)

        # ┏━━━━━━━━━━ Xfeature columns Forward-Filling ━━━━━━━━━━┓
        if xfeat_ext_cols:
            asset_df[xfeat_ext_cols] = asset_df[xfeat_ext_cols].ffill().fillna(0.0)
        if ext_cols:
            extrinsic_vals = asset_df[ext_cols].values.astype(np.float32)
        else:
            extrinsic_vals = np.zeros((n, 0), dtype=np.float32)
        
        # ┏━━━━━━━━━━ Target values ━━━━━━━━━━┓
        if target_col in asset_df.columns:
            # Map 'False' (string) and False (bool) to np.nan so the windowing logic skips it
            t_col = asset_df[target_col].replace({'False': np.nan, False: np.nan})
            target_vals = t_col.values.astype(np.float32)
        else:
            target_vals = np.zeros(n, dtype=np.float32)
            
        # ┏━━━━━━━━━━ Pre-computed Returns ━━━━━━━━━━┓
        # If the CSV already has a return column (common in meta-labeling), use it.
        # Otherwise, we will re-compute it from prices.
        return_col = next((c for c in ["returns", "return", "ret"] if c in asset_df.columns), None)
        if return_col:
            return_series = asset_df[return_col].fillna(0.0).values.astype(np.float32)
        else:
            return_series = None

        # ┏━━━━━━━━━━ M1 Predictions (Historical) ━━━━━━━━━━┓
        if 'prediction' in asset_df.columns and 'close' in asset_df.columns:
            m1_pred_price = asset_df['prediction'].values.astype(np.float32)
            close_price = asset_df['close'].values.astype(np.float32)
            # M1 Predicted Return: (Prediction - Close) / Close
            m1_pred_return_series = (m1_pred_price - close_price) / (close_price + 1e-9)
        else:
            m1_pred_return_series = None
            
        if 'pred' in asset_df.columns:
            m1_pred_label_series = asset_df['pred'].values.astype(np.float32)
        else:
            m1_pred_label_series = None
            
        if 'lab' in asset_df.columns:
            m1_true_label_series = asset_df['lab'].values.astype(np.float32)
        else:
            m1_true_label_series = None
        
        # ┏━━━━━━━━━━ Time features ━━━━━━━━━━┓
        timestamps = pd.to_datetime(asset_df['date'])
        time_features = {'minute': timestamps.dt.minute.values.astype(np.int64),
                         'hour':    timestamps.dt.hour.values.astype(np.int64),
                         'dow':     timestamps.dt.dayofweek.values.astype(np.int64),
                         'dom':     (timestamps.dt.day - 1).values.astype(np.int64),
                         'month':   (timestamps.dt.month - 1).values.astype(np.int64)}
        
        asset_id = asset_to_id[asset]
        
        # ┏━━━━━━━━━━ Dynamic Labels (Pre-compute entire series) ━━━━━━━━━━┓
        dynamic_labels = None
        dt_cfg = cfg.get('data', {}).get('load', {}).get('dynamic_threshold', {})
        if target_col == 'ground_truth' and dt_cfg.get('enabled', False):
            print(f"  [Dynamic Labeling] Computing rolling thresholds for {asset}...")
            dynamic_labels = compute_dynamic_labels(asset_df, 
                                                    target_col='close', 
                                                    horizon=forecast_horizon, 
                                                    lookback=dt_cfg.get('lookback', 24), 
                                                    percentile=dt_cfg.get('percentile', 33))
        
        # ┏━━━━━━━━━━ Build sliding windows ━━━━━━━━━━┓
        for i in range(N):
            start, end = i, i + seq_len
            
            # ┏━━━━━━━━━━ Check for NaNs in base indicator columns only (skip window if incomplete) ━━━━━━━━━━┓
            if n_base_ext > 0:
                window_base_ext = extrinsic_vals[start:end, :n_base_ext]
                if np.isnan(window_base_ext).any():
                    skip_nan_extrinsic += 1
                    continue  # Skip incomplete window

            # ┏━━━━━━━━━━ Check for NaNs in target ━━━━━━━━━━┓
            if target_col == 'ground_truth':
                # Special 3-class logic
                if end >= n:
                    skip_boundary += 1
                    continue
                
                if dynamic_labels is not None:
                    # Use pre-computed dynamic label for this window end
                    # The label at 'end' corresponds to the return from 'end' to 'end+h'
                    target_val = dynamic_labels[end]
                else:
                    # Fallback to static 1% logic
                    try:
                        close_idx = list(column_features).index('close')
                    except ValueError:
                        close_idx = 3 if len(column_features) > 3 else 0
                    
                    last_close = ohlcv_vals[end - 1, close_idx]
                    next_close = ohlcv_vals[end,     close_idx]
                    
                    change = (next_close - last_close) / (last_close + 1e-9)
                    
                    if change > 0.01:
                        target_val = 0.0 # UP
                    elif change < -0.01:
                        target_val = 2.0 # DN
                    else:
                        target_val = 1.0 # FLAT
            elif task_type == "forecasting":
                if end + forecast_horizon > n:
                    skip_boundary += 1
                    continue
                target_val = target_vals[end : end + forecast_horizon]
                if np.isnan(target_val).any():
                     skip_nan_target += 1
                     continue # Skip incomplete window
            else:
                target_val = target_vals[end - 1]

            # ┏━━━━━━━━━━ OHLCV window (raw, no normalization — handled by IN-Flow) ━━━━━━━━━━┓
            window = ohlcv_vals[start:end, :]  # (seq_len, C)
            if np.isnan(window).any():
                skip_nan_ohlcv += 1
                continue

            all_ohlcv.append(window.T)  # (C, seq_len)

            # ┏━━━━━━━━━━ Engineered window-level features ━━━━━━━━━━┓
            _ext_w = extrinsic_vals[start:end, :] if extrinsic_vals.shape[1] > 0 else np.zeros((seq_len, 0), dtype=np.float32)
            all_eng_features.append(compute_window_features(window, _ext_w, ext_cols))

            # ┏━━━━━━━━━━ Time features for the window ━━━━━━━━━━┓
            for key in all_time_ids:
                all_time_ids[key].append(time_features[key][start:end])

            # ┏━━━━━━━━━━ Label at window end ━━━━━━━━━━┓
            all_labels.append(target_val)
            
            # ┏━━━━━━━━━━ Actual Continuous Return ━━━━━━━━━━┓
            if return_series is not None:
                # Use the pre-computed return from the CSV
                ret = return_series[end - 1]
            elif 'ground_truth' in asset_df.columns:
                # ground_truth = close[t + horizon], pre-attached in the CSV
                gt_val = asset_df['ground_truth'].iloc[end - 1]
                try:
                    close_idx = list(column_features).index('close')
                except ValueError:
                    close_idx = 3 if len(column_features) > 3 else 0
                curr_close = ohlcv_vals[end - 1, close_idx]
                if np.isnan(gt_val) or curr_close == 0:
                    ret = np.nan
                else:
                    ret = (gt_val - curr_close) / curr_close
            else:
                # Fallback: re-compute from prices (loses boundary rows)
                try:
                    close_idx = list(column_features).index('close')
                except ValueError:
                    close_idx = 3 if len(column_features) > 3 else 0

                if end - 1 + forecast_horizon < len(ohlcv_vals):
                    curr_close = ohlcv_vals[end - 1, close_idx]
                    fut_close  = ohlcv_vals[end - 1 + forecast_horizon, close_idx]
                    ret = (fut_close - curr_close) / (curr_close + 1e-9)
                else:
                    ret = np.nan
            
            all_returns.append(ret)
            
            # ┏━━━━━━━━━━ M1 Prediction for the window ━━━━━━━━━━┓
            if m1_pred_return_series is not None:
                all_m1_pred_returns.append(m1_pred_return_series[end - 1])
            else:
                all_m1_pred_returns.append(np.nan)
                
            if m1_pred_label_series is not None:
                all_m1_pred_labels.append(m1_pred_label_series[end - 1])
            else:
                all_m1_pred_labels.append(np.nan)
                
            if m1_true_label_series is not None:
                all_m1_true_labels.append(m1_true_label_series[end - 1])
            else:
                all_m1_true_labels.append(np.nan)
            
            # ┏━━━━━━━━━━ Store date and asset for sorting ━━━━━━━━━━┓
            all_dates.append(timestamps.iloc[end - 1])
            all_asset_ids.append(asset_id)
    
    # ┏━━━━━━━━━━ Check if any windows were created ━━━━━━━━━━┓
    if not all_ohlcv:
        raise ValueError("No windows created from any asset")
    
    # ┏━━━━━━━━━━ Convert to tensors ━━━━━━━━━━┓
    result = {'ohlcv':           torch.tensor(np.stack(all_ohlcv), dtype=torch.float32),
              'eng_features':    torch.tensor(np.stack(all_eng_features), dtype=torch.float32),
              'labels':          torch.tensor(np.array(all_labels), dtype=torch.float32),
              'returns':         torch.tensor(np.array(all_returns), dtype=torch.float32),
              'm1_pred_returns': torch.tensor(np.array(all_m1_pred_returns), dtype=torch.float32),
              'm1_pred_labels':  torch.tensor(np.array(all_m1_pred_labels), dtype=torch.float32),
              'm1_true_labels':  torch.tensor(np.array(all_m1_true_labels), dtype=torch.float32),
              'asset_ids':       torch.tensor(all_asset_ids, dtype=torch.long),
              'dates':           all_dates,
              'time_ids':        {k: torch.tensor(np.stack(v), dtype=torch.long) for k, v in all_time_ids.items()},
              'asset_map':       {v: k for k, v in asset_to_id.items()}}  # id → name

    # ┏━━━━━━━━━━ Window Creation Summary ━━━━━━━━━━┓
    created = len(all_ohlcv)
    total_skipped = total_possible_windows - created
    nan_label_count = int(np.isnan(np.array(all_labels)).sum())
    print(f"[prepare_multi_asset_dataset] Window summary:")
    print(f"    Possible windows (rows - {len(assets)} assets * {seq_len - 1} rows):  {total_possible_windows:,}")
    print(f"    Created windows:                                {created:,}")
    if skip_short_asset:   print(f"    Skipped — asset too short:               {skip_short_asset} assets")
    if skip_nan_ohlcv:     print(f"    Skipped — NaN in OHLCV:                  {skip_nan_ohlcv:,}")
    if skip_nan_extrinsic: print(f"    Skipped — NaN in extrinsic features:     {skip_nan_extrinsic:,}")
    if skip_boundary:      print(f"    Skipped — boundary overflow:             {skip_boundary:,}")
    if skip_nan_target:    print(f"    Skipped — NaN in target:                 {skip_nan_target:,}")
    if nan_label_count:    print(f"    Windows with NaN label (filtered later in split_by_global_time): {nan_label_count:,} ({nan_label_count/created*100:.2f}%)")
    print(f"    Total Final Windows:                                             {total_possible_windows - nan_label_count:,} ({((total_possible_windows - nan_label_count)/total_possible_windows*100):.2f}%)\n")
    return result


def split_by_global_time(dataset,
                         train_frac: float = 0.60,
                         meta_frac:  float = 0.10,
                         val_frac:   float = 0.20,
                         return_raw: bool = False,
                         train_end:  str  = None,
                         val_end:    str  = None) -> Any:
    """
    Split dataset indices by global time (not per-asset).
    All assets share the same cutoff dates to avoid data leakage.

    Supports two modes:
      1. Fraction-based (original): uses train_frac / meta_frac / val_frac.
      2. Date-based (new): uses explicit train_end / val_end timestamps.
         Meta split is absorbed into training in this mode.

    Works with both plain dataset dicts AND MultiGranDataset instances.
    """
    print(f"\n┏━━━━━━━━━━ Splitting dataset by global time ━━━━━━━━━━┓")
    # ┏━━━━━━━━━━ Extract dates and labels (works for dict OR MultiGranDataset) ━━━━━━━━━━┓
    if isinstance(dataset, dict):
        dates  = dataset['dates']
        labels = dataset['labels']
    else:
        dates  = dataset.dates
        labels = dataset.labels

    # ┏━━━━━━━━━━ Filter valid (non-NaN) indices ━━━━━━━━━━┓
    n_total = len(dates)
    valid_indices = [i for i in range(n_total) if not torch.isnan(labels[i])]
    n_nan = n_total - len(valid_indices)
    sorted_valid  = sorted(valid_indices, key=lambda i: dates[i])
    print(f"[split_by_global_time] Total: {n_total:,} Windows → {len(valid_indices):,} with valid labels"
          f" ({n_nan:,} had NaN meta_label, filtered out)")

    # ┏━━━━━━━━━━ Mode 2: Explicit Date Boundaries ━━━━━━━━━━┓
    if train_end is not None and val_end is not None:
        t_train_end = pd.Timestamp(train_end)
        t_val_end   = pd.Timestamp(val_end)

        # ┏━━━━━━━━━━ Split indices into train, val, test ━━━━━━━━━━┓
        idx_train, idx_val, idx_test = [], [], []
        for i in sorted_valid:
            t = dates[i]
            if t <= t_train_end:
                idx_train.append(i)
            elif t <= t_val_end:
                idx_val.append(i)
            else:
                idx_test.append(i)
       
        # Meta absorbed into training
        idx_meta = []

        print(f"[split_by_global_time] Date-based split (meta absorbed):")
        print(f"    Train: ≤ {t_train_end}  ({len(idx_train):,} samples)")
        print(f"    Val:   ≤ {t_val_end}  ({len(idx_val):,} samples)")
        print(f"    Test:  > {t_val_end}  ({len(idx_test):,} samples)")

        # ┏━━━━━━━━━━ Return Non-Raw Indices ━━━━━━━━━━┓
        if not return_raw:
            return idx_train, idx_meta, idx_val, idx_test

        # ┏━━━━━━━━━━ Raw indices (including NaN labels) allocated by same boundaries ━━━━━━━━━━┓
        idx_train_raw, idx_meta_raw, idx_val_raw, idx_test_raw = [], [], [], []
        for i in sorted(range(len(dates)), key=lambda i: dates[i]):
            t = dates[i]
            if t <= t_train_end:
                idx_train_raw.append(i)
            elif t <= t_val_end:
                idx_val_raw.append(i)
            else:
                idx_test_raw.append(i)

        return ((idx_train, idx_train_raw), (idx_meta, idx_meta_raw),
                (idx_val, idx_val_raw), (idx_test, idx_test_raw))

    # ┏━━━━━━━━━━ Mode 1: Fraction-based (original logic) ━━━━━━━━━━┓
    n_valid = len(sorted_valid)
    n_train = int(train_frac * n_valid)
    n_meta  = int(meta_frac * n_valid)
    n_val   = int(val_frac * n_valid)

    idx_train = sorted_valid[:n_train]
    idx_meta  = sorted_valid[n_train:n_train + n_meta]
    idx_val   = sorted_valid[n_train + n_meta:n_train + n_meta + n_val]
    idx_test  = sorted_valid[n_train + n_meta + n_val:]

    # ┏━━━━━━━━━━ Return Non-Raw Indices ━━━━━━━━━━┓
    if not return_raw:
        return idx_train, idx_meta, idx_val, idx_test

    # ┏━━━━━━━━━━ Raw indices (including NaN labels) allocated by same boundaries ━━━━━━━━━━┓
    def get_max_date(indices):
        return max(dates[i] for i in indices) if indices else pd.Timestamp.min

    t_train_end = get_max_date(idx_train)
    t_meta_end  = get_max_date(idx_meta)  if idx_meta else t_train_end
    t_val_end   = get_max_date(idx_val)   if idx_val  else t_meta_end

    # ┏━━━━━━━━━━ Raw indices (including NaN labels) allocated by same boundaries ━━━━━━━━━━┓
    idx_train_raw, idx_meta_raw, idx_val_raw, idx_test_raw = [], [], [], []
    for i in sorted(range(len(dates)), key=lambda i: dates[i]):
        t = dates[i]
        if t <= t_train_end:
            idx_train_raw.append(i)
        elif t <= t_meta_end:
            idx_meta_raw.append(i)
        elif t <= t_val_end:
            idx_val_raw.append(i)
        else:
            idx_test_raw.append(i)

    return ((idx_train, idx_train_raw), (idx_meta, idx_meta_raw),
            (idx_val, idx_val_raw), (idx_test, idx_test_raw))


def load_dataset_from_config(config: dict) -> pd.DataFrame:
    """
    Load dataset based on config options.
    
    Supports:
    - asset: null → Load all assets
    - asset: "BTC" → Load single asset
    - asset: ["BTC", "ETH"] → Load specific subset
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with 'dataset' key
        
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with 'asset' column
    """
    # ┏━━━━━━━━━━ Get dataset config ━━━━━━━━━━┓
    dataset_cfg = config.get('data', {}).get('load', {})  
    split_cfg   = config.get('data', {}).get('split', {})   
    data_dir    = config['paths']['csv_dir']
    direction   = dataset_cfg.get('direction', 'up')
    asset_cfg   = dataset_cfg.get('symbol', None)
    
    # ┏━━━━━━━━━━ Dynamically adjust data_dir for granularity & meta_label_mode ━━━━━━━━━━┓
    granularity = dataset_cfg.get('granularity', '1h')
    meta_mode   = dataset_cfg.get('meta_label_mode', 'og')
    data_dir_path = Path(data_dir)

    def _resolve_mode_root(path: Path) -> Optional[Path]:
        """Resolve TP/FP/OG root if present; otherwise return None."""
        if path.name.lower() in ("tp", "fp", "og"):
            return path
        for candidate in (path / meta_mode.upper(), path / meta_mode.lower()):
            if candidate.exists():
                return candidate
        return None

    def _resolve_horizon_dir(path: Path) -> Optional[Path]:
        """Resolve horizon directory when using new TP/horizon_X layout."""
        if path.name.startswith("horizon_"):
            return path
        if path.parent.name.startswith("horizon_"):
            return path.parent
        mode_root = _resolve_mode_root(path)
        if mode_root is not None:
            horizon = config.get("data", {}).get("load", {}).get("forecast_horizon", None)
            if horizon is None:
                horizon = split_cfg.get("forecast_horizon", None)
            if horizon is not None:
                candidate = mode_root / f"horizon_{horizon}"
                if candidate.exists():
                    return candidate
        return None

    # ┏━━━━━━━━━━ Normalize asset config to list or None ━━━━━━━━━━┓
    if asset_cfg is None:
        assets = None  # Load all
    elif isinstance(asset_cfg, str):
        assets = [asset_cfg]  # Single asset
    elif isinstance(asset_cfg, list):
        assets = asset_cfg  # Specific list
    else:
        raise ValueError(f"Invalid asset config: {asset_cfg}")

    # ┏━━━━━━━━━━ Multi-granularity: load ALL matching dirs ━━━━━━━━━━┓
    if granularity == "all":
        # ┏━━━━━━━━━━ Get parent directory and suffix ━━━━━━━━━━┓
        horizon_dir = _resolve_horizon_dir(data_dir_path)
        parent_dir = horizon_dir if horizon_dir is not None else data_dir_path.parent  # e.g. .../Crypto/ or .../horizon_7
        suffix = f"_{meta_mode.lower()}"
        gran_dirs = sorted([d for d in parent_dir.iterdir()
                            if d.is_dir() and d.name.endswith(suffix)])
        if not gran_dirs:
            raise FileNotFoundError(f"No *{suffix} directories found in {parent_dir}")
        print("┏━━━━━━━━━━ Loading Datasets ━━━━━━━━━━┓")
        print(f"[load_dataset_from_config] Multi-granularity mode: Found {len(gran_dirs)} dirs.")
        
        # ┏━━━━━━━━━━ Load and combine data from each granularity ━━━━━━━━━━┓
        all_dfs = []
        for gdir in gran_dirs:
            gran_name = gdir.name.replace(suffix, "")  # e.g. "1d", "4h"
            if gran_name not in GRAN_SEQ_LEN:
                print(f"  [SKIP] {gdir.name} — unknown granularity '{gran_name}'")
                continue
            try:
                gdf = load_multi_asset_dataset(data_dir  = str(gdir), 
                                               direction = direction, 
                                               assets    = assets)
                gdf['granularity'] = gran_name
                all_dfs.append(gdf)
            except Exception as e:
                print(f"[load_multi_asset_dataset] [WARN] Failed to load {gdir.name}: {e}")

        if not all_dfs:
            raise ValueError(f"No data loaded from any granularity dir in {parent_dir}")
        
        # ┏━━━━━━━━━━ Concatenate and sort ━━━━━━━━━━┓
        df = pd.concat(all_dfs, ignore_index=True).sort_values('date').reset_index(drop=True)
        print(f"[load_dataset_from_config] Concatenated: {len(df):,} rows across "
              f"{df['granularity'].nunique()} granularities.")
        print("┏━━━━━━━━━━ Finished: Loading & Concatenating Datasets ━━━━━━━━━━┓")
    else:
        # ┏━━━━━━━━━━ Single (auto-switch) granularity (existing logic) ━━━━━━━━━━┓
        new_name = f"{granularity}_{meta_mode.lower()}"
        horizon_dir = _resolve_horizon_dir(data_dir_path)
        if horizon_dir is not None:
            data_dir = str(horizon_dir / new_name)
        else:
            data_dir = str(data_dir_path.with_name(new_name))
        print(f"[load_dataset_from_config] Auto-resolved data_dir to: {data_dir} "
              f"(granularity: {granularity}, mode: {meta_mode})")
        df = load_multi_asset_dataset(data_dir = data_dir, 
                                      direction = direction, 
                                      assets = assets)

    
    # ┏━━━━━━━━━━ Filter by date range ━━━━━━━━━━┓
    start_date = split_cfg.get('start_date', None)
    end_date   = split_cfg.get('end_date', None)
    print("\n┏━━━━━━━━━━ Filtering by Date Range: From start_date:", start_date,"to end_date:", end_date," ━━━━━━━━━━┓")
    if start_date is not None or end_date is not None:
        n_before = len(df)

        # ┏━━━━━━━━━━ Adjust start_date for context window ━━━━━━━━━━┓
        if start_date is not None:
            # For multi-gran, use the LARGEST context buffer (5m=480 @ 5min = 40h)
            try:
                if granularity == "all":
                    # ┏━━━━━━━━━━ Dynamic Context Buffer ━━━━━━━━━━┓
                    # We need to buffer enough history so the LARGEST window can be formed
                    # starting exactly at `start_date`.
                    # Ex: If we only train on 1h, 2h, 4h, the longest window might be 
                    # 4h * 90 tokens = 360 hours. We must load data from start_date - 360h.
                    # If we hardcoded freq_min=5 but didn't actually load 5m data, we'd 
                    # underestimate the required buffer (e.g. 480 * 5m = 40 hours) and 
                    # crash/drop windows because 40h < 360h.
                    
                    loaded_grans = df['granularity'].unique()
                    
                    # ┏━━━━━━━━━━ Helper: Convert granularity string to minutes ━━━━━━━━━━┓
                    def gran_to_minutes(g: str) -> int:
                        if g.endswith('m'): return int(g[:-1])
                        if g.endswith('h'): return int(g[:-1]) * 60
                        if g.endswith('d'): return int(g[:-1]) * 1440
                        return 60
                    
                    # ┏━━━━━━━━━━ Find the maximum absolute time span across all loaded granularities ━━━━━━━━━━┓
                    max_buffer_minutes = 0
                    max_ctx = 0
                    freq_min = 60  # default fallback
                    for g in loaded_grans:
                        g_min = gran_to_minutes(g)
                        g_seq = GRAN_SEQ_LEN.get(g, 96)
                        g_span = g_min * g_seq
                        if g_span > max_buffer_minutes:
                            max_buffer_minutes = g_span
                            max_ctx = g_seq
                            freq_min = g_min
                else:
                    # ┏━━━━━━━━━━ Single granularity: compute buffer from granularity string ━━━━━━━━━━┓
                    max_ctx = int(config.get('data', {}).get('split', {}).get('context_length', 96))
                    if granularity.endswith('m'):
                        freq_min = int(granularity[:-1])
                    elif granularity.endswith('h'):
                        freq_min = int(granularity[:-1]) * 60
                    elif granularity.endswith('d'):
                        freq_min = int(granularity[:-1]) * 1440
                    else:
                        freq_min = 60

                # ┏━━━━━━━━━━ Calculate buffer and adjust start date ━━━━━━━━━━┓
                buffer = pd.Timedelta(minutes = freq_min * max_ctx)
                user_start_ts = pd.Timestamp(start_date)
                effective_start_ts = user_start_ts - buffer
                print(f"[load_dataset_from_config] Context buffer: {max_ctx} steps @ {freq_min}min = {buffer}. Effective Data Load Start: {effective_start_ts}")
            except Exception as e:
                print(f"[WARN] Failed to auto-adjust start_date: {e}")
                effective_start_ts = pd.Timestamp(start_date)
            df = df[df['date'] >= effective_start_ts]

        if end_date is not None:
            df = df[df['date'] <= pd.Timestamp(end_date)]

        df = df.reset_index(drop=True)
        print(f"[load_dataset_from_config] Date filter: {n_before:,} → {len(df):,} rows")
        print("┏━━━━━━━━━━ Finished: Filtering by Date Range ━━━━━━━━━━┓")

    return df
