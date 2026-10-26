"""Regime-detection feature builders for Jump Models (Stage 0).

Several feature menus are provided, mostly derived from log-returns so they
transfer across assets and granularities:

  • "nystrup" — the 15 backward-looking return features from Shu dissertation
    section 2.3.1 / Algorithm 1, based on Nystrup et al. (2020a), with the
    original window lengths interpreted as days and scaled to bars.

  • "shu" — the OHLCV-derived subset of Exhibit 3 from Shu et al. (2024),
    adapted to cap long lookbacks for limited-context workflows: return momentum
    features (EWMA return, RSI, stochastic %K, MACD) and log downside deviation.
    Market beta, VIX, and yield features are omitted because this repository does
    not provide the required market/FRED series.

  • "fshu" — the parsimonious financial-return feature set from Shu section
    3.3.4 / Table 3.2, adapted to cap long lookbacks for limited-context
    workflows: EWM downside deviation and two EWM Sortino ratios.

  • "cortese" — the paper-selected crypto-market features that are
    available in this repository: cross-asset RSI, 7/14-day return EMAs, and
    BTC/ETH return-volume correlations. The Google Trends sentiment correlations
    selected by the paper are excluded because no close production proxy exists.

  • "non-momentum" — the 16 non raw-return / non momentum columns from the
    combined menu: return-change, volatility/risk, and return-volume
    correlation features.

For EWM-based menus, half-lives are specified in **days** and scaled to the bar
count of the target granularity (e.g. 4h → 6 bars/day → hl 20d = 120 bars),
preserving the time scale rather than the bar count.

The capped shu/fshu settings are practical adaptations for limited context; they
should not be read as the paper-recommended feature definitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Utils.regime.data_access import load_ohlcv

# ┏━━━━━━━━━━ Bars per calendar day, per granularity ━━━━━━━━━━┓
# Used to convert a half-life expressed in *days* into a half-life in *bars*.
BARS_PER_DAY = {"1d": 1, 
                "12h": 2, 
                "8h": 3, 
                "6h": 4, 
                "4h": 6,
                "2h": 12, 
                "1h": 24, 
                "30m": 48,
}

# Named feature-set menus selectable from config / CLI.
#   nystrup      — 15 backward-looking return features from Shu section 2.3.1,
#                  with 6/14-day windows scaled to the target granularity.
#   shu          — OHLCV-derived subset of Shu et al. (2024) Exhibit 3,
#                  capped for limited-context workflows: return momentum + log
#                  downside deviation; omits unavailable external data.
#   fshu         — Shu section 3.3.4 / Table 3.2 financial return/risk features,
#                  capped for limited-context workflows: EWM downside deviation
#                  and EWM Sortino ratios.
#   cortese      — paper-selected, available crypto-market drivers from
#                  Cortese-Kolm-Lindstrom 2023; excludes unavailable GT sentiment.
#   non-momentum — 16 non raw-return / non momentum columns from the combined
#                  menu: return-change, volatility/risk, and return-volume
#                  correlation features.
#   combined     — union of every other feature set, for the
#                  sparse JM to feature-select from a rich candidate pool.
FEATURE_SETS = ("nystrup", "shu", "fshu", "cortese", "non-momentum", "combined")

NYSTRUP_WINDOWS_DAYS = (6, 14)
SHU_WINDOWS_DAYS = (8, 21, 63)
# SHU_WINDOWS_DAYS = (8, 21, 30)
SHU_MACD_WINDOWS_DAYS = ((8, 21), (21, 63))
# SHU_MACD_WINDOWS_DAYS = ((8, 21), (21, 30))
FSHU_DD_HALFLIFE_DAYS = 10
FSHU_SORTINO_HALFLIVES_DAYS = (20, 60)
# FSHU_SORTINO_HALFLIVES_DAYS = (20, 30)

CORTESE_ASSETS = (("BTCUSDT", "btc"), ("ETHUSDT", "eth"),
                  ("LTCUSDT", "ltc"), ("BCHUSDT", "bch"))
CORTESE_CORR_ASSETS = (("BTCUSDT", "btc"), ("ETHUSDT", "eth"))
CORTESE_SOURCE_COLUMNS = (
    [f"rsi14d_{sym}" for _, sym in CORTESE_ASSETS]
    + [f"ema7d_ret_{sym}" for _, sym in CORTESE_ASSETS]
    + [f"ema14d_ret_{sym}" for _, sym in CORTESE_ASSETS]
    + [
        f"ewm_corr{days}d_log_amount_ret_{sym}"
        for _, sym in CORTESE_CORR_ASSETS
        for days in (7, 14)
    ]
)
CORTESE_COLUMNS = [f"cortese_{c}" for c in CORTESE_SOURCE_COLUMNS]

NON_MOMENTUM_COLUMNS = (
    "nystrup_abs_dx",
    "nystrup_prev_abs_dx",
    "nystrup_center_std_w6d",
    "nystrup_left_std_w6d",
    "nystrup_right_std_w6d",
    "nystrup_center_std_w14d",
    "nystrup_left_std_w14d",
    "nystrup_right_std_w14d",
    "shu_log_dd_w21d",
    "fshu_dd_hl10d",
    "fshu_sortino_hl20d",
    "fshu_sortino_hl30d",
    "cortese_ewm_corr7d_log_amount_ret_btc",
    "cortese_ewm_corr14d_log_amount_ret_btc",
    "cortese_ewm_corr7d_log_amount_ret_eth",
    "cortese_ewm_corr14d_log_amount_ret_eth",
)


def _bars_for_halflife_days(hl_days: float, granularity: str) -> float:
    """Convert a half-life in days to the equivalent half-life in bars."""
    bpd = BARS_PER_DAY.get(granularity)
    if bpd is None:
        raise ValueError(f"Unknown granularity '{granularity}'. Known: {list(BARS_PER_DAY)}")
    return float(hl_days) * bpd


def _ewma(series: pd.Series, halflife_bars: float) -> pd.Series:
    """Exponentially-weighted moving average with a bar-count half-life."""
    return series.ewm(halflife=halflife_bars, adjust=False).mean()


def _ewma_span(series: pd.Series, span_bars: int) -> pd.Series:
    """Exponentially-weighted moving average using a paper-style window span."""
    return series.ewm(span=max(int(span_bars), 1), adjust=False).mean()


def _ewm_corr(x: pd.Series, y: pd.Series, halflife_bars: float) -> pd.Series:
    """Exponentially weighted linear correlation between two aligned series."""
    return x.astype(float).ewm(halflife=halflife_bars, adjust=False).corr(y.astype(float))


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Log returns of a close-price series."""
    return np.log(close).diff()


def _target_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the datetime index that a regime feature frame should align to."""
    if "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce")
    else:
        idx = pd.to_datetime(df.index, errors="coerce")
    if idx.isna().any():
        raise ValueError("Regime feature input contains non-datetime index/date values.")
    return pd.DatetimeIndex(idx)


def _load_peer_ohlcva(asset: str, granularity: str,
                      raw_root: str | Path | None = None) -> pd.DataFrame:
    """Load a required peer OHLCVA frame (close + amount) for cortese features."""
    return load_ohlcv(asset, granularity, root=raw_root,
                      required_columns=("close", "amount"))


def _window_days_to_bars(days: int, granularity: str) -> int:
    """Convert a whole-day window length from a paper into bars."""
    return max(1, int(round(_bars_for_halflife_days(days, granularity))))


def _stochastic_k(series: pd.Series, window: int) -> pd.Series:
    """Stochastic oscillator %K on a positive price/index series."""
    lo = series.rolling(window, min_periods=window).min()
    hi = series.rolling(window, min_periods=window).max()
    return 100.0 * (series - lo) / (hi - lo + 1e-10)


def _prefix_feature_columns(feats: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Prefix returned feature names with their standalone feature-set source."""
    prefix = f"{feature_set}_"
    return feats.rename(columns={
        col: col if str(col).startswith(prefix) else f"{prefix}{col}"
        for col in feats.columns
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_nystrup_features(df: pd.DataFrame, granularity: str,
                           asset: str | None = None,
                           raw_root: str | Path | None = None) -> pd.DataFrame:
    """15-feature Nystrup/Shu set from backward-looking log returns.

    Implements Shu_princeton_0181D_15320.pdf, section 2.3.1 / Algorithm 1.
    The univariate input series is `x_t = log(close_t).diff()`. For window
    lengths w=6 and w=14 days, all means/stds are backward-looking and available
    at time t. The day windows are scaled to bar counts for intraday granularities.

    The paper standardizes these features before fitting; this project already
    standardizes inside RegimeJumpModel, so this builder returns raw features.

    ``asset``/``raw_root`` are accepted for a uniform builder signature but unused
    here — this set derives only from the fitted asset's own close series.
    """
    del asset, raw_root  # single-asset feature set; peer data not needed.
    close = df["close"].astype(float)
    x = compute_log_returns(close)
    dx_abs = (x - x.shift(1)).abs()

    cols: dict[str, pd.Series] = {
        "nystrup_x": x,
        "nystrup_abs_dx": dx_abs,
        "nystrup_prev_abs_dx": dx_abs.shift(1),
    }

    for days in NYSTRUP_WINDOWS_DAYS:
        w = _window_days_to_bars(days, granularity)
        half = w // 2
        rolling_full = x.rolling(w, min_periods=w)
        rolling_half = x.rolling(half, min_periods=half)

        cols[f"nystrup_center_mean_w{days}d"] = rolling_full.mean()
        cols[f"nystrup_center_std_w{days}d"] = rolling_full.std(ddof=0)
        cols[f"nystrup_left_mean_w{days}d"] = rolling_half.mean().shift(half)
        cols[f"nystrup_left_std_w{days}d"] = rolling_half.std(ddof=0).shift(half)
        cols[f"nystrup_right_mean_w{days}d"] = rolling_half.mean()
        cols[f"nystrup_right_std_w{days}d"] = rolling_half.std(ddof=0)

    feats = pd.DataFrame(cols, index=df.index)
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return _prefix_feature_columns(feats, "nystrup")


def build_shu_features(df: pd.DataFrame, granularity: str,
                       asset: str | None = None,
                       raw_root: str | Path | None = None) -> pd.DataFrame:
    """OHLCV-derived Shu et al. (2024) Exhibit 3 feature subset.

    The paper fits SJMs on factor active returns and includes market beta,
    market return, VIX, and yield-curve features. This repo only has per-asset
    crypto OHLCV in the regime builder, so this feature set intentionally keeps
    only features derivable from the target asset's close series. Long lookbacks
    are capped for limited-context workflows, so these are practical adaptations
    rather than the paper-recommended feature definitions:

    - EWMA log return over 8/21/30 trading-day windows.
    - RSI over 8/21/30 trading-day windows.
    - Stochastic %K over 8/21/30 trading-day windows.
    - MACD-style normalized EMA spreads for 8/21 and 21/30 days.
    - log downside deviation over 21 days.

    ``asset``/``raw_root`` are accepted for a uniform builder signature but unused
    here — this set derives only from the fitted asset's own close series.
    """
    del asset, raw_root  # single-asset feature set; peer data not needed.
    # Deferred import: Data_MLA.indicators pulls the broader indicator stack, so
    # we import compute_rsi only when a shu/cortese build actually needs it
    # rather than at package import time.
    from Data_MLA.indicators import compute_rsi

    close = df["close"].astype(float)
    log_ret = compute_log_returns(close)
    # A return index preserves the return path while giving price-style
    # indicators (RSI/%K/MACD) a positive level series.
    ret_index = np.exp(log_ret.fillna(0.0).cumsum())

    cols: dict[str, pd.Series] = {}
    for days in SHU_WINDOWS_DAYS:
        window = _window_days_to_bars(days, granularity)
        cols[f"shu_ewma_ret_w{days}d"] = _ewma_span(log_ret, window)
        cols[f"shu_rsi_w{days}d"] = compute_rsi(ret_index, period=window)
        cols[f"shu_stoch_k_w{days}d"] = _stochastic_k(ret_index, window)

    for short_days, long_days in SHU_MACD_WINDOWS_DAYS:
        short_window = _window_days_to_bars(short_days, granularity)
        long_window = _window_days_to_bars(long_days, granularity)
        ema_short = _ewma_span(ret_index, short_window)
        ema_long = _ewma_span(ret_index, long_window)
        cols[f"shu_macd_{short_days}d_{long_days}d"] = 100.0 * (ema_short - ema_long) / (ema_long + 1e-10)

    dd_window = _window_days_to_bars(21, granularity)
    downside = log_ret.clip(upper=0.0)
    dd = np.sqrt((downside ** 2).ewm(span=dd_window, adjust=False).mean())
    cols["shu_log_dd_w21d"] = np.log(dd.replace(0.0, np.nan))

    feats = pd.DataFrame(cols, index=df.index)
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return _prefix_feature_columns(feats, "shu")


def build_fshu_features(df: pd.DataFrame, granularity: str,
                        asset: str | None = None,
                        raw_root: str | Path | None = None) -> pd.DataFrame:
    """Financial Shu feature set from Section 3.3.4 / Table 3.2.

    The paper derives these three features from an excess return series. The
    regime raw data available here is crypto OHLCV without a risk-free series, so
    this implementation uses target-asset log returns as excess returns with
    risk-free return fixed at zero. Long lookbacks are capped for limited-context
    workflows, so these are practical adaptations rather than the
    paper-recommended feature definitions:

    - EWM downside deviation with halflife 10 days.
    - EWM Sortino ratio with halflife 20 days.
    - EWM Sortino ratio with halflife 30 days.

    ``asset``/``raw_root`` are accepted for a uniform builder signature but unused
    here — this set derives only from the fitted asset's own close series.
    """
    del asset, raw_root  # single-asset feature set; peer data not needed.
    excess_ret = compute_log_returns(df["close"].astype(float))

    def _ewm_downside_deviation(ret: pd.Series, halflife_bars: float) -> pd.Series:
        downside_sq = ret.where(ret < 0.0, 0.0) ** 2
        return np.sqrt(_ewma(downside_sq, halflife_bars))

    dd_hl = _bars_for_halflife_days(FSHU_DD_HALFLIFE_DAYS, granularity)
    cols: dict[str, pd.Series] = {
        f"fshu_dd_hl{FSHU_DD_HALFLIFE_DAYS}d": _ewm_downside_deviation(
            excess_ret,
            dd_hl,
        )
    }

    for days in FSHU_SORTINO_HALFLIVES_DAYS:
        hl = _bars_for_halflife_days(days, granularity)
        avg_ret = _ewma(excess_ret, hl)
        dd = _ewm_downside_deviation(excess_ret, hl)
        cols[f"fshu_sortino_hl{days}d"] = avg_ret / dd.replace(0.0, np.nan)

    feats = pd.DataFrame(cols, index=df.index)
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return _prefix_feature_columns(feats, "fshu")


def build_cortese_features(df: pd.DataFrame, granularity: str,
                           asset: str | None = None,
                           raw_root: str | Path | None = None) -> pd.DataFrame:
    """Paper-selected available crypto-market drivers from Cortese et al. (2023).

    Implements the selected non-sentiment features reported in Fig. 3 / Table 5
    of s42521-023-00085-x.pdf. Google Trends correlations are deliberately not
    included: this repository only has generic news/Fear&Greed sentiment series,
    which are not close proxies for coin-specific Google Trends-return correlations.

    The feature set is market-wide/cross-asset. It always loads BTC, ETH, LTC and
    BCH raw OHLCVA data for the requested granularity and aligns the result to
    the input frame's index, so it can be used for any fitted asset.
    """
    del asset  # Cross-asset market features are independent of the fitted asset.
    # Deferred import: see build_shu_features — compute_rsi is imported lazily to
    # keep Data_MLA.indicators off the package import path.
    from Data_MLA.indicators import compute_rsi

    idx = _target_index(df)
    # _window_days_to_bars / _bars_for_halflife_days raise on an unknown
    # granularity, so no separate guard is needed here.
    rsi_period = _window_days_to_bars(14, granularity)
    hl7 = _bars_for_halflife_days(7, granularity)
    hl14 = _bars_for_halflife_days(14, granularity)

    cols: dict[str, pd.Series] = {}
    corr_assets = {peer for peer, _ in CORTESE_CORR_ASSETS}

    for peer, sym in CORTESE_ASSETS:
        panel = _load_peer_ohlcva(peer, granularity, raw_root=raw_root)
        close = panel["close"].astype(float).reindex(idx)
        ret = compute_log_returns(close)

        cols[f"rsi14d_{sym}"] = compute_rsi(close, period=rsi_period)
        cols[f"ema7d_ret_{sym}"] = _ewma(ret, hl7)
        cols[f"ema14d_ret_{sym}"] = _ewma(ret, hl14)

        if peer in corr_assets:
            amount = panel["amount"].astype(float).reindex(idx)
            log_amount_diff = np.log(amount.replace(0.0, np.nan)).diff()
            cols[f"ewm_corr7d_log_amount_ret_{sym}"] = _ewm_corr(log_amount_diff, ret, hl7)
            cols[f"ewm_corr14d_log_amount_ret_{sym}"] = _ewm_corr(log_amount_diff, ret, hl14)

    feats = pd.DataFrame(cols, index=idx)
    feats = feats.loc[:, CORTESE_SOURCE_COLUMNS]
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return _prefix_feature_columns(feats, "cortese")


def build_combined_features(df: pd.DataFrame, granularity: str,
                            asset: str | None = None,
                            raw_root: str | Path | None = None) -> pd.DataFrame:
    """Union of all standalone regime feature sets.

    Intended for the Sparse JM, which feature-selects from a rich candidate pool.
    Inner-joined on the common bar index so every row is fully populated.
    """
    parts = [
        build_nystrup_features(df, granularity, asset=asset, raw_root=raw_root),
        build_shu_features(df, granularity, asset=asset, raw_root=raw_root),
        build_fshu_features(df, granularity, asset=asset, raw_root=raw_root),
        build_cortese_features(df, granularity, asset=asset, raw_root=raw_root),
    ]
    return pd.concat(parts, axis=1, join="inner").dropna()


def build_non_momentum_features(df: pd.DataFrame, granularity: str,
                                asset: str | None = None,
                                raw_root: str | Path | None = None) -> pd.DataFrame:
    """Non raw-return / non momentum subset of the combined feature menu.

    The 16 columns are return-change, volatility/risk, and return-volume
    correlation features. Momentum-style return means, EWMA returns, RSI,
    stochastic, MACD, and cross-asset EMA-return columns are deliberately
    excluded.
    """
    combined = build_combined_features(
        df,
        granularity,
        asset=asset,
        raw_root=raw_root,
    )
    return combined.loc[:, list(NON_MOMENTUM_COLUMNS)]


def build_regime_features(df: pd.DataFrame, granularity: str,
                          feature_set: str = "nystrup",
                          asset: str | None = None,
                          raw_root: str | Path | None = None) -> pd.DataFrame:
    """Dispatch to the requested feature menu.

    Args:
        feature_set: one of FEATURE_SETS — 'nystrup' (15 backward-looking return features from Shu section
            2.3.1), 'shu' (OHLCV-derived subset of Shu et al. 2024 Exhibit 3),
            'fshu' (Shu section 3.3.4 financial return/risk features),
            'cortese' (paper-selected available crypto-market drivers; no
            sentiment), 'non-momentum' (16 non raw-return / non momentum
            combined-menu columns), or 'combined' (union of every other
            standalone feature set).
        asset: fitted asset symbol. Currently only passed through for
            feature-set builders that need fitted-asset context.
        raw_root: optional override for Data_MLA/Raw when cross-asset builders
            need peer OHLCVA data.
    """
    builders = {
        "nystrup": build_nystrup_features,
        "shu": build_shu_features,
        "fshu": build_fshu_features,
        "cortese": build_cortese_features,
        "non-momentum": build_non_momentum_features,
        "combined": build_combined_features,
    }
    builder = builders.get(feature_set)
    if builder is None:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Choices: {FEATURE_SETS}")
    return builder(
        df,
        granularity,
        asset=asset,
        raw_root=raw_root,
    )


__all__ = [
    "BARS_PER_DAY",
    "FEATURE_SETS",
    "FSHU_DD_HALFLIFE_DAYS",
    "FSHU_SORTINO_HALFLIVES_DAYS",
    "NON_MOMENTUM_COLUMNS",
    "compute_log_returns",
    "build_nystrup_features",
    "build_shu_features",
    "build_fshu_features",
    "build_cortese_features",
    "build_non_momentum_features",
    "build_combined_features",
    "build_regime_features",
]
