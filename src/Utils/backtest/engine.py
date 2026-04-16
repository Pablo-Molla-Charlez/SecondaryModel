"""Financial backtest helpers extracted from kronos_tree.py."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ┏━━━━━━━━━━ Utils ━━━━━━━━━━┓
from Utils.utils import m1_display_label as _m1_display_label
from Utils.utils import model_label as _model_label
from Utils.classifier import MODELS_NO_SCALING


__all__ = [
    "_load_raw_close_prices",
    "_annualization_factor",
    "_calc_sharpe",
    "_calc_drawdown",
    "_build_spread_equity",
    "_equity_horizon_returns",
    "run_feature_backtest",
    "run_combined_backtest",
]



# ┏━━━━━━━━━━ Load Raw Close Prices from CSVs ━━━━━━━━━━┓
def _load_raw_close_prices(config: dict, granularity: str, direction: str) -> pd.DataFrame:
    """Load raw close prices from CSVs for Buy & Hold calculation."""
    # ┏━━━━━━━━━━ Load Config Data ━━━━━━━━━━┓
    meta_mode = config['data']['load']['meta_label_mode']
    gran_dir = Path(config['paths']['csv_dir']) / f"{granularity}_{meta_mode}"
    if not gran_dir.exists():
        print(f"  [backtest] WARNING: {gran_dir} not found, skipping B&H curve")
        return pd.DataFrame()

    # ┏━━━━━━━━━━ Load CSVs ━━━━━━━━━━┓
    dfs = []
    for f in sorted(gran_dir.glob(f"*_{direction}.csv")):
        asset = f.stem.replace(f"_{direction}", "")
        df = pd.read_csv(f, usecols=["date", "close"])
        df["asset"] = asset
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    # ┏━━━━━━━━━━ Concatenate & Format ━━━━━━━━━━┓
    result = pd.concat(dfs, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    return result


# ┏━━━━━━━━━━ Annualization Factor ━━━━━━━━━━┓
def _annualization_factor(granularity: str) -> float:
    """Return sqrt(bars_per_year) for Sharpe ratio annualization."""
    if granularity.endswith("m"):
        bars = 365.25 * 24 * 60 / int(granularity[:-1])
    elif granularity.endswith("h"):
        bars = 365.25 * 24 / int(granularity[:-1])
    elif granularity.endswith("d"):
        bars = 365.25 / int(granularity[:-1])
    else:
        bars = 365.25
    return np.sqrt(bars)


# ┏━━━━━━━━━━ Calculate Sharpe Ratio ━━━━━━━━━━┓
def _calc_sharpe(returns: np.ndarray, ann: float) -> float:
    """Sharpe = mean / std * ann, where ann = sqrt(observations_per_year)."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * ann)


# ┏━━━━━━━━━━ Calculate Max Drawdown ━━━━━━━━━━┓
def _calc_drawdown(equity: np.ndarray) -> float:
    """Max drawdown from an equity curve."""
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    return float(np.min(dd))


# ┏━━━━━━━━━━ Build Spread Equity ━━━━━━━━━━┓
def _build_spread_equity(trades_df: pd.DataFrame, timeline: pd.DatetimeIndex, horizon: int):
    """Spread each trade return over horizon bars, then accumulate multiplicatively."""
    # ┏━━━━━━━━━━ Handle Empty Trades ━━━━━━━━━━┓
    if len(trades_df) == 0:
        return pd.Series(1.0, index=timeline), np.array([])

    # ┏━━━━━━━━━━ Build Bar Map ━━━━━━━━━━┓
    bar_map = {d: i for i, d in enumerate(timeline)}
    n_bars = len(timeline)
    bar_sums = np.zeros(n_bars)
    bar_counts = np.zeros(n_bars)
    per_bar_ret = trades_df["return"].values / horizon

    # ┏━━━━━━━━━━ Accumulate Returns ━━━━━━━━━━┓
    for row_idx in range(len(trades_df)):
        entry_date = trades_df["date"].iloc[row_idx]
        if entry_date not in bar_map:
            continue
        start = bar_map[entry_date]
        end = min(start + horizon, n_bars)
        r = per_bar_ret[row_idx]
        for b in range(start, end):
            bar_sums[b] += r
            bar_counts[b] += 1

    # ┏━━━━━━━━━━ Normalize & Build Equity ━━━━━━━━━━┓
    active_mask = bar_counts > 0
    avg_conc = bar_counts[active_mask].mean() if active_mask.any() else 1.0
    bar_normed = np.where(active_mask, bar_sums / avg_conc, 0.0)
    equity = pd.Series((1 + bar_normed).cumprod(), index=timeline)
    return equity, trades_df["return"].dropna().values


# ┏━━━━━━━━━━ Calculate Horizon Returns ━━━━━━━━━━┓
def _equity_horizon_returns(equity: pd.Series, horizon: int) -> np.ndarray:
    """Non-overlapping pct-change returns sampled every horizon bars."""
    vals = equity.values
    rets = []
    for i in range(horizon, len(vals), horizon):
        if vals[i - horizon] != 0:
            rets.append((vals[i] - vals[i - horizon]) / abs(vals[i - horizon]))
    return np.array(rets)


# ┏━━━━━━━━━━ Run M2 Backtest ━━━━━━━━━━┓
def run_feature_backtest(dataset,
                         split_indices,
                         artifacts,
                         cfg,
                         save_dir,
                         class_names,
                         meta_mode,
                         granularity,
                         direction,
                         model_name  = "rf",
                         desc        = "all features",
                         file_prefix = "10_backtest_all",
                         fee         = 0.0,
                         thres_mode  = "utility",
                         ocp_alpha   = 0.10):
    """Equity curve plot + per-asset table + ROI report for RF/XGB/AG selective model."""
    # ┏━━━━━━━━━━ Model Label & Split Indices ━━━━━━━━━━┓
    mlabel = _model_label(model_name)
    idx_test = split_indices[-1]

    # ┏━━━━━━━━━━ Load Model & Artifacts ━━━━━━━━━━┓
    model       = artifacts["model"]
    scaler      = artifacts["scaler"]
    col_indices = artifacts["col_indices"]
    val_op      = artifacts["val_op"]
    threshold   = val_op["threshold"]
    
    # ┏━━━━━━━━━━ Load Engineered Features ━━━━━━━━━━┓
    eng = dataset["eng_features"] if isinstance(dataset, dict) else dataset.eng_features
    if isinstance(eng, torch.Tensor):
        eng = eng.numpy()
    
    # ┏━━━━━━━━━━ Load Labels ━━━━━━━━━━┓
    labels_all = dataset["labels"] if isinstance(dataset, dict) else dataset.labels
    if isinstance(labels_all, torch.Tensor):
        labels_all = labels_all.numpy()
    
    # ┏━━━━━━━━━━ Load Returns ━━━━━━━━━━┓
    returns_all = dataset["returns"] if isinstance(dataset, dict) else dataset.returns
    if isinstance(returns_all, torch.Tensor):
        returns_all = returns_all.numpy()
    
    # ┏━━━━━━━━━━ Load Asset IDs ━━━━━━━━━━┓
    asset_ids_all = dataset["asset_ids"] if isinstance(dataset, dict) else dataset.asset_ids
    if isinstance(asset_ids_all, torch.Tensor):
        asset_ids_all = asset_ids_all.numpy()
    
    # ┏━━━━━━━━━━ Load Asset Map ━━━━━━━━━━┓
    asset_map = dataset.get("asset_map", {}) if isinstance(dataset, dict) else dataset.asset_map
    if not isinstance(asset_map, dict) and hasattr(dataset, "asset_map"):
        asset_map = dataset.asset_map

    # ┏━━━━━━━━━━ Prepare Test Data ━━━━━━━━━━┓
    X_test = eng[idx_test][:, col_indices].copy()
    y_test = labels_all[idx_test].astype(int)
    test_returns = returns_all[idx_test].copy()
    test_asset_ids = asset_ids_all[idx_test]
    test_dates_raw = [dataset["dates"][i] for i in idx_test] if isinstance(dataset, dict) else [dataset.dates[i] for i in idx_test]
    test_assets = [asset_map.get(int(aid), str(aid)) for aid in test_asset_ids]

    # ┏━━━━━━━━━━ Scale Test Features (skip for models that need raw data) ━━━━━━━━━━┓
    if model_name not in MODELS_NO_SCALING:
        X_test = scaler.transform(X_test)

    # ┏━━━━━━━━━━ Predict Probabilities ━━━━━━━━━━┓
    probs_raw = model.predict_proba(X_test)[:, 1]
    calibrator = artifacts.get("calibrator")
    probs = calibrator.predict(probs_raw) if calibrator is not None else probs_raw
    if thres_mode.startswith("OCP") and "_ocp_test_approved" in val_op:
        m2_approved = val_op["_ocp_test_approved"]
    else:
        m2_approved = probs >= threshold

    # ┏━━━━━━━━━━ Apply Direction ━━━━━━━━━━┓
    if direction.lower() == "down":
        test_returns = -test_returns

    # ┏━━━━━━━━━━ Apply Fees ━━━━━━━━━━┓
    net_returns = test_returns - fee

    # ┏━━━━━━━━━━ Build Trades DataFrame ━━━━━━━━━━┓
    df_trades = pd.DataFrame({"date": pd.to_datetime(test_dates_raw),
                              "asset": test_assets,
                              "return": net_returns,
                              "label": y_test,
                              "m2_approved": m2_approved,
                              "m2_prob": probs})
    
    # ┏━━━━━━━━━━ Handle NaN Returns ━━━━━━━━━━┓
    n_nan_rets = int(df_trades["return"].isna().sum())
    if n_nan_rets > 0:
        print(f"  [backtest] WARNING: {n_nan_rets} trades have NaN returns (dropped)")
        df_trades = df_trades.dropna(subset=["return"]).reset_index(drop=True)
        m2_approved = df_trades["m2_approved"].values
        probs = df_trades["m2_prob"].values
        net_returns = df_trades["return"].values
        y_test = df_trades["label"].values
    
    # ┏━━━━━━━━━━ Get Test Date Range ━━━━━━━━━━┓
    test_start = df_trades["date"].min()
    test_end   = df_trades["date"].max()

    # ┏━━━━━━━━━━ Get Horizon ━━━━━━━━━━┓
    horizon = int(cfg.get("data", {}).get("load", {}).get("forecast_horizon", 7))
    m2_df = df_trades[df_trades["m2_approved"]]

    # ┏━━━━━━━━━━ Check for Utility Threshold Approval ━━━━━━━━━━┓
    has_m2_util = False
    if thres_mode.startswith("OCP"):
        m2_util_approved = probs >= val_op["threshold"]
        m2_util_df = df_trades[m2_util_approved].copy()
        has_m2_util = len(m2_util_df) > 0

    # ┏━━━━━━━━━━ Load Raw Close Prices ━━━━━━━━━━┓
    raw_close = _load_raw_close_prices(cfg, granularity, direction=direction)
    has_bh = len(raw_close) > 0

    # ┏━━━━━━━━━━ Filter Raw Close Prices by Test Date Range ━━━━━━━━━━┓
    if has_bh:
        raw_close = raw_close[(raw_close["date"] >= test_start) & (raw_close["date"] <= test_end)]
        bh_pivot = raw_close.pivot_table(index="date", columns="asset", values="close")
        bh_first = bh_pivot.iloc[0]
        bh_equity = (bh_pivot / bh_first).mean(axis=1)
        asset_bh_rets = (bh_pivot.iloc[-1] / bh_first) - 1
        full_idx = bh_equity.index
    else:
        full_idx = pd.DatetimeIndex(sorted(df_trades["date"].unique()))

    # ┏━━━━━━━━━━ Build Equity Curves ━━━━━━━━━━┓
    m1_equity, _ = _build_spread_equity(df_trades, full_idx, horizon)
    m2_equity, _ = _build_spread_equity(m2_df, full_idx, horizon)
    if has_m2_util:
        m2_util_equity, _ = _build_spread_equity(m2_util_df, full_idx, horizon)

    # ┏━━━━━━━━━━ Calculate Per-Asset Returns ━━━━━━━━━━┓
    asset_m1_rets = df_trades.groupby("asset")["return"].mean()
    asset_m2_rets = m2_df.groupby("asset")["return"].mean() if len(m2_df) > 0 else pd.Series(dtype=float)
    if has_m2_util:
        asset_m2_util_rets = (
            m2_util_df.groupby("asset")["return"].mean() if len(m2_util_df) > 0 else pd.Series(dtype=float)
        )

    # ┏━━━━━━━━━━ Calculate Execution Rate ━━━━━━━━━━┓
    execution_rate = m2_approved.sum() / len(m2_approved) * 100 if len(m2_approved) > 0 else 0

    # ┏━━━━━━━━━━ Calculate Win Rates ━━━━━━━━━━┓
    n_total = len(y_test)
    n_approved = int(m2_approved.sum())
    n_m2_good = int((m2_approved & (y_test == 1)).sum())
    m2_wr = n_m2_good / n_approved * 100 if n_approved > 0 else 0
    m1_good = int((y_test == 1).sum())
    m1_wr = m1_good / n_total * 100 if n_total > 0 else 0

    # ┏━━━━━━━━━━ Define Strategy Names ━━━━━━━━━━┓
    m2_name = f"M2 {mlabel} {thres_mode}" if thres_mode.startswith("OCP") else f"M2 {mlabel} selective"
    m2_util_name = f"M2 {mlabel} Utility"
    m1_name = f"{_m1_display_label(cfg)} (all trades)"
    bh_name = "Buy & Hold"

    # ┏━━━━━━━━━━ Calculate Annualization Factors ━━━━━━━━━━┓
    ann_bar = _annualization_factor(granularity)
    ann_horizon = np.sqrt(ann_bar ** 2 / horizon)

    # ┏━━━━━━━━━━ Calculate Strategy Statistics ━━━━━━━━━━┓
    strats = {}
    for name, eq in [(m2_name, m2_equity), (m1_name, m1_equity)]:
        h_rets = _equity_horizon_returns(eq, horizon) if len(eq) > horizon else np.array([])
        strats[name] = {"total_ret": (eq.iloc[-1] - 1) * 100 if len(eq) > 0 else 0,
                        "mdd": _calc_drawdown(eq.values) * 100 if len(eq) > 0 else 0,
                        "sharpe": _calc_sharpe(h_rets, ann_horizon)}
    
    # ┏━━━━━━━━━━ Add Buy & Hold Strategy Statistics ━━━━━━━━━━┓
    if has_bh:
        bh_h_rets = _equity_horizon_returns(bh_equity, horizon) if len(bh_equity) > horizon else np.array([])
        strats[bh_name] = {"total_ret": (bh_equity.iloc[-1] - 1) * 100,
                           "mdd": _calc_drawdown(bh_equity.values) * 100,
                           "sharpe": _calc_sharpe(bh_h_rets, ann_horizon)}
    
    # ┏━━━━━━━━━━ Add Utility Threshold Strategy Statistics ━━━━━━━━━━┓
    if has_m2_util:
        m2u_h_rets = _equity_horizon_returns(m2_util_equity, horizon) if len(m2_util_equity) > horizon else np.array([])
        n_m2_util = int(m2_util_approved.sum())
        m2_util_exec = n_m2_util / len(m2_util_approved) * 100 if len(m2_util_approved) > 0 else 0
        strats[m2_util_name] = {"total_ret": (m2_util_equity.iloc[-1] - 1) * 100 if len(m2_util_equity) > 0 else 0,
                                "mdd": _calc_drawdown(m2_util_equity.values) * 100 if len(m2_util_equity) > 0 else 0,
                                "sharpe": _calc_sharpe(m2u_h_rets, ann_horizon)}

    # ┏━━━━━━━━━━ Determine OCP vs Utility Threshold ━━━━━━━━━━┓
    direction_label = direction.upper()
    if thres_mode.startswith("OCP") and "_ocp_test_thresholds" in val_op:
        ocp_s_hats = val_op["_ocp_test_thresholds"]
        threshold = float(np.median(np.maximum(ocp_s_hats, 1.0 - ocp_s_hats)))
        constraint_tag = f"{thres_mode} Median-Adaptive"
    else:
        constraint_tag = "Utility-Opt" if val_op["constraint_satisfied"] else "fallback"
    fee_tag = f" fee={fee*100:.2f}%" if fee > 0 else ""

    # ┏━━━━━━━━━━ Plot Equity Curves ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(18, 8))
    plt.subplots_adjust(right=0.68)

    ax.plot((m2_equity - 1) * 100,
            label=f"{m2_name} (SR: {strats[m2_name]['sharpe']:.2f}, Exec: {execution_rate:.1f}%)",
            color="green",
            linewidth=3.0)
    if has_m2_util:
        ax.plot((m2_util_equity - 1) * 100,
                label=f"{m2_util_name} (SR: {strats[m2_util_name]['sharpe']:.2f}, Exec: {m2_util_exec:.1f}%)",
                color="#E67E22",
                linewidth=2.5,
                linestyle="--")
    ax.plot((m1_equity - 1) * 100,
            label=f"{m1_name} (SR: {strats[m1_name]['sharpe']:.2f})",
            color="blue",
            alpha=0.6,
            linewidth=2.0)
    if has_bh:
        ax.plot((bh_equity - 1) * 100,
                label=f"{bh_name} (SR: {strats[bh_name]['sharpe']:.2f})",
                color="gray",
                linestyle="--",
                linewidth=1.5)

    # ┏━━━━━━━━━━ Set Title ━━━━━━━━━━┓
    title = (f"Cumulative Returns (%) — {mlabel} {granularity.upper()}+{direction_label}"
             f"+{meta_mode.upper()} ({desc}) thr={threshold:.3f} ({constraint_tag}){fee_tag}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # ┏━━━━━━━━━━ Calculate Per-Asset Statistics ━━━━━━━━━━┓
    m1_counts = df_trades.groupby("asset").size()
    m2_counts = m2_df.groupby("asset").size() if len(m2_df) > 0 else pd.Series(dtype=int)

    all_assets = sorted(df_trades["asset"].unique())
    table_data = []
    if has_m2_util:
        m2_util_counts = m2_util_df.groupby("asset").size() if len(m2_util_df) > 0 else pd.Series(dtype=int)
        for asset in all_assets:
            m2_a  = asset_m2_rets.get(asset, 0.0) * 100
            m2u_a = asset_m2_util_rets.get(asset, 0.0) * 100
            m1_a  = asset_m1_rets.get(asset, 0.0) * 100
            bh_a  = asset_bh_rets.get(asset, 0.0) * 100 if has_bh else 0.0
            m2_n  = int(m2_counts.get(asset, 0))
            m2u_n = int(m2_util_counts.get(asset, 0))
            m1_n  = int(m1_counts.get(asset, 0))
            table_data.append(
                [
                    asset,
                    f"{m2_a:+.1f}% ({m2_n})",
                    f"{m2u_a:+.1f}% ({m2u_n})",
                    f"{m1_a:+.1f}% ({m1_n})",
                    f"{bh_a:+.1f}%",
                ]
            )
    else:
        for asset in all_assets:
            m2_a = asset_m2_rets.get(asset, 0.0) * 100
            m1_a = asset_m1_rets.get(asset, 0.0) * 100
            bh_a = asset_bh_rets.get(asset, 0.0) * 100 if has_bh else 0.0
            m2_n = int(m2_counts.get(asset, 0))
            m1_n = int(m1_counts.get(asset, 0))
            table_data.append([asset, f"{m2_a:+.1f}% ({m2_n})", f"{m1_a:+.1f}% ({m1_n})", f"{bh_a:+.1f}%"])

    # ┏━━━━━━━━━━ Calculate Portfolio Statistics ━━━━━━━━━━┓
    ptf = {k: strats[k]["total_ret"] for k in [m2_name, m1_name]}
    ptf[bh_name] = strats[bh_name]["total_ret"] if has_bh else 0.0
    if has_m2_util:
        ptf[m2_util_name] = strats[m2_util_name]["total_ret"]
        table_data.append(
            [
                "Portfolio Return",
                f"{ptf[m2_name]:+.2f}%",
                f"{ptf[m2_util_name]:+.2f}%",
                f"{ptf[m1_name]:+.2f}%",
                f"{ptf[bh_name]:+.2f}%",
            ]
        )
        avg_m2 = m2_df["return"].mean() * 100 if len(m2_df) > 0 else 0.0
        avg_m2u = m2_util_df["return"].mean() * 100 if len(m2_util_df) > 0 else 0.0
        avg_m1 = df_trades["return"].mean() * 100
        avg_bh = asset_bh_rets.mean() * 100 if has_bh and len(asset_bh_rets) > 0 else 0.0
        table_data.append(
            [
                "Avg Ret/Trade",
                f"{avg_m2:+.2f}% ({n_approved})",
                f"{avg_m2u:+.2f}% ({n_m2_util})",
                f"{avg_m1:+.2f}% ({n_total})",
                f"{avg_bh:+.2f}%",
            ]
        )
        mdd = {k: strats[k]["mdd"] for k in [m2_name, m2_util_name, m1_name]}
        mdd[bh_name] = strats[bh_name]["mdd"] if has_bh else 0.0
        table_data.append(
            [
                "Max Drawdown",
                f"{mdd[m2_name]:+.2f}%",
                f"{mdd[m2_util_name]:+.2f}%",
                f"{mdd[m1_name]:+.2f}%",
                f"{mdd[bh_name]:+.2f}%",
            ]
        )
        col_labels = ["Asset", "M2 OCP", "M2 Utility", _m1_display_label(cfg), "B&H"]
    else:
        table_data.append(
            ["Portfolio Return", f"{ptf[m2_name]:+.2f}%", f"{ptf[m1_name]:+.2f}%", f"{ptf[bh_name]:+.2f}%"]
        )
        avg_m2 = m2_df["return"].mean() * 100 if len(m2_df) > 0 else 0.0
        avg_m1 = df_trades["return"].mean() * 100
        avg_bh = asset_bh_rets.mean() * 100 if has_bh and len(asset_bh_rets) > 0 else 0.0
        table_data.append(
            [
                "Avg Ret/Trade",
                f"{avg_m2:+.2f}% ({n_approved})",
                f"{avg_m1:+.2f}% ({n_total})",
                f"{avg_bh:+.2f}%",
            ]
        )
        mdd = {k: strats[k]["mdd"] for k in [m2_name, m1_name]}
        mdd[bh_name] = strats[bh_name]["mdd"] if has_bh else 0.0
        table_data.append(
            ["Max Drawdown", f"{mdd[m2_name]:+.2f}%", f"{mdd[m1_name]:+.2f}%", f"{mdd[bh_name]:+.2f}%"]
        )
        col_labels = ["Asset", f"M2 {mlabel}", _m1_display_label(cfg), "B&H"]

    # ┏━━━━━━━━━━ Set Column Widths ━━━━━━━━━━┓
    n_cols_tbl = len(col_labels)
    if n_cols_tbl == 5:
        col_widths = [0.28, 0.22, 0.22, 0.22, 0.14]
    else:
        col_widths = [0.35, 0.27, 0.27, 0.18]

    # ┏━━━━━━━━━━ Create Table ━━━━━━━━━━┓
    the_table = plt.table(cellText  = table_data,
                          colLabels = col_labels,
                          loc       = "right",
                          bbox      = [1.03, 0.0, 0.55, 1.0],
                          cellLoc   = "center",
                          colWidths = col_widths)
    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    n_data_rows = len(table_data)
    n_summary = 3
    first_summary_row = n_data_rows - n_summary + 1

    # ┏━━━━━━━━━━ Get All Values ━━━━━━━━━━┓
    all_vals = []
    for (row, col), cell in the_table.get_celld().items():
        if row > 0 and col > 0:
            val_str = cell.get_text().get_text().split("%")[0].strip()
            try:
                all_vals.append(float(val_str))
            except ValueError:
                pass
    abs_max = max(abs(v) for v in all_vals) if all_vals else 1.0

    # ┏━━━━━━━━━━ Color Function ━━━━━━━━━━┓
    def _val_color(val: float, scale: float) -> str:
        intensity = min(abs(val) / scale, 1.0)
        if val > 0:
            r = int(255 - intensity * (255 - 46))
            g = int(255 - intensity * (255 - 139))
            b = int(255 - intensity * (255 - 87))
        elif val < 0:
            r = int(255 - intensity * (255 - 180))
            g = int(255 - intensity * (255 - 40))
            b = int(255 - intensity * (255 - 40))
        else:
            return "#ffffff"
        return f"#{r:02x}{g:02x}{b:02x}"

    # ┏━━━━━━━━━━ Color Cells ━━━━━━━━━━┓
    for (row, col), cell in the_table.get_celld().items():
        cell.get_text().set_weight("bold")
        cell.set_text_props(ha="center", va="center")
        if row > 0 and col > 0:
            val_str = cell.get_text().get_text().split("%")[0].strip()
            try:
                val = float(val_str)
                cell.set_facecolor(_val_color(val, abs_max))
                if abs(val) / abs_max > 0.55:
                    cell.get_text().set_color("white")
            except ValueError:
                pass
        if row == 0 and col >= 0:
            cell.set_facecolor("#2b5797")
            cell.get_text().set_color("white")
        elif row == 0:
            cell.set_facecolor("#f2f2f2")
        if row >= first_summary_row:
            cell.set_fontsize(9)
            if col == 0:
                cell.set_facecolor("#2b5797")
                cell.get_text().set_color("white")

    # ┏━━━━━━━━━━ Save Plot ━━━━━━━━━━┓
    ax.text(1.225, 1.02, "Individual Asset Performance", transform=ax.transAxes, fontsize=12, fontweight="bold", ha="center")
    plot_path = save_dir / f"{file_prefix}_curve.png"
    fig.savefig(str(plot_path), bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ┏━━━━━━━━━━ Calculate Statistics ━━━━━━━━━━┓
    app_rets = net_returns[m2_approved]
    rej_rets = net_returns[~m2_approved]
    avg_app  = float(np.nanmean(app_rets)) * 100 if n_approved > 0 and np.any(np.isfinite(app_rets)) else 0.0
    avg_rej  = float(np.nanmean(rej_rets)) * 100 if (~m2_approved).sum() > 0 and np.any(np.isfinite(rej_rets)) else 0.0
    delta    = avg_app - avg_rej

    # ┏━━━━━━━━━━ Build Lines ━━━━━━━━━━┓
    lines = [
        "=" * 60,
        f"FINANCIAL BACKTEST: {mlabel} {granularity.upper()} {direction_label} {meta_mode.upper()} ({desc})",
        f"Period: {test_start.date()} to {test_end.date()}",
        f"Threshold: {threshold:.4f} ({constraint_tag}) | Fee: {fee*100:.3f}%",
        "=" * 60,
        f"Total Test Trades:     {n_total}",
        f"{_m1_display_label(cfg)} Baseline Win-Rate:  {m1_wr:.1f}% ({m1_good}/{n_total})",
        "-" * 60,
        f"M2 Approved Trades:    {n_approved} ({execution_rate:.1f}% execution)",
        f"M2 Rejected Trades:    {n_total - n_approved}",
        f"M2 Win-Rate:           {m2_wr:.1f}% ({n_m2_good}/{n_approved})",
        "-" * 60,
        f"Avg Return APPROVED:   {avg_app:+.3f}%",
        f"Avg Return REJECTED:   {avg_rej:+.3f}%",
    ]

    # ┏━━━━━━━━━━ Add Delta Line ━━━━━━━━━━┓
    if delta > 0:
        lines.append(f"M2 Edge: Approved trades yield {delta:.3f}% more per trade")
    else:
        lines.append(f"M2 Failure: Rejected trades were {-delta:.3f}% more profitable")
    lines.extend(["=" * 60,
                  f"{'Strategy':<25} {'Total Ret':>10} {'MaxDD':>8} {'Sharpe':>8}",
                  "-" * 60])
    for name in [m2_name, m1_name] + ([bh_name] if has_bh else []):
        s = strats[name]
        lines.append(f"{name:<25} {s['total_ret']:>+9.2f}% {s['mdd']:>+7.2f}% {s['sharpe']:>7.2f}")
    lines.append("=" * 60)

    # ┏━━━━━━━━━━ Save Report ━━━━━━━━━━┓
    report = "\n".join(lines)
    print(f"\n{report}")
    report_path = save_dir / f"{file_prefix}_ROI.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # ┏━━━━━━━━━━ Save Trades ━━━━━━━━━━┓
    trades_dump = df_trades.copy()
    trades_dump["date"] = trades_dump["date"].astype(str)
    trades_dump["direction"] = direction
    trades_dump["return_pct"] = trades_dump["return"] * 100
    trades_path = save_dir / f"{file_prefix}_trades.csv"
    trades_dump.to_csv(trades_path, index=False, float_format="%.6f")

    # ┏━━━━━━━━━━ Print Summary ━━━━━━━━━━┓
    print(f"  Equity curve saved: {plot_path.name}")
    print(f"  ROI report saved:   {report_path.name}")
    print(f"  Trade dump saved:   {trades_path.name}")

    return {
        "threshold": float(threshold),
        "constraint_satisfied": bool(val_op["constraint_satisfied"]),
        "execution_rate": float(round(execution_rate, 2)),
        "n_total_trades": int(n_total),
        "n_m2_trades": int(n_approved),
        "m1_win_rate": float(round(m1_wr, 2)),
        "m2_win_rate": float(round(m2_wr, 2)),
        "m2_total_return": float(round(strats[m2_name]["total_ret"], 4)),
        "m1_total_return": float(round(strats[m1_name]["total_ret"], 4)),
        "bh_total_return": float(round(strats[bh_name]["total_ret"], 4)) if has_bh else None,
        "m2_sharpe": float(round(strats[m2_name]["sharpe"], 4)),
        "m1_sharpe": float(round(strats[m1_name]["sharpe"], 4)),
        "bh_sharpe": float(round(strats[bh_name]["sharpe"], 4)) if has_bh else None,
        "m2_max_drawdown": float(round(strats[m2_name]["mdd"], 4)),
        "m1_max_drawdown": float(round(strats[m1_name]["mdd"], 4)),
        "bh_max_drawdown": float(round(strats[bh_name]["mdd"], 4)) if has_bh else None,
        "fee": float(fee),
    }


# ┏━━━━━━━━━━ Combined UP + DOWN Backtest ━━━━━━━━━━┓
def run_combined_backtest(up_dir: str | Path,
                          dn_dir: str | Path,
                          save_dir: str | Path,
                          config: dict,
                          granularity="1d",
                          model_name: str = "rf",
                          file_prefix: str = "combined_backtest"):
    """Combine UP and DOWN trade CSVs per granularity and produce a joint backtest.

    Reads ``10_backtest_all_trades.csv`` from every ``{gran}_tp/`` subfolder in
    *up_dir* and *dn_dir*, merges UP+DOWN trades, and writes per-granularity
    equity curves + per-asset performance tables into *save_dir*.

    The performance table splits M2 and M1 columns into UP | DOWN sub-columns
    so each direction's contribution is visible at a glance.
    """
    up_dir  = Path(up_dir)
    dn_dir  = Path(dn_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mlabel = _model_label(model_name)  # TODO why cant we jsut stick to one nameing convention
    m1_label = config["experiment"]["m1"]
    horizon = int(config["data"]["load"]["forecast_horizon"])

    # Discover granularities present in both directions
    # up_grans = {p.name for p in up_dir.iterdir() if p.is_dir() and (p / "10_backtest_all_trades.csv").exists()}
    # dn_grans = {p.name for p in dn_dir.iterdir() if p.is_dir() and (p / "10_backtest_all_trades.csv").exists()}
    # common_grans = sorted(up_grans & dn_grans)

    # if not common_grans:
    #     print(f"  [combined_backtest] No matching granularities found in UP ({up_dir}) and DOWN ({dn_dir})")
    #     return

    print(f"\n{'='*60}")
    print(f"[combined_backtest] Merging UP+DOWN trades for: {', '.join(granularity)}")
    print(f"[combined_backtest] Output: {save_dir}")
    print(f"{'='*60}\n")

    up_csv = up_dir / f"{granularity}_{config['data']['load']['meta_label_mode']}" / "10_backtest_all_trades.csv"
    dn_csv = dn_dir / f"{granularity}_{config['data']['load']['meta_label_mode']}" / "10_backtest_all_trades.csv"

    df_up = pd.read_csv(up_csv, parse_dates=["date"])
    df_dn = pd.read_csv(dn_csv, parse_dates=["date"])

    # Tag direction if missing
    if "direction" not in df_up.columns:
        df_up["direction"] = "up"
    if "direction" not in df_dn.columns:
        df_dn["direction"] = "down"

    df_all = pd.concat([df_up, df_dn], ignore_index=True).sort_values("date").reset_index(drop=True)

    # Gran name from folder (e.g. "6h_tp" -> "6h")
    # gran_name = granularity.split("_")[0]
    ann_bar = _annualization_factor(granularity)
    ann_horizon = np.sqrt(ann_bar ** 2 / horizon)

    # M2 approved vs all (M1) trades
    m2_mask = df_all["m2_approved"].astype(bool)
    df_m2 = df_all[m2_mask]
    full_idx = pd.DatetimeIndex(sorted(df_all["date"].unique()))

    # Build equity curves: combined M2, combined M1, and per-direction
    m1_eq, _ = _build_spread_equity(df_all, full_idx, horizon)
    m2_eq, _ = _build_spread_equity(df_m2,  full_idx, horizon)

    df_m2_up = df_m2[df_m2["direction"] == "up"]
    df_m2_dn = df_m2[df_m2["direction"] == "down"]
    df_m1_up = df_all[df_all["direction"] == "up"]
    df_m1_dn = df_all[df_all["direction"] == "down"]

    m2_up_eq, _ = _build_spread_equity(df_m2_up, full_idx, horizon)
    m2_dn_eq, _ = _build_spread_equity(df_m2_dn, full_idx, horizon)

    # Buy & Hold
    raw_close = _load_raw_close_prices(config, granularity, direction="up")  # TODO is it correct that we hard code direction here?
    has_bh = len(raw_close) > 0
    if has_bh:
        t_start, t_end = full_idx.min(), full_idx.max()
        raw_close = raw_close[(raw_close["date"] >= t_start) & (raw_close["date"] <= t_end)]
        bh_pivot = raw_close.pivot_table(index="date", columns="asset", values="close")
        if len(bh_pivot) > 0:
            bh_first = bh_pivot.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
            bh_equity = (bh_pivot / bh_first).mean(axis=1)
        else:
            has_bh = False

    # Calculate metrics for each strategy
    def _metrics(eq):
        h_rets = _equity_horizon_returns(eq, horizon) if len(eq) > horizon else np.array([])
        return {"total_ret": (eq.iloc[-1] - 1) * 100 if len(eq) > 0 else 0.0,
                "mdd": _calc_drawdown(eq.values) * 100 if len(eq) > 0 else 0.0,
                "sharpe": _calc_sharpe(h_rets, ann_horizon)}

    # M1 per-direction equity curves
    m1_up_eq, _ = _build_spread_equity(df_m1_up, full_idx, horizon)
    m1_dn_eq, _ = _build_spread_equity(df_m1_dn, full_idx, horizon)

    strats = {f"M2 {mlabel} (UP+DN)": _metrics(m2_eq),
              f"M2 {mlabel} UP":       _metrics(m2_up_eq),
              f"M2 {mlabel} DN":       _metrics(m2_dn_eq),
              f"{m1_label} (UP+DN)":   _metrics(m1_eq),
              f"{m1_label} UP":        _metrics(m1_up_eq),
              f"{m1_label} DN":        _metrics(m1_dn_eq)}
    if has_bh:
        strats["Buy & Hold"] = _metrics(bh_equity)

    # ┏━━━━━━━━━━ Plot ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(20, 9))
    plt.subplots_adjust(right=0.52)

    # M2 family: green palette
    s_m2 = strats[f"M2 {mlabel} (UP+DN)"]
    ax.plot((m2_eq - 1) * 100, label=f"M2 {mlabel} Combined (SR:{s_m2['sharpe']:.2f})",
            color="#1a7a3a", linewidth=3.0)
    s_up = strats[f"M2 {mlabel} UP"]
    ax.plot((m2_up_eq - 1) * 100, label=f"M2 {mlabel} UP (SR:{s_up['sharpe']:.2f})",
            color="#5cb85c", linewidth=1.6, linestyle="--")
    s_dn = strats[f"M2 {mlabel} DN"]
    ax.plot((m2_dn_eq - 1) * 100, label=f"M2 {mlabel} DN (SR:{s_dn['sharpe']:.2f})",
            color="#a3d977", linewidth=1.6, linestyle=":")

    # M1 family: blue palette
    s_m1 = strats[f"{m1_label} (UP+DN)"]
    ax.plot((m1_eq - 1) * 100, label=f"{m1_label} Combined (SR:{s_m1['sharpe']:.2f})",
            color="#1a3a7a", linewidth=2.5)
    s_m1_up = strats[f"{m1_label} UP"]
    ax.plot((m1_up_eq - 1) * 100, label=f"{m1_label} UP (SR:{s_m1_up['sharpe']:.2f})",
            color="#5b9bd5", linewidth=1.4, linestyle="--")
    s_m1_dn = strats[f"{m1_label} DN"]
    ax.plot((m1_dn_eq - 1) * 100, label=f"{m1_label} DN (SR:{s_m1_dn['sharpe']:.2f})",
            color="#9dc3e6", linewidth=1.4, linestyle=":")

    # B&H: neutral gray
    if has_bh:
        s_bh = strats["Buy & Hold"]
        ax.plot((bh_equity - 1) * 100, label=f"B&H (SR:{s_bh['sharpe']:.2f})",
                color="#888888", linestyle="--", linewidth=1.5)

    ax.set_title(f"Combined UP+DOWN — {mlabel} {granularity.upper()} TP  |  horizon={horizon}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # ┏━━━━━━━━━━ Per-asset table with Combined + UP + DOWN sub-columns ━━━━━━━━━━┓
    all_assets = sorted(df_all["asset"].unique())

    # Precompute B&H per-asset
    asset_bh_map = {}
    if has_bh:
        bh_pivot_close = raw_close.pivot_table(index="date", columns="asset", values="close")
        for asset in all_assets:
            if asset in bh_pivot_close.columns:
                c = bh_pivot_close[asset].dropna()
                asset_bh_map[asset] = (c.iloc[-1] / c.iloc[0] - 1) * 100 if len(c) > 1 else 0.0
            else:
                asset_bh_map[asset] = 0.0

    col_labels = ["Asset",
                  f"M2 {mlabel}\n(UP+DN)", f"M2 {mlabel}\nUP", f"M2 {mlabel}\nDOWN",
                  f"{m1_label}\n(UP+DN)", f"{m1_label}\nUP", f"{m1_label}\nDOWN",
                  "B&H"]

    table_data = []
    for asset in all_assets:
        m2u = df_m2_up[df_m2_up["asset"] == asset]
        m2d = df_m2_dn[df_m2_dn["asset"] == asset]
        m2c = df_m2[df_m2["asset"] == asset]
        m1u = df_m1_up[df_m1_up["asset"] == asset]
        m1d = df_m1_dn[df_m1_dn["asset"] == asset]
        m1c = df_all[df_all["asset"] == asset]

        m2c_ret = m2c["return"].mean() * 100 if len(m2c) > 0 else 0.0
        m2u_ret = m2u["return"].mean() * 100 if len(m2u) > 0 else 0.0
        m2d_ret = m2d["return"].mean() * 100 if len(m2d) > 0 else 0.0
        m1c_ret = m1c["return"].mean() * 100 if len(m1c) > 0 else 0.0
        m1u_ret = m1u["return"].mean() * 100 if len(m1u) > 0 else 0.0
        m1d_ret = m1d["return"].mean() * 100 if len(m1d) > 0 else 0.0
        bh_a = asset_bh_map.get(asset, 0.0)

        table_data.append([asset,
                           f"{m2c_ret:+.1f}% ({len(m2c)})",
                           f"{m2u_ret:+.1f}% ({len(m2u)})",
                           f"{m2d_ret:+.1f}% ({len(m2d)})",
                           f"{m1c_ret:+.1f}% ({len(m1c)})",
                           f"{m1u_ret:+.1f}% ({len(m1u)})",
                           f"{m1d_ret:+.1f}% ({len(m1d)})",
                           f"{bh_a:+.1f}%"])

    # Summary rows
    n_m2 = int(m2_mask.sum())
    n_m2_up = len(df_m2_up); n_m2_dn = len(df_m2_dn)
    n_m1 = len(df_all)
    n_m1_up = len(df_m1_up); n_m1_dn = len(df_m1_dn)
    avg_m2 = df_m2["return"].mean() * 100 if n_m2 > 0 else 0.0
    avg_m2_up = df_m2_up["return"].mean() * 100 if n_m2_up > 0 else 0.0
    avg_m2_dn = df_m2_dn["return"].mean() * 100 if n_m2_dn > 0 else 0.0
    avg_m1 = df_all["return"].mean() * 100 if n_m1 > 0 else 0.0
    avg_m1_up = df_m1_up["return"].mean() * 100 if n_m1_up > 0 else 0.0
    avg_m1_dn = df_m1_dn["return"].mean() * 100 if n_m1_dn > 0 else 0.0
    bh_avg = np.mean(list(asset_bh_map.values())) if asset_bh_map else 0.0

    table_data.append(["Ptf Return",
                       f"{s_m2['total_ret']:+.2f}%",
                       f"{s_up['total_ret']:+.2f}%",
                       f"{s_dn['total_ret']:+.2f}%",
                       f"{s_m1['total_ret']:+.2f}%",
                       f"{s_m1_up['total_ret']:+.2f}%",
                       f"{s_m1_dn['total_ret']:+.2f}%",
                       f"{strats.get('Buy & Hold', {}).get('total_ret', 0.0):+.2f}%"])
    table_data.append(["Avg Ret/Trade",
                       f"{avg_m2:+.2f}% ({n_m2})",
                       f"{avg_m2_up:+.2f}% ({n_m2_up})",
                       f"{avg_m2_dn:+.2f}% ({n_m2_dn})",
                       f"{avg_m1:+.2f}% ({n_m1})",
                       f"{avg_m1_up:+.2f}% ({n_m1_up})",
                       f"{avg_m1_dn:+.2f}% ({n_m1_dn})",
                       f"{bh_avg:+.2f}%"])
    table_data.append(["Max DD",
                       f"{s_m2['mdd']:+.2f}%",
                       f"{s_up['mdd']:+.2f}%",
                       f"{s_dn['mdd']:+.2f}%",
                       f"{s_m1['mdd']:+.2f}%",
                       f"{s_m1_up['mdd']:+.2f}%",
                       f"{s_m1_dn['mdd']:+.2f}%",
                       f"{strats.get('Buy & Hold', {}).get('mdd', 0.0):+.2f}%"])

    # Render table
    col_widths = [0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.08]
    the_table = plt.table(cellText=table_data, colLabels=col_labels, loc="right",
                          bbox=[1.03, 0.0, 0.92, 1.0], cellLoc="center", colWidths=col_widths)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7)

    n_data_rows = len(table_data)
    first_summary_row = n_data_rows - 3 + 1

    # Color coding
    all_vals = []
    for (r, c), cell in the_table.get_celld().items():
        if r > 0 and c > 0:
            v_str = cell.get_text().get_text().split("%")[0].strip()
            try:
                all_vals.append(float(v_str))
            except ValueError:
                pass
    abs_max = max(abs(v) for v in all_vals) if all_vals else 1.0

    def _val_color(val, scale):
        intensity = min(abs(val) / scale, 1.0)
        if val > 0:
            r = int(255 - intensity * 209); g = int(255 - intensity * 116); b = int(255 - intensity * 168)
        elif val < 0:
            r = int(255 - intensity * 75); g = int(255 - intensity * 215); b = int(255 - intensity * 215)
        else:
            return "#ffffff"
        return f"#{r:02x}{g:02x}{b:02x}"

    for (r, c), cell in the_table.get_celld().items():
        cell.get_text().set_weight("bold")
        cell.set_text_props(ha="center", va="center")
        if r > 0 and c > 0:
            v_str = cell.get_text().get_text().split("%")[0].strip()
            try:
                val = float(v_str)
                cell.set_facecolor(_val_color(val, abs_max))
                if abs(val) / abs_max > 0.55:
                    cell.get_text().set_color("white")
            except ValueError:
                pass
        if r == 0:
            cell.set_facecolor("#2b5797")
            cell.get_text().set_color("white")
        if r >= first_summary_row:
            cell.set_fontsize(8)
            if c == 0:
                cell.set_facecolor("#2b5797")
                cell.get_text().set_color("white")

    ax.text(1.42, 1.02, "Per-Asset Performance (UP | DOWN)", transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="center")

    gran_save = save_dir / f"{granularity}_{config['data']['load']['meta_label_mode']}"
    gran_save.mkdir(parents=True, exist_ok=True)
    plot_path = gran_save / f"{file_prefix}_curve.png"
    fig.savefig(str(plot_path), bbox_inches="tight", dpi=200)
    plt.close(fig)

    # Save combined trades CSV
    trades_path = gran_save / f"{file_prefix}_trades.csv"
    df_all.to_csv(trades_path, index=False, float_format="%.6f")

    # Save text report
    n_total = len(df_all)
    n_m2 = int(m2_mask.sum())
    m2_wr_up = df_m2_up["label"].mean() * 100 if len(df_m2_up) > 0 else 0
    m2_wr_dn = df_m2_dn["label"].mean() * 100 if len(df_m2_dn) > 0 else 0
    m1_wr_up = df_m1_up["label"].mean() * 100 if len(df_m1_up) > 0 else 0
    m1_wr_dn = df_m1_dn["label"].mean() * 100 if len(df_m1_dn) > 0 else 0

    lines = [
        "=" * 70,
        f"COMBINED UP+DOWN BACKTEST: {mlabel} {granularity.upper()} TP",
        f"Period: {df_all['date'].min().date()} to {df_all['date'].max().date()}",
        "=" * 70,
        f"Total Test Trades:  {n_total} (UP: {len(df_m1_up)}, DOWN: {len(df_m1_dn)})",
        f"M2 Approved:        {n_m2} (UP: {n_m2_up}, DOWN: {n_m2_dn})",
        f"M2 WinRate UP:      {m2_wr_up:.1f}%   |  M1 WinRate UP:   {m1_wr_up:.1f}%",
        f"M2 WinRate DOWN:    {m2_wr_dn:.1f}%   |  M1 WinRate DOWN: {m1_wr_dn:.1f}%",
        "-" * 70,
        f"{'Strategy':<30} {'Total Ret':>10} {'MaxDD':>8} {'Sharpe':>8}",
        "-" * 70,
    ]
    for name, s in strats.items():
        lines.append(f"{name:<30} {s['total_ret']:>+9.2f}% {s['mdd']:>+7.2f}% {s['sharpe']:>7.2f}")
    lines.append("=" * 70)

    report_path = gran_save / f"{file_prefix}_ROI.txt"
    report = "\n".join(lines)
    print(report)
    with open(report_path, "w") as f:
        f.write(report)

    print(f"  Saved: {plot_path.name}, {trades_path.name}, {report_path.name}\n")
