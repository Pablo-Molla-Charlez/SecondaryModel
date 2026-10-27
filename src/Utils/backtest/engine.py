"""Financial backtest helpers extracted from m2_pipeline.py."""

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
    """Sharpe = mean / std * ann, where ann = sqrt(observations_per_year).

    NaN-safe: drops NaN entries before computing (mean/std would otherwise
    propagate NaN and poison the result even when only a few entries are NaN).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    std = np.std(r)
    if std == 0:
        return 0.0
    return float(np.mean(r) / std * ann)


# ┏━━━━━━━━━━ Calculate Max Drawdown ━━━━━━━━━━┓
def _calc_drawdown(equity: np.ndarray) -> float:
    """Max drawdown from an equity curve."""
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / peaks
    return float(np.min(dd))


# ┏━━━━━━━━━━ Position sizing (cash-account engine only) ━━━━━━━━━━┓
# Confidence-weighted sizing with a base floor: a p=0.5 signal still stakes _SIZE_BASE of
# the per-trade allowance, a near-certain signal stakes ~all of it.
_SIZE_BASE = 0.10
# Per-trade EXPOSURE multiplier: each signal stakes _EXPOSURE × frac × free_cash (not the
# full free cash), so early clustered signals cannot consume the whole account.
_EXPOSURE = 0.10
# Single-asset books have no cross-asset competition for cash → no per-trade cap needed.
_EXPOSURE_SINGLE_ASSET = 1.0


def _z_size(prob: float) -> float:
    """QuantConnect-style ZSize with a base floor → fraction of the per-trade allowance."""
    if prob != prob:          # NaN
        return _SIZE_BASE
    p = min(max(float(prob), 0.5), 1.0 - 1e-9)
    zsize = (p - 0.5) / (1.0 - p) if p < 1.0 else 1.0
    zsize = min(zsize, 1.0)
    return _SIZE_BASE + (1.0 - _SIZE_BASE) * zsize


# ┏━━━━━━━━━━ Build Cash-Account Equity (NEW engine) ━━━━━━━━━━┓
def _build_cash_equity(trades_df: pd.DataFrame, timeline: pd.DatetimeIndex, horizon: int,
                       capital_log: bool = False, flat_size: bool = False,
                       close: pd.DataFrame | None = None, fee: float = 0.002,
                       direction: str | None = None):
    """Stateful CASH-FLOW equity simulator (QuantConnect-style), normalized to start at 1.0.

    Models a real cash account instead of a vectorized concurrency average. Equity =
    free_cash + Σ(open-position book value), starting at 1.0 (all cash). Walking the
    timeline bar by bar:
      1. EXITS first: any position reaching its `horizon`-bar mark is liquidated — its
         locked capital × (1 + realized trade return) returns to free cash. (DOWN trades
         are symmetric shorts: the stored `return` is already direction-signed upstream.)
      2. ENTRIES next: every trade entered on this bar requests `_EXPOSURE × frac ×
         free_cash`, where frac = _z_size(m2_prob) unless ``flat_size=True`` (then
         frac = 1.0). Over-subscribed bars are filled PRO-RATA, so the result does not
         depend on trade ordering.
      3. Equity is marked as free cash + the value of open positions. With ``close``
         (bar closes, columns = assets) each open position is MARKED TO MARKET:
         locked capital × (1 + signed move since entry − fee), the sign taken from the
         trades' ``direction`` column (or ``direction``). Without ``close`` positions are
         held at entry cost until exit, so moves inside the horizon never reach the curve
         and drawdown is understated. Exits always book the stored ``return``, so the
         final equity is the same either way; only the path between exits differs.

    ``flat_size=True`` MUST be used for the M1 "trade-all" baseline: otherwise M1 borrows
    the M2 model's probabilities to size its positions, and the (fixed) M1 trade set would
    produce a different curve for every M2 model — M1 is the model-free yardstick.

    Returns ``(equity, realized_returns)``, or ``(equity, realized_returns, ledger)`` when
    ``capital_log=True``.
    """
    # ┏━━━━━━━━━━ Handle Empty Trades ━━━━━━━━━━┓
    if len(trades_df) == 0:
        empty_eq = pd.Series(1.0, index=timeline)
        if capital_log:
            ledger = pd.DataFrame({"capital": 1.0, "free_cash": 1.0, "locked_capital": 0.0,
                                   "invested_new": 0.0, "realized_pnl": 0.0, "n_open": 0},
                                  index=timeline)
            return empty_eq, np.array([]), ledger
        return empty_eq, np.array([])

    n_bars = len(timeline)
    bar_map = {d: i for i, d in enumerate(timeline)}

    # Single-asset book → no cross-asset competition for cash, drop the per-trade cap.
    n_assets = trades_df["asset"].nunique() if "asset" in trades_df.columns else 1
    exposure = _EXPOSURE_SINGLE_ASSET if n_assets <= 1 else _EXPOSURE

    # ┏━━━━━━━━━━ Group entries by bar index (entries off-timeline are dropped) ━━━━━━━━━━┓
    rets = trades_df["return"].values
    probs = (trades_df["m2_prob"].values if "m2_prob" in trades_df.columns
             else np.full(len(trades_df), np.nan))
    mtm = close is not None
    if mtm:
        C = close.reindex(timeline).ffill()
        if "direction" in trades_df.columns:
            signs = np.where(trades_df["direction"].astype(str).str.lower() == "down", -1.0, 1.0)
        elif direction is not None:
            signs = np.full(len(trades_df), -1.0 if direction.lower() == "down" else 1.0)
        else:
            raise ValueError("mark-to-market needs a 'direction' column or the direction argument")
        assets = trades_df["asset"].values
    entries_at = {}   # bar_idx → list of (return, prob, k)
    for k in range(len(trades_df)):
        bi = bar_map.get(trades_df["date"].iloc[k])
        if bi is None:
            continue
        entries_at.setdefault(bi, []).append(
            (float(rets[k]), float(probs[k]) if probs[k] == probs[k] else np.nan, k))

    # ┏━━━━━━━━━━ Sequential cash walk ━━━━━━━━━━┓
    cash = 1.0                          # free cash (fraction of starting capital)
    equity_curve = np.empty(n_bars)
    open_pos = []                       # [exit_bar_idx, locked_capital, trade_return, mark_fn]
    realized = []
    led_cash = np.empty(n_bars); led_locked = np.empty(n_bars)
    led_inv = np.zeros(n_bars); led_pnl = np.zeros(n_bars)
    led_nopen = np.zeros(n_bars, dtype=int)

    for b in range(n_bars):
        # 1) EXITS at this bar → return capital×(1+ret) to cash
        still_open = []
        for ex_bar, cap, r, mk in open_pos:
            if ex_bar <= b:
                cash += cap * (1.0 + r)
                realized.append(r)
                led_pnl[b] += cap * r
            else:
                still_open.append([ex_bar, cap, r, mk])
        open_pos = still_open

        # 2) ENTRIES at this bar → request exposure×frac×free_cash, pro-rata if over-subscribed
        todays = entries_at.get(b, [])
        if todays and cash > 1e-12:
            reqs = [(exposure * (1.0 if flat_size else _z_size(p)) * cash, r, k) for (r, p, k) in todays]
            total_req = sum(q for q, _, _ in reqs)
            scale = min(1.0, cash / total_req) if total_req > 1e-12 else 0.0
            # Clamp to the LAST bar: the exit loop only runs for b ≤ n_bars-1, so an
            # exit_bar of n_bars would never liquidate and its P&L would be lost.
            exit_bar = min(b + horizon, n_bars - 1)
            for q, r, k in reqs:
                cap = q * scale
                if cap <= 1e-12:
                    continue
                cash -= cap
                led_inv[b] += cap
                # (asset column, entry close, direction sign) for marking; None = hold at cost
                mk = (C[assets[k]], float(C[assets[k]].iloc[b]), signs[k]) if mtm else None
                open_pos.append([exit_bar, cap, r, mk])

        # 3) Mark equity = free cash + open positions (marked to market when close is given)
        locked = sum(cap for _, cap, _, _ in open_pos)
        if mtm:
            held = sum(cap * (1.0 + mk[2] * (mk[0].iloc[b] / mk[1] - 1.0) - fee)
                       for _, cap, _, mk in open_pos)
        else:
            held = locked
        equity_curve[b] = cash + held
        led_cash[b] = cash; led_locked[b] = locked; led_nopen[b] = len(open_pos)

    equity = pd.Series(equity_curve, index=timeline)
    if capital_log:
        ledger = pd.DataFrame({"capital": equity_curve, "free_cash": led_cash,
                               "locked_capital": led_locked, "invested_new": led_inv,
                               "realized_pnl": led_pnl, "n_open": led_nopen},
                              index=timeline)
        return equity, np.asarray(realized, dtype=float), ledger
    return equity, np.asarray(realized, dtype=float)


# ┏━━━━━━━━━━ Build Spread Equity (OLD engine — concurrency average) ━━━━━━━━━━┓
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


# ┏━━━━━━━━━━ Cash-engine artifact emitter (curve + ROI + trades CSV) ━━━━━━━━━━┓
def _emit_cash_engine_artifacts(*, df_trades, m2_df, full_idx, horizon, save_dir,
                                file_prefix, cfg, mlabel, m2_name, m1_name, bh_name,
                                bh_equity, has_bh, ann_horizon, granularity, direction,
                                meta_mode, desc, threshold, constraint_tag, fee,
                                execution_rate, test_start, test_end, n_total, n_approved,
                                m1_wr, m2_wr, m1_good, n_m2_good, avg_app, avg_rej,
                                asset_m2_rets, asset_m1_rets, asset_bh_rets, close=None):
    """Emit the NEW cash-account engine's own curve / ROI / trades CSV.

    Mirrors the old engine's three artifacts under a distinct ``file_prefix`` so both
    versions can be compared side by side. The trade set and fee-adjusted returns are
    passed in already computed — only the equity simulation differs.
    """
    save_dir = Path(save_dir)

    # ┏━━━━━━━━━━ Equity curves (M1 flat-sized; M2 confidence-sized) ━━━━━━━━━━┓
    # open positions marked to market at the bar closes (see _build_cash_equity)
    m1_equity_c, _m1_real = _build_cash_equity(df_trades, full_idx, horizon, flat_size=True,
                                               close=close, fee=fee, direction=direction)
    m2_equity_c, _m2_real, m2_ledger = _build_cash_equity(m2_df, full_idx, horizon,
                                                          capital_log=True, close=close,
                                                          fee=fee, direction=direction)

    # ┏━━━━━━━━━━ Strategy statistics ━━━━━━━━━━┓
    strats = {}
    for name, eq in [(m2_name, m2_equity_c), (m1_name, m1_equity_c)]:
        h_rets = _equity_horizon_returns(eq, horizon) if len(eq) > horizon else np.array([])
        strats[name] = {"total_ret": (eq.iloc[-1] - 1) * 100 if len(eq) > 0 else 0,
                        "mdd": _calc_drawdown(eq.values) * 100 if len(eq) > 0 else 0,
                        "sharpe": _calc_sharpe(h_rets, ann_horizon)}
    if has_bh and bh_equity is not None:
        bh_h = _equity_horizon_returns(bh_equity, horizon) if len(bh_equity) > horizon else np.array([])
        strats[bh_name] = {"total_ret": (bh_equity.iloc[-1] - 1) * 100,
                           "mdd": _calc_drawdown(bh_equity.values) * 100,
                           "sharpe": _calc_sharpe(bh_h, ann_horizon)}

    # ┏━━━━━━━━━━ Plot Equity Curves ━━━━━━━━━━┓
    # Mirrors the old engine's figure exactly (curve + per-asset table), so the two
    # images differ ONLY by the equity math behind them.
    direction_label = direction.upper()
    fee_tag = f" fee={fee*100:.2f}%" if fee > 0 else ""
    fig, ax = plt.subplots(figsize=(18, 8))
    plt.subplots_adjust(right=0.68)

    ax.plot((m2_equity_c - 1) * 100,
            label=f"{m2_name} (SR: {strats[m2_name]['sharpe']:.2f}, Exec: {execution_rate:.1f}%)",
            color="green",
            linewidth=3.0)
    ax.plot((m1_equity_c - 1) * 100,
            label=f"{m1_name} (SR: {strats[m1_name]['sharpe']:.2f})",
            color="blue",
            alpha=0.6,
            linewidth=2.0)
    if has_bh and bh_equity is not None:
        ax.plot((bh_equity - 1) * 100,
                label=f"{bh_name} (SR: {strats[bh_name]['sharpe']:.2f})",
                color="gray",
                linestyle="--",
                linewidth=1.5)

    # ┏━━━━━━━━━━ Set Title ━━━━━━━━━━┓
    title = (f"[CASH ENGINE] Cumulative Returns (%) — {mlabel} {granularity.upper()}+{direction_label}"
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

    # ┏━━━━━━━━━━ ROI report ━━━━━━━━━━┓
    lines = [
        "=" * 60,
        f"FINANCIAL BACKTEST [CASH-ACCOUNT ENGINE]: {mlabel} {granularity.upper()} "
        f"{direction.upper()} {meta_mode.upper()} ({desc})",
        f"Period: {test_start.date()} to {test_end.date()}",
        f"Threshold: {threshold:.4f} ({constraint_tag}) | Fee: {fee*100:.3f}%",
        f"Sizing: exposure={_EXPOSURE if df_trades['asset'].nunique() > 1 else _EXPOSURE_SINGLE_ASSET:.2f}"
        f" base={_SIZE_BASE:.2f} | M1 flat-sized, M2 confidence-sized",
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
        "=" * 60,
        f"{'Strategy':<25} {'Total Ret':>10} {'MaxDD':>8} {'Sharpe':>8}",
        "-" * 60,
    ]
    for name in [m2_name, m1_name] + ([bh_name] if has_bh else []):
        s = strats[name]
        lines.append(f"{name:<25} {s['total_ret']:>+9.2f}% {s['mdd']:>+7.2f}% {s['sharpe']:>7.2f}")
    lines.append("=" * 60)
    report = "\n".join(lines)
    print(f"\n{report}")
    report_path = save_dir / f"{file_prefix}_ROI.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # ┏━━━━━━━━━━ Trades CSV (+ per-bar capital ledger from capital_log) ━━━━━━━━━━┓
    trades_dump = df_trades.copy()
    trades_dump["date"] = trades_dump["date"].astype(str)
    trades_dump["direction"] = direction
    trades_dump["return_pct"] = trades_dump["return"] * 100
    # Per-trade sizing fraction actually used by the cash engine for APPROVED trades
    # (rejected trades never enter the book → NaN).
    _appr = trades_dump["m2_approved"].values if "m2_approved" in trades_dump.columns else None
    if _appr is not None and "m2_prob" in trades_dump.columns:
        trades_dump["size_frac"] = [
            _z_size(p) if a else np.nan for p, a in zip(trades_dump["m2_prob"].values, _appr)]
    trades_path = save_dir / f"{file_prefix}_trades.csv"
    trades_dump.to_csv(trades_path, index=False, float_format="%.6f")

    # Per-bar capital ledger (free cash / locked / new investment / realized P&L / #open).
    ledger_path = save_dir / f"{file_prefix}_capital_ledger.csv"
    m2_ledger.to_csv(ledger_path, index_label="date", float_format="%.6f")

    print(f"  [cash] Equity curve saved:   {plot_path.name}")
    print(f"  [cash] ROI report saved:     {report_path.name}")
    print(f"  [cash] Trade dump saved:     {trades_path.name}")
    print(f"  [cash] Capital ledger saved: {ledger_path.name}")


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
                         thres_mode  = "utility"):
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
    if model_name == "ctts":
        from Utils.classifier.ctts.ctts_features import extract_close_windows
        _seq = artifacts.get("seq_len", 40)
        eng = extract_close_windows(dataset, seq_len=_seq)
    else:
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
    if model_name == "ctts":
        X_test = eng[idx_test].copy()
    else:
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
                              "m2_prob_raw": probs_raw,
                              "m2_prob": probs})

    # ┏━━━━━━━━━━ Handle NaN Returns ━━━━━━━━━━┓
    n_nan_rets = int(df_trades["return"].isna().sum())
    if n_nan_rets > 0:
        print(f"  [backtest] WARNING: {n_nan_rets} trades have NaN returns (dropped)")
        df_trades = df_trades.dropna(subset=["return"]).reset_index(drop=True)
        m2_approved = df_trades["m2_approved"].values
        probs = df_trades["m2_prob"].values
        probs_raw = df_trades["m2_prob_raw"].values
        net_returns = df_trades["return"].values
        y_test = df_trades["label"].values
    
    # ┏━━━━━━━━━━ Get Test Date Range ━━━━━━━━━━┓
    test_start = df_trades["date"].min()
    test_end   = df_trades["date"].max()

    # ┏━━━━━━━━━━ Get Horizon ━━━━━━━━━━┓
    horizon = int(cfg.get("data", {}).get("load", {}).get("forecast_horizon", 7))
    m2_df = df_trades[df_trades["m2_approved"]]

    # ┏━━━━━━━━━━ Load Raw Close Prices ━━━━━━━━━━┓
    raw_close = _load_raw_close_prices(cfg, granularity, direction=direction)
    has_bh = len(raw_close) > 0

    # ┏━━━━━━━━━━ Filter Raw Close Prices by Test Date Range ━━━━━━━━━━┓
    if has_bh:
        raw_close = raw_close[(raw_close["date"] >= test_start) & (raw_close["date"] <= test_end)]
        bh_pivot = raw_close.pivot_table(index="date", columns="asset", values="close")
        # Per-asset first/last VALID price, not the pivot's global first/last row: an asset
        # delisted mid-window (e.g. TONUSDT stops trading 2026-06-30) has NaN in the final
        # rows, which would make its B&H return NaN and blank out that table cell. Measuring
        # each asset to its own last traded bar keeps the number meaningful and finite.
        bh_first = bh_pivot.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
        bh_last  = bh_pivot.apply(lambda c: c.dropna().iloc[-1] if c.notna().any() else np.nan)
        bh_equity = (bh_pivot / bh_first).mean(axis=1)
        asset_bh_rets = (bh_last / bh_first) - 1
        full_idx = bh_equity.index
    else:
        full_idx = pd.DatetimeIndex(sorted(df_trades["date"].unique()))

    # ┏━━━━━━━━━━ Build Equity Curves ━━━━━━━━━━┓
    m1_equity, _ = _build_spread_equity(df_trades, full_idx, horizon)
    m2_equity, _ = _build_spread_equity(m2_df, full_idx, horizon)

    # ┏━━━━━━━━━━ Calculate Per-Asset Returns ━━━━━━━━━━┓
    asset_m1_rets = df_trades.groupby("asset")["return"].mean()
    asset_m2_rets = m2_df.groupby("asset")["return"].mean() if len(m2_df) > 0 else pd.Series(dtype=float)

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
    m2_name = f"M2 {mlabel} selective"
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
    
    # ┏━━━━━━━━━━ Utility Threshold ━━━━━━━━━━┓
    direction_label = direction.upper()
    constraint_tag = "Utility-Opt" if val_op["constraint_satisfied"] else "fallback"
    fee_tag = f" fee={fee*100:.2f}%" if fee > 0 else ""

    # ┏━━━━━━━━━━ Plot Equity Curves ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(18, 8))
    plt.subplots_adjust(right=0.68)

    ax.plot((m2_equity - 1) * 100,
            label=f"{m2_name} (SR: {strats[m2_name]['sharpe']:.2f}, Exec: {execution_rate:.1f}%)",
            color="green",
            linewidth=3.0)
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

    # ┏━━━━━━━━━━ NEW cash-account engine: parallel artifact set ━━━━━━━━━━┓
    # Same trades, same fee-adjusted returns, same threshold — ONLY the equity math
    # differs, so this is a controlled old-vs-new comparison. Written under a separate
    # file_prefix so the original '10_backtest_all_*' artifacts stay untouched (they are
    # what run_combined_backtest reads).
    _cash_prefix = f"{file_prefix.replace('_all', '')}_cash" if "_all" in file_prefix else f"{file_prefix}_cash"
    _emit_cash_engine_artifacts(
        df_trades=df_trades, m2_df=m2_df, full_idx=full_idx, horizon=horizon,
        save_dir=save_dir, file_prefix=_cash_prefix, cfg=cfg, mlabel=mlabel,
        m2_name=m2_name, m1_name=m1_name, bh_name=bh_name,
        bh_equity=bh_equity if has_bh else None, has_bh=has_bh,
        ann_horizon=ann_horizon, granularity=granularity, direction=direction,
        meta_mode=meta_mode, desc=desc, threshold=threshold,
        constraint_tag=constraint_tag, fee=fee, execution_rate=execution_rate,
        test_start=test_start, test_end=test_end, n_total=n_total,
        n_approved=n_approved, m1_wr=m1_wr, m2_wr=m2_wr, m1_good=m1_good,
        n_m2_good=n_m2_good, avg_app=avg_app, avg_rej=avg_rej,
        asset_m2_rets=asset_m2_rets, asset_m1_rets=asset_m1_rets,
        asset_bh_rets=asset_bh_rets if has_bh else pd.Series(dtype=float),
        close=bh_pivot if has_bh else None,
    )

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
    print(f"[combined_backtest] Merging UP+DOWN trades for: {granularity}")
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
