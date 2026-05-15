"""Financial backtest helpers extracted from m2_pipeline.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ┏━━━━━━━━━━ Utils ━━━━━━━━━━┓
from Utils.utils import m1_display_label as _m1_display_label

# ┏━━━━━━━━━━ Engine ━━━━━━━━━━┓
from Utils.backtest.engine import (_annualization_factor,
                                  _build_spread_equity,
                                  _calc_drawdown,
                                  _calc_sharpe,
                                  _equity_horizon_returns,
                                  _load_raw_close_prices)


__all__ = ["_plot_path_equity"]


# ┏━━━━━━━━━━ Plot path equity ━━━━━━━━━━┓
def _plot_path_equity(path_metrics, dates_by_path, save_path, gran, direction, fee, horizon, gran_name, cfg):
    # ┏━━━━━━━━━━ Initialize variables ━━━━━━━━━━┓
    n_paths = len(path_metrics)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_paths, 2)))
    ann_bar = _annualization_factor(gran_name)
    ann_horizon = np.sqrt(ann_bar ** 2 / max(horizon, 1))

    # ┏━━━━━━━━━━ Create figure ━━━━━━━━━━┓
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(28, 24))

    # ┏━━━━━━━━━━ Helper to get value color ━━━━━━━━━━┓
    def _val_color(val, scale):
        intensity = min(abs(val) / scale, 1.0)
        if val > 0:
            r, g, b = int(255 - intensity * 209), int(255 - intensity * 116), int(255 - intensity * 168)
        elif val < 0:
            r, g, b = int(255 - intensity * 75), int(255 - intensity * 215), int(255 - intensity * 215)
        else:
            return "#ffffff"
        return f"#{r:02x}{g:02x}{b:02x}"

    # ┏━━━━━━━━━━ Initialize financial metrics lists ━━━━━━━━━━┓
    sharpes_sel, total_rets_sel, mdds_sel = [], [], []
    sharpes_pred, total_rets_pred, mdds_pred = [], [], []
    list_m2_sel, list_m2_pred = [], []
    all_dates, all_rets, all_assets = [], [], []

    # ┏━━━━━━━━━━ Iterate over paths ━━━━━━━━━━┓
    for pi, (pm, p_dates) in enumerate(zip(path_metrics, dates_by_path)):
        # ┏━━━━━━━━━━ Append path, dates and metrics to lists ━━━━━━━━━━┓
        all_dates.extend(p_dates)
        all_rets.extend(pm["_rets"] - fee)
        all_assets.extend(pm["_assets"])
        df_p = pd.DataFrame({"date": pd.to_datetime(p_dates), "return": pm["_rets"] - fee, "asset": pm["_assets"]})
        list_m2_sel.append(df_p[pm["_sel"] == 1].copy())
        list_m2_pred.append(df_p[pm["_preds"] == 1].copy())

    # ┏━━━━━━━━━━ Compute M1 metrics ━━━━━━━━━━┓
    df_m1 = pd.DataFrame({"date": pd.to_datetime(all_dates), "return": all_rets, "asset": all_assets})
    m1_tl = pd.DatetimeIndex(sorted(df_m1["date"].unique()))
    m1_eq, _ = _build_spread_equity(df_m1, m1_tl, horizon)
    m1_ret = (m1_eq.iloc[-1] - 1) * 100 if len(m1_eq) > 0 else 0.0
    m1_sr = _calc_sharpe(_equity_horizon_returns(m1_eq, horizon) if len(m1_eq) > horizon else np.array([]), ann_horizon)
    m1_mdd = _calc_drawdown(m1_eq.values) * 100 if len(m1_eq) > 0 else 0.0
    m1_label = f"M1 All Trades (Ret={m1_ret:+.1f}%, SR={m1_sr:.2f})"

    # ┏━━━━━━━━━━ Compute BH metrics ━━━━━━━━━━┓
    raw_close = _load_raw_close_prices(cfg, gran_name, direction=direction)
    has_bh = len(raw_close) > 0
    bh_ret = 0.0
    bh_mdd = 0.0
    asset_bh_rets = pd.Series(dtype=float)

    # ┏━━━━━━━━━━ Compute BH metrics ━━━━━━━━━━┓
    if has_bh:
        # ┏━━━━━━━━━━ Get time range ━━━━━━━━━━┓
        t_start, t_end = m1_tl.min(), m1_tl.max()
        raw_close = raw_close[(raw_close["date"] >= t_start) & (raw_close["date"] <= t_end)]
        bh_pivot = raw_close.pivot_table(index="date", columns="asset", values="close")
        if len(bh_pivot) > 0:
            # ┏━━━━━━━━━━ Normalize each asset from its own first available price (handles late-listed assets) ━━━━━━━━━━┓
            bh_first = bh_pivot.apply(lambda col: col.dropna().iloc[0] if col.notna().any() else np.nan)
            bh_equity = (bh_pivot / bh_first).mean(axis=1)
            asset_bh_rets = (bh_pivot.iloc[-1] / bh_first) - 1
            bh_ret = (bh_equity.iloc[-1] - 1) * 100
            bh_mdd = _calc_drawdown(bh_equity.values) * 100
            for ax in [ax1, ax2]:
                ax.plot((bh_equity - 1) * 100, label=f"B&H (Ret={bh_ret:+.1f}%)", color="gray", ls="--", lw=1.5)

    # ┏━━━━━━━━━━ Compute M2 metrics ━━━━━━━━━━┓
    for pi, df_pred in enumerate(list_m2_pred):
        # ┏━━━━━━━━━━ Build spread equity ━━━━━━━━━━┓
        m2_pred_eq, _ = _build_spread_equity(df_pred, m1_tl, horizon)
        
        # ┏━━━━━━━━━━ Compute metrics ━━━━━━━━━━┓
        pred_ret = (m2_pred_eq.iloc[-1] - 1) * 100 if len(m2_pred_eq) > 0 else 0.0
        pred_sr = _calc_sharpe(_equity_horizon_returns(m2_pred_eq, horizon) if len(m2_pred_eq) > horizon else np.array([]), ann_horizon)
        pred_mdd = _calc_drawdown(m2_pred_eq.values) * 100 if len(m2_pred_eq) > 0 else 0.0
        sharpes_pred.append(pred_sr)
        total_rets_pred.append(pred_ret)
        mdds_pred.append(pred_mdd)
        
        # ┏━━━━━━━━━━ Plot equity curve ━━━━━━━━━━┓
        ax1.plot((m2_pred_eq - 1) * 100, color=colors[pi], lw=2, label=f"Path {pi+1} (Ret={pred_ret:+.1f}%, SR={pred_sr:.2f}, MDD={pred_mdd:.1f}%)")

    # ┏━━━━━━━━━━ Compute Selective M2 metrics ━━━━━━━━━━┓
    for pi, df_sel in enumerate(list_m2_sel):
        # ┏━━━━━━━━━━ Build spread equity ━━━━━━━━━━┓
        m2_sel_eq, _ = _build_spread_equity(df_sel, m1_tl, horizon)
        
        # ┏━━━━━━━━━━ Compute metrics ━━━━━━━━━━┓
        sel_ret = (m2_sel_eq.iloc[-1] - 1) * 100 if len(m2_sel_eq) > 0 else 0.0
        sel_sr = _calc_sharpe(_equity_horizon_returns(m2_sel_eq, horizon) if len(m2_sel_eq) > horizon else np.array([]), ann_horizon)
        sel_mdd = _calc_drawdown(m2_sel_eq.values) * 100 if len(m2_sel_eq) > 0 else 0.0
        sharpes_sel.append(sel_sr)
        total_rets_sel.append(sel_ret)
        mdds_sel.append(sel_mdd)
        
        # ┏━━━━━━━━━━ Plot equity curve ━━━━━━━━━━┓
        ax2.plot((m2_sel_eq - 1) * 100, color=colors[pi], lw=2, label=f"Path {pi+1} (Ret={sel_ret:+.1f}%, SR={sel_sr:.2f}, MDD={sel_mdd:.1f}%)")

    # ┏━━━━━━━━━━ Plot M1 equity curve ━━━━━━━━━━┓
    for ax in [ax1, ax2]:
        # ┏━━━━━━━━━━ Plot M1 equity curve ━━━━━━━━━━┓
        ax.plot((m1_eq - 1) * 100, color="blue", lw=1.5, ls="--", alpha=0.7, label=m1_label)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        
        # ┏━━━━━━━━━━ Set labels and title ━━━━━━━━━━┓
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Cumulative Return (%)", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)

    # ┏━━━━━━━━━━ Set titles ━━━━━━━━━━┓
    ax1.set_title(f"M2 Baseline (@0.5 Threshold) | {gran} {direction.upper()} | fee={fee*100:.2f}%", fontsize=18, fontweight="bold")
    ax2.set_title(f"M2 Selective (Optimized Threshold) | {gran} {direction.upper()} | fee={fee*100:.2f}%", fontsize=18, fontweight="bold")

    # ┏━━━━━━━━━━ Render table ━━━━━━━━━━┓
    def _render_table(ax, df_m1, list_m2, rets_m2, mdds_m2):
        """Render individual asset performance table to the right of ax (backtest style)."""
        # ┏━━━━━━━━━━ Get asset counts and returns ━━━━━━━━━━┓
        m1_counts = df_m1.groupby("asset").size()
        asset_m1_rets = df_m1.groupby("asset")["return"].mean()
        asset_list = sorted(df_m1["asset"].unique())
        m1_model_name = _m1_display_label(cfg)
        col_labels = ["Asset"] + [f"P{i+1}" for i in range(n_paths)] + [m1_model_name, "B&H"]

        # ┏━━━━━━━━━━ Build table data ━━━━━━━━━━┓
        table_data = []

        # ┏━━━━━━━━━━ Loop through assets ━━━━━━━━━━┓
        for asset in asset_list:
            row = [asset]
            # ┏━━━━━━━━━━ Loop through paths ━━━━━━━━━━┓
            for pi in range(n_paths):
                # ┏━━━━━━━━━━ Get path data ━━━━━━━━━━┓
                df_p = list_m2[pi]
                m2_counts = df_p.groupby("asset").size() if len(df_p) > 0 else pd.Series(dtype=int)
                m2_rets = df_p.groupby("asset")["return"].mean() if len(df_p) > 0 else pd.Series(dtype=float)
                m2_a = m2_rets.get(asset, 0.0) * 100
                m2_n = int(m2_counts.get(asset, 0))
                row.append(f"{m2_a:+.1f}% ({m2_n})")
            
            # ┏━━━━━━━━━━ Get M1 data ━━━━━━━━━━┓
            m1_a = asset_m1_rets.get(asset, 0.0) * 100
            m1_n = int(m1_counts.get(asset, 0))
            
            # ┏━━━━━━━━━━ Get B&H data ━━━━━━━━━━┓
            _bh_val = asset_bh_rets.get(asset, np.nan) if has_bh else np.nan
            bh_a = float(_bh_val) * 100 if pd.notna(_bh_val) else None
            
            # ┏━━━━━━━━━━ Append to table data ━━━━━━━━━━┓
            row.append(f"{m1_a:+.1f}% ({m1_n})")
            row.append(f"{bh_a:+.1f}%" if bh_a is not None else "N/A")
            table_data.append(row)

        # ┏━━━━━━━━━━ Build summary rows ━━━━━━━━━━┓
        row_ret = ["Portfolio Return"]
        row_avg = ["Avg Ret/Trade"]
        row_mdd = ["Max Drawdown"]

        # ┏━━━━━━━━━━ Loop through paths ━━━━━━━━━━┓
        for pi in range(n_paths):
            # ┏━━━━━━━━━━ Get M2 data ━━━━━━━━━━┓
            df_p = list_m2[pi]
            
            # ┏━━━━━━━━━━ Append to table data ━━━━━━━━━━┓
            row_ret.append(f"{rets_m2[pi]:+.2f}%")
            avg_m2 = df_p["return"].mean() * 100 if len(df_p) > 0 else 0.0
            row_avg.append(f"{avg_m2:+.2f}% ({len(df_p)})")
            row_mdd.append(f"{mdds_m2[pi]:+.2f}%")

        # ┏━━━━━━━━━━ Get M1 data ━━━━━━━━━━┓
        row_ret.append(f"{m1_ret:+.2f}%")
        avg_m1 = df_m1["return"].mean() * 100 if len(df_m1) > 0 else 0.0
        row_avg.append(f"{avg_m1:+.2f}% ({len(df_m1)})")
        row_mdd.append(f"{m1_mdd:+.2f}%")

        # ┏━━━━━━━━━━ Get B&H data ━━━━━━━━━━┓
        if has_bh:
            row_ret.append(f"{bh_ret:+.2f}%")
            avg_bh = asset_bh_rets.mean() * 100 if len(asset_bh_rets) > 0 else 0.0
            row_avg.append(f"{avg_bh:+.2f}%")
            row_mdd.append(f"{bh_mdd:+.2f}%")
        else:
            row_ret.append("N/A")
            row_avg.append("N/A")
            row_mdd.append("N/A")

        # ┏━━━━━━━━━━ Append to table data ━━━━━━━━━━┓
        table_data.extend([row_ret, row_avg, row_mdd])

        # ┏━━━━━━━━━━ Set table properties ━━━━━━━━━━┓
        n_cols_tbl  = len(col_labels)
        col_asset_w = 0.18
        col_bh_w    = 0.08
        col_m1_w    = 0.15
        remaining   = 1.0 - col_asset_w - col_bh_w - col_m1_w
        col_path_w  = remaining / max(n_paths, 1)
        col_widths  = [col_asset_w] + [col_path_w] * n_paths + [col_m1_w, col_bh_w]

        # ┏━━━━━━━━━━ Create table ━━━━━━━━━━┓
        table_bbox = [1.03, 0.0, 0.82, 1.0]
        the_table = ax.table(cellText=table_data, colLabels=col_labels, loc="right", bbox=table_bbox, cellLoc="center", colWidths=col_widths)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9.5)
        
        # ┏━━━━━━━━━━ Set cell properties ━━━━━━━━━━┓
        for (r, c), cell in the_table.get_celld().items():
            cell.set_height(1.0 / (len(table_data) + 1) * 1.1)

        # ┏━━━━━━━━━━ Get summary statistics ━━━━━━━━━━┓
        n_data_rows = len(table_data)
        n_summary = 3
        first_summary_row = n_data_rows - n_summary + 1

        # ┏━━━━━━━━━━ Get all values ━━━━━━━━━━┓
        all_vals = []
        for (r, c), cell in the_table.get_celld().items():
            if r > 0 and c > 0:
                v_str = cell.get_text().get_text().split("%")[0].strip()
                try: all_vals.append(float(v_str))
                except ValueError: pass
        abs_max = max(abs(v) for v in all_vals) if all_vals else 1.0

        # ┏━━━━━━━━━━ Set cell properties ━━━━━━━━━━┓
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
                except ValueError: pass
            if r == 0 and c >= 0:
                cell.set_facecolor("#2b5797")
                cell.get_text().set_color("white")
            elif r == 0:
                cell.set_facecolor("#f2f2f2")
            if r >= first_summary_row:
                cell.set_fontsize(9)
                if c == 0:
                    cell.set_facecolor("#2b5797")
                    cell.get_text().set_color("white")

        # ┏━━━━━━━━━━ Set title ━━━━━━━━━━┓
        title_x = table_bbox[0] + table_bbox[2] / 2.0 + 0.01
        ax.text(title_x, 1.02, "Individual Asset Performance", transform=ax.transAxes, fontsize=18, fontweight="bold", ha="center")

    # ┏━━━━━━━━━━ Render tables ━━━━━━━━━━┓
    _render_table(ax1, df_m1, list_m2_pred, total_rets_pred, mdds_pred)
    _render_table(ax2, df_m1, list_m2_sel, total_rets_sel, mdds_sel)

    # ┏━━━━━━━━━━ Adjust layout and save ━━━━━━━━━━┓
    fig.subplots_adjust(hspace=0.20, right=0.55, top=0.92)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return sharpes_sel, total_rets_sel
