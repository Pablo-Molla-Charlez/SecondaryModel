"""Calibration-study plotting functions. Pure plotting; no data loading."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants & display names
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Coarsest → finest (this is the canonical x-axis order across every plot).
_GRAN_ORDER = ["1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m"]
_SPLIT_ORDER = ["train", "val_cal", "val_opt", "test"]
_SPLIT_TITLES = {"train": "Train", "val_cal": "Val-Cal", "val_opt": "Val-Opt", "test": "Test"}

_STAGE_ORDER  = ["raw_tau05", "cal_tau05", "raw_opt", "cal_opt"]
_STAGE_LABELS = {"raw_tau05": "No Cal @0.5",
                 "cal_tau05": "Cal @0.5",
                 "raw_opt":   "No Cal @τ*",
                 "cal_opt":   "Cal @τ*"}
_STAGE_COLORS = {"raw_tau05": "#c8c8c8",
                 "cal_tau05": "#6aa5d8",
                 "raw_opt":   "#f4a261",
                 "cal_opt":   "#2a9d8f"}

_TRIGGER_COLORS = {"trig_unique": "#e76f51",
                   "trig_range":  "#f4a261",
                   "trig_squash": "#9d4edd"}
_TRIGGER_LABELS = {"trig_unique": "(a) < 5 distinct outputs",
                   "trig_range":  "(b) output range < 0.10",
                   "trig_squash": "(c) positive-mass squash"}

_SOURCE_ORDER  = ["Utility-Opt", "Precision-Coverage", "Risk-Fallback", "Baseline", "Baseline-Override"]
_SOURCE_COLORS = {"Utility-Opt":        "#2a9d8f",
                  "Precision-Coverage": "#6aa5d8",
                  "Risk-Fallback":      "#f4a261",
                  "Baseline":           "#e76f51",
                  "Baseline-Override":  "#9d4edd"}

_M2_DISPLAY = {"rf":        "Random Forest",
               "tabpfn":    "TabPFN",
               "tabicl":    "TabICL",
               "autogluon": "AutoGluon",
               "xgboost":   "XGBoost"}

_METRIC_TITLES = {"accuracy":      "Accuracy",
                  "winrate":       "Winrate",
                  "coverage":      "Coverage",
                  "sel_accuracy":  "Selective Accuracy",
                  "sel_winrate":   "Selective Winrate",
                  "sel_coverage":  "Selective Coverage",
                  "mean_net_ret":  "Mean Net Return",
                  "mean_win_ret":  "Mean Win Return",
                  "mean_loss_ret": "Mean Loss Return"}

_XLABEL_ROT = 25


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _save(fig, path: Path, dpi: int = 200, pad_inches: float = 0.08) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)


def _m1_title_suffix(df: pd.DataFrame) -> str:
    """' — M1 = Kronos' if only a single M1 model is present, else '' (use full df)."""
    if "m1_model" in df.columns:
        m1s = df["m1_model"].dropna().unique().tolist()
        if len(m1s) == 1:
            return f" — M1 = {m1s[0]}"
    return ""


def _single_m1(df: pd.DataFrame) -> bool:
    return "m1_model" in df.columns and df["m1_model"].dropna().nunique() == 1


def _display_m2(m2: str) -> str:
    return _M2_DISPLAY.get(m2, m2)


def _sorted_grans(df: pd.DataFrame) -> List[str]:
    present = df["granularity"].dropna().unique().tolist()
    return [g for g in _GRAN_ORDER if g in present]


def _config_key_frame(df: pd.DataFrame,
                      include_m1: bool,
                      include_m2: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Build a canonical (and consistently sorted) config axis + readable labels.

    Sort priority: m2 → direction (UP before DOWN) → granularity (coarse→fine) → m1.
    Columns in the returned frame follow the same ``include_m1`` / ``include_m2`` toggles,
    so callers can merge on them without pulling in unwanted grouping variables.
    """
    cols = []
    if include_m1 and "m1_model" in df.columns: cols.append("m1_model")
    if include_m2 and "m2_model" in df.columns: cols.append("m2_model")
    for c in ("direction", "granularity"):
        if c in df.columns: cols.append(c)
    have = cols
    keys = df[have].drop_duplicates().copy()
    # enforce gran ordering
    drop_cols = []
    sort_by = []
    if "m2_model" in have: sort_by.append("m2_model")
    if "direction" in have:
        keys["_dir_rank"]  = keys["direction"].map({"UP": 0, "DOWN": 1})
        sort_by.append("_dir_rank"); drop_cols.append("_dir_rank")
    if "granularity" in have:
        keys["_gran_rank"] = keys["granularity"].map({g: i for i, g in enumerate(_GRAN_ORDER)})
        sort_by.append("_gran_rank"); drop_cols.append("_gran_rank")
    if "m1_model" in have: sort_by.append("m1_model")
    if sort_by:
        keys = keys.sort_values(sort_by)
    if drop_cols:
        keys = keys.drop(columns=drop_cols)
    keys = keys.reset_index(drop=True)
    labels = []
    for r in keys.itertuples(index=False):
        d = r._asdict() if hasattr(r, "_asdict") else dict(zip(keys.columns, r))
        parts = []
        if include_m2 and "m2_model" in d: parts.append(_display_m2(d["m2_model"]))
        if include_m1 and "m1_model" in d: parts.append(d["m1_model"])
        if "direction" in d:   parts.append(d["direction"])
        if "granularity" in d: parts.append(d["granularity"])
        labels.append(" / ".join(parts))
    return keys, labels


def _iqr_stats(vals: np.ndarray) -> dict:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan, "n": 0}
    return {"mean":   float(np.mean(vals)),
            "std":    float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "median": float(np.median(vals)),
            "q25":    float(np.percentile(vals, 25)),
            "q75":    float(np.percentile(vals, 75)),
            "n":      int(vals.size)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (1) Per-split TP-rate heatmaps
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_split_tp_rate_heatmaps(df: pd.DataFrame, save_dir: Path) -> Optional[Path]:
    """4-panel heatmap (Train / Val-Cal / Val-Opt / Test). Data-driven colour range."""
    if df is None or df.empty:
        return None
    grans = _sorted_grans(df)
    df = df[df["granularity"].isin(grans)].copy()
    single_m1 = _single_m1(df)
    if single_m1:
        df["row_label"] = df["direction"]
    else:
        df["row_label"] = df["m1_model"] + " / " + df["direction"]
    # Row ordering: for multi-M1 alphabetical by m1, within each UP first
    if single_m1:
        row_order = [r for r in ["UP", "DOWN"] if r in df["row_label"].unique().tolist()]
    else:
        row_order = sorted(df["row_label"].unique().tolist(),
                           key=lambda s: (s.split(" / ")[0], 0 if s.endswith("UP") else 1))

    # Data-driven colour range
    vmin = float(np.nanmin(df["tp_rate"]))
    vmax = float(np.nanmax(df["tp_rate"]))
    if vmax - vmin < 1e-9:
        vmin, vmax = vmin - 0.01, vmax + 0.01

    n_rows = len(row_order)
    cell_w = 0.55
    cell_h = 0.38
    fig_w = 4 * (len(grans) * cell_w + 0.6) + 1.1     # +1.1 for colorbar column
    fig_h = max(2.2, n_rows * cell_h + 1.6)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.04], wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cax  = fig.add_subplot(gs[0, 4])

    im = None
    for ax, split in zip(axes, _SPLIT_ORDER):
        sub = df[df["split"] == split]
        mat = (sub.pivot_table(index="row_label", columns="granularity",
                               values="tp_rate", aggfunc="mean")
                  .reindex(index=row_order, columns=grans))
        im = ax.imshow(mat.values, aspect="auto", cmap="RdYlGn",
                       vmin=vmin, vmax=vmax, origin="upper")
        ax.set_xticks(range(len(grans))); ax.set_xticklabels(grans, rotation=0, fontsize=9)
        ax.set_yticks(range(len(row_order)))
        if ax is axes[0]:
            ax.set_yticklabels(row_order, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_title(_SPLIT_TITLES[split], fontsize=11)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isnan(v):
                    continue
                # Use luminance heuristic vs the chosen vmin/vmax range
                norm = (v - vmin) / max(vmax - vmin, 1e-9)
                txt_color = "black" if 0.25 <= norm <= 0.75 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=txt_color)
    fig.colorbar(im, cax=cax, label="TP rate")

    title = "True-positive (base-rate) distribution per embargoed split" + _m1_title_suffix(df)
    # Generous top margin so the suptitle never overlaps subplot titles
    fig.suptitle(title, fontsize=13, y=1.04)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.93)

    out = Path(save_dir) / "calibration_split_tp_rate.png"
    _save(fig, out)
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (2) Degeneracy triggers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_degeneracy_triggers(df: pd.DataFrame, save_dir: Path):
    """Per-m2 0/1-trigger bars (one subplot per m2) + aggregate count/fraction bars."""
    if df is None or df.empty:
        return None, None
    grans = _sorted_grans(df)
    df = df[df["granularity"].isin(grans)].copy()
    single_m1 = _single_m1(df)
    m2_models = sorted(df["m2_model"].unique().tolist(),
                       key=lambda m: list(_M2_DISPLAY.keys()).index(m)
                                      if m in _M2_DISPLAY else 99)

    x_keys, x_labels = _config_key_frame(df, include_m1=not single_m1, include_m2=False)
    x_pos = np.arange(len(x_labels))
    width = 0.25

    # ── Per-m2 subplots ──
    n_m2 = len(m2_models)
    fig_h = 2.4 * n_m2 + 1.6
    fig_w = max(11, 0.55 * len(x_labels) + 3.0)
    fig, axes = plt.subplots(n_m2, 1, figsize=(fig_w, fig_h),
                             squeeze=False, sharey=True)
    for ax, m2 in zip(axes[:, 0], m2_models):
        sub = df[df["m2_model"] == m2]
        merged = x_keys.merge(sub, on=list(x_keys.columns), how="left")
        for k, col in enumerate(("trig_unique", "trig_range", "trig_squash")):
            vals = merged[col].fillna(False).infer_objects(copy=False).astype(float).values
            ax.bar(x_pos + (k - 1) * width, vals, width=width,
                   color=_TRIGGER_COLORS[col], label=_TRIGGER_LABELS[col])
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(_display_m2(m2), fontsize=10)
        ax.set_ylabel("Trigger (0/1)")
        ax.grid(axis="y", alpha=0.3)
        # Repeat x labels on every subplot
        ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, rotation=_XLABEL_ROT,
                                                  fontsize=7, ha="right")
    # Vertical legend anchored to the top-right of the first subplot
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.002, 1.0),
                      fontsize=8, frameon=True, ncol=1, borderaxespad=0.0)
    title = "Isotonic-regression degeneracy triggers on Val-Cal" + _m1_title_suffix(df)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.86, 0.97])
    out1 = Path(save_dir) / "calibration_degeneracy_triggers.png"
    _save(fig, out1)

    # ── Aggregate: counts & fractions across m2 models ──
    # For each (m1, direction, gran), count how many m2 triggered each check,
    # and what fraction of available m2 that represents.
    n_m2_per_cfg = (df.groupby(list(x_keys.columns))["m2_model"].nunique()
                     .rename("n_m2").reset_index())
    counts = (df.groupby(list(x_keys.columns))[["trig_unique", "trig_range", "trig_squash"]]
                .sum().reset_index())
    agg = x_keys.merge(counts, on=list(x_keys.columns), how="left").merge(n_m2_per_cfg, how="left")
    agg[["trig_unique", "trig_range", "trig_squash"]] = \
        agg[["trig_unique", "trig_range", "trig_squash"]].fillna(0).astype(int)
    agg["n_m2"] = agg["n_m2"].fillna(1).astype(int)

    fig, ax = plt.subplots(figsize=(max(10, 0.32 * len(x_labels) + 2.5), 4.6))
    for k, col in enumerate(("trig_unique", "trig_range", "trig_squash")):
        frac = agg[col] / agg["n_m2"].replace(0, 1)
        bars = ax.bar(x_pos + (k - 1) * width, frac.values, width=width,
                       color=_TRIGGER_COLORS[col], label=_TRIGGER_LABELS[col])
        # Annotate with "count/total" + percentage
        for bar, cnt, tot in zip(bars, agg[col].values, agg["n_m2"].values):
            if cnt == 0:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{int(cnt)}/{int(tot)}",
                    ha="center", va="bottom", fontsize=6.5, rotation=0)
    # Headline totals in the legend title
    tot_u = int(agg["trig_unique"].sum())
    tot_r = int(agg["trig_range"].sum())
    tot_s = int(agg["trig_squash"].sum())
    n_total = int(agg["n_m2"].sum())
    ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, rotation=_XLABEL_ROT,
                                              fontsize=7, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Fraction of m2 models triggering")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title=f"Totals across all configs (n_total = {n_total})\n"
                    f"(a) n = {tot_u}   (b) n = {tot_r}   (c) n = {tot_s}",
              loc="upper left", bbox_to_anchor=(1.002, 1.0), fontsize=8, title_fontsize=8)
    title = "Aggregate isotonic-regression degeneracy triggers on Val-Cal" + _m1_title_suffix(df)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    out2 = Path(save_dir) / "calibration_degeneracy_triggers_aggregate.png"
    _save(fig, out2)
    return out1, out2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (3) / (4) Val-Opt 4-stage comparisons
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CLF_METRICS_GRID = [("accuracy",     "sel_accuracy"),
                     ("winrate",      "sel_winrate"),
                     ("coverage",     "sel_coverage")]
_FIN_METRICS      = ["mean_net_ret", "mean_win_ret", "mean_loss_ret"]


def _configs_sort(df: pd.DataFrame, include_m1: bool):
    keys, labels = _config_key_frame(df, include_m1=include_m1, include_m2=True)
    return keys, labels


def _row_dict(row):
    return row._asdict() if hasattr(row, "_asdict") else dict(row)


def _filter_by_row(df: pd.DataFrame, row) -> pd.DataFrame:
    d = _row_dict(row)
    mask = pd.Series(True, index=df.index)
    for key in ("m1_model", "m2_model", "direction", "granularity"):
        if key in d:
            mask &= (df[key] == d[key])
    return df[mask]


def _stage_value(df: pd.DataFrame, row, stage: str, metric: str) -> float:
    sel = _filter_by_row(df, row)
    sel = sel[sel["stage"] == stage]
    if sel.empty:
        return np.nan
    return float(sel[metric].iloc[0])


def _stage_m1_ref(df: pd.DataFrame, row, col: str) -> float:
    sel = _filter_by_row(df, row)
    if sel.empty:
        return np.nan
    return float(sel[col].iloc[0])


def plot_val_opt_classification(df: pd.DataFrame, save_dir: Path) -> tuple:
    """Per-config 3x2 grid + aggregate with median/IQR + mean/std overlay."""
    if df is None or df.empty:
        return None, None
    single_m1 = _single_m1(df)
    stages = [s for s in _STAGE_ORDER if s in df["stage"].unique().tolist()]
    x_keys, x_labels = _configs_sort(df, include_m1=not single_m1)
    x_pos = np.arange(len(x_labels))
    width = 0.8 / max(len(stages), 1)

    # ── Per-config 3×2 grid ──
    fig, axes = plt.subplots(3, 2, figsize=(max(12, 0.32 * len(x_labels) + 3), 9), squeeze=False)
    for r, (col_l, col_r) in enumerate(_CLF_METRICS_GRID):
        for c, metric in enumerate((col_l, col_r)):
            ax = axes[r, c]
            # Choose baseline column
            if "accuracy" in metric or metric == "coverage":
                base_col = "m1_acc"
            else:
                base_col = "m1_prec"
            has_baseline = metric != "coverage" and metric != "sel_coverage"

            for k, st in enumerate(stages):
                vals, below = [], []
                for row in x_keys.itertuples(index=False):
                    v = _stage_value(df, row, st, metric)
                    vals.append(v)
                    if has_baseline:
                        b = _stage_m1_ref(df, row, base_col)
                        below.append(not (np.isnan(v) or np.isnan(b) or v >= b))
                    else:
                        below.append(False)
                vals = np.asarray(vals, dtype=float)
                # Draw two-pass: below-baseline dimmed, above full
                base_color = _STAGE_COLORS.get(st, "#888888")
                for xi, (vv, is_below) in enumerate(zip(vals, below)):
                    if np.isnan(vv):
                        continue
                    alpha = 0.30 if is_below else 1.0
                    ax.bar(x_pos[xi] + (k - (len(stages) - 1) / 2) * width,
                           vv, width=width, color=base_color, alpha=alpha,
                           label=_STAGE_LABELS[st] if xi == 0 else None)

            if has_baseline:
                yb = [_stage_m1_ref(df, r, base_col) for r in x_keys.itertuples(index=False)]
                ax.plot(x_pos, yb, "k--", linewidth=1.0, alpha=0.75,
                        label=f"M1 {'precision' if base_col == 'm1_prec' else 'accuracy'}")
            ax.set_title(_METRIC_TITLES.get(metric, metric), fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, rotation=_XLABEL_ROT,
                                                      fontsize=6.5, ha="center")
            if r == 0 and c == 1:
                # Single legend at top-right of first row
                ax.legend(loc="upper left", bbox_to_anchor=(1.002, 1.0),
                          fontsize=7.5, frameon=True)

    title = ("Val-Opt Classification Metrics — 4 stages (bars dimmed when ≤ M1 baseline)"
             + _m1_title_suffix(df))
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.9, 0.97])
    per_cfg = Path(save_dir) / "calibration_val_opt_gain_classification.png"
    _save(fig, per_cfg)

    # ── Aggregate: box-like with median bar + IQR + mean marker ──
    agg = Path(save_dir) / "calibration_val_opt_gain_classification_aggregate.png"
    _plot_stage_aggregate_boxstyle(df, ["accuracy", "winrate", "coverage"],
                                    title="Val-Opt Classification Metrics — Aggregate Across M2 Models" + _m1_title_suffix(df),
                                    out_path=agg, is_classification=True)
    return per_cfg, agg


def _plot_stage_aggregate_boxstyle(df: pd.DataFrame,
                                    metrics: list,
                                    title: str,
                                    out_path: Path,
                                    is_classification: bool = False):
    """Aggregate: bar height = median, IQR error bars, mean markers, clear annotation.

    Grid is (number of M2 models) rows × (number of metrics) columns.
    A separate numeric table sits below each bar.
    Layout uses a generous top margin and per-subplot y-range padding.
    """
    stages = [s for s in _STAGE_ORDER if s in df["stage"].unique().tolist()]
    m2_models = sorted(df["m2_model"].unique().tolist(),
                       key=lambda m: list(_M2_DISPLAY.keys()).index(m)
                                      if m in _M2_DISPLAY else 99)
    n_metrics = len(metrics)
    n_m2 = len(m2_models)
    
    fig, axes = plt.subplots(n_m2, n_metrics,
                             figsize=(5.0 * n_metrics, 4.6 * n_m2),
                             squeeze=False)
    
    for r, m2 in enumerate(m2_models):
        df_m2 = df[df["m2_model"] == m2]
        row_valid_vals = []
        for c, metric_base in enumerate(metrics):
            ax = axes[r, c]
            
            stats = []
            for s in stages:
                col_name = metric_base
                vals = df_m2[df_m2["stage"] == s][col_name].values
                stats.append(_iqr_stats(vals))
                
            meds = [s["median"] for s in stats]
            q25s = [s["q25"]    for s in stats]
            q75s = [s["q75"]    for s in stats]
            mns  = [s["mean"]   for s in stats]
            stds = [s["std"]    for s in stats]

            xpos = np.arange(len(stages))
            colors = [_STAGE_COLORS[s] for s in stages]
            ax.bar(xpos, meds, color=colors, edgecolor="black", linewidth=0.5, zorder=2)

            # IQR whiskers
            for i, (m, lo, hi) in enumerate(zip(meds, q25s, q75s)):
                ax.plot([i, i], [lo, hi], color="black", linewidth=1.3, zorder=3)
                ax.plot([i - 0.14, i + 0.14], [lo, lo], color="black", linewidth=1.0, zorder=3)
                ax.plot([i - 0.14, i + 0.14], [hi, hi], color="black", linewidth=1.0, zorder=3)

            # Mean marker
            ax.scatter(xpos, mns, marker="D", s=34, color="white",
                       edgecolor="black", linewidth=0.9, zorder=4)

            # Determine y-range so the annotation fits comfortably BELOW each bar
            valid_vals = np.array([v for v in (list(meds) + list(q25s) + list(q75s) + list(mns))
                                    if not np.isnan(v)])
            row_valid_vals.append(valid_vals)
            if valid_vals.size:
                y_lo = float(np.min(valid_vals))
                y_hi = float(np.max(valid_vals))
                span = max(y_hi - y_lo, 1e-6)
                # Normal padding
                ax.set_ylim(y_lo - span * 0.10, y_hi + span * 0.30)

            # Annotation: a table of numbers anchored below the axes, aligned under each bar.
            # Plotted in axes-fraction coordinates so it never crashes into the bars or title.
            y_txt = -0.28
            for i, (m, lo, hi, mu, sd) in enumerate(zip(meds, q25s, q75s, mns, stds)):
                if np.isnan(m):
                    continue
                frac_x = (i + 0.5) / len(stages)
                ax.text(frac_x, y_txt,
                        f"med {m:.3f}\nQ25 {lo:.3f}\nQ75 {hi:.3f}\nμ {mu:.3f}±{sd:.3f}",
                        ha="center", va="top", fontsize=7,
                        transform=ax.transAxes,
                        bbox=dict(facecolor="white", edgecolor="#bbbbbb",
                                  alpha=0.95, pad=2.0, linewidth=0.5))

            ax.set_xticks(xpos)
            ax.set_xticklabels([_STAGE_LABELS[s] for s in stages],
                               rotation=_XLABEL_ROT, ha="center", fontsize=8.5)
            
            # Subplot title indicating both M2 model and Metric
            m2_name = _display_m2(m2)
            met_name = _METRIC_TITLES.get(metric_base, metric_base)
            ax.set_title(f"{m2_name} — {met_name}", fontsize=11, pad=8)
            ax.grid(axis="y", alpha=0.3)

        # ┏━━━━━━━━━━ Synchronize Y-axis limits for Accuracy and Winrate ━━━━━━━━━━┓
        if is_classification and len(row_valid_vals) >= 2:
            v0 = row_valid_vals[0]
            v1 = row_valid_vals[1]
            v_combo = np.concatenate([v0, v1]) if (v0.size and v1.size) else (v0 if v0.size else v1)
            if v_combo.size:
                y_lo = float(np.min(v_combo))
                y_hi = float(np.max(v_combo))
                span = max(y_hi - y_lo, 1e-6)
                shared_ylim = (y_lo - span * 0.10, y_hi + span * 0.30)
                axes[r, 0].set_ylim(*shared_ylim)
                axes[r, 1].set_ylim(*shared_ylim)

    # Single legend below the figure, outside all axes
    handles = [plt.Rectangle((0, 0), 1, 1, color=_STAGE_COLORS[s]) for s in stages]
    labels  = [_STAGE_LABELS[s] for s in stages]
    handles.append(plt.Line2D([0], [0], marker="D", color="w",
                              markerfacecolor="white", markeredgecolor="black",
                              markersize=8, linewidth=0))
    labels.append("mean")
    handles.append(plt.Line2D([0], [0], color="black", linewidth=1.3))
    labels.append("IQR (Q25 - Q75)")
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, -0.01), ncol=len(labels),
               fontsize=9, frameon=True)
    fig.suptitle(title, fontsize=13, y=0.995)
    
    # Reserve extra room bottom for the annotations and legend, and hspace for row separation
    top_margin = 0.98 - (0.015 * n_m2)
    fig.subplots_adjust(top=top_margin, bottom=0.10, left=0.06, right=0.98,
                        hspace=1.1, wspace=0.30)
    _save(fig, out_path, pad_inches=0.2)


def plot_val_opt_financial(df: pd.DataFrame, save_dir: Path) -> tuple:
    """Financial metrics: per-config 2×2 (3 metrics + a winners-count panel) + aggregate."""
    if df is None or df.empty:
        return None, None
    single_m1 = _single_m1(df)
    stages = [s for s in _STAGE_ORDER if s in df["stage"].unique().tolist()]
    x_keys, x_labels = _configs_sort(df, include_m1=not single_m1)
    x_pos = np.arange(len(x_labels))
    width = 0.8 / max(len(stages), 1)

    # ── Per-config 2×2 grid (3 metrics + winners count panel) ──
    fig, axes = plt.subplots(2, 2, figsize=(max(12, 0.32 * len(x_labels) + 3), 8), squeeze=False)
    axes_flat = axes.ravel()
    for idx, metric in enumerate(_FIN_METRICS):
        ax = axes_flat[idx]
        for k, st in enumerate(stages):
            vals = [_stage_value(df, row, st, metric) for row in x_keys.itertuples(index=False)]
            ax.bar(x_pos + (k - (len(stages) - 1) / 2) * width, vals, width=width,
                   color=_STAGE_COLORS.get(st, "#888888"),
                   label=_STAGE_LABELS[st] if idx == 0 else None)
        ax.set_title(_METRIC_TITLES[metric], fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, rotation=_XLABEL_ROT,
                                                  fontsize=6.5, ha="right")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    if len(stages) > 0:
        axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.002, 1.0),
                          fontsize=7.5, frameon=True)

    # 4th panel: winners-count bars for each financial metric, with ties shared
    ax_w = axes_flat[3]
    winners_by_metric: dict = {}
    for metric in _FIN_METRICS:
        counts = {s: 0 for s in stages}
        ties = 0
        for row in x_keys.itertuples(index=False):
            stage_vals = {s: _stage_value(df, row, s, metric) for s in stages}
            finite = {s: v for s, v in stage_vals.items() if not np.isnan(v)}
            if not finite:
                continue
            # For Mean Loss Return: higher (less negative) is better — max works for all 3
            best_v = max(finite.values())
            winners = [s for s, v in finite.items() if v >= best_v - 1e-12]
            if len(winners) > 1:
                ties += 1
            for s in winners:
                counts[s] += 1
        winners_by_metric[metric] = (counts, ties)

    xw = np.arange(len(stages))
    bw = 0.25
    for mi, metric in enumerate(_FIN_METRICS):
        counts, _ = winners_by_metric[metric]
        vals = [counts[s] for s in stages]
        bars = ax_w.bar(xw + (mi - 1) * bw, vals, width=bw,
                         label=_METRIC_TITLES[metric])
        for bar, v in zip(bars, vals):
            if v == 0:
                continue
            ax_w.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                      f"{v}", ha="center", va="bottom", fontsize=7)
    # Head-to-head highlight: No-Cal τ* vs Cal τ*
    if "raw_opt" in stages and "cal_opt" in stages:
        h2h_lines = []
        for metric in _FIN_METRICS:
            counts, _ = winners_by_metric[metric]
            h2h_lines.append(f"{_METRIC_TITLES[metric]}:  "
                              f"No Cal τ* {counts['raw_opt']}   vs   Cal τ* {counts['cal_opt']}")
        ax_w.text(1.02, 1.0, "Head-to-head\n" + "\n".join(h2h_lines),
                  transform=ax_w.transAxes, ha="left", va="top", fontsize=8,
                  bbox=dict(facecolor="#fff7e6", edgecolor="#f4a261", alpha=0.9))
    ax_w.set_xticks(xw); ax_w.set_xticklabels([_STAGE_LABELS[s] for s in stages],
                                                rotation=_XLABEL_ROT, ha="center", fontsize=8)
    ax_w.set_ylabel("# configs where stage is best")
    ax_w.set_title("Winners count per financial metric (ties count for all winners)", fontsize=10)
    ax_w.grid(axis="y", alpha=0.3)
    ax_w.legend(loc="upper left", fontsize=7.5)

    title = "Val-Opt Financial Metrics — 4 stages x (M2 / Direction / Granularity)" + _m1_title_suffix(df)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.9, 0.97])
    per_cfg = Path(save_dir) / "calibration_val_opt_gain_financial.png"
    _save(fig, per_cfg)

    # ── Aggregate with median/IQR/mean ──
    agg = Path(save_dir) / "calibration_val_opt_gain_financial_aggregate.png"
    _plot_stage_aggregate_boxstyle(df, ["mean_net_ret", "mean_win_ret", "mean_loss_ret"],
                                    title="Val-Opt Financial Metrics — Aggregate Across M2 Models" + _m1_title_suffix(df),
                                    out_path=agg, is_classification=False)
    return per_cfg, agg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (5) Threshold-optimization success/failure — unified single figure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_threshold_optimization_stats(df: pd.DataFrame, save_dir: Path):
    """Single figure: per-m2 stacked bars (one per variant) + overall aggregate bars.

    Each stacked segment is annotated with count and percentage.
    """
    if df is None or df.empty:
        return None, None
    m2_models = sorted(df["m2_model"].unique().tolist(),
                       key=lambda m: list(_M2_DISPLAY.keys()).index(m)
                                      if m in _M2_DISPLAY else 99)
    variants = ["raw", "cal"]

    # Layout: 2 columns = variants; rows: per-m2 top row + aggregate bottom row,
    # rendered as one gridspec figure for compactness.
    fig = plt.figure(figsize=(3.2 * len(m2_models) + 3.0, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], hspace=0.45, wspace=0.18)

    # ── Top row: per-m2 stacked bars, one axis per variant ──
    for ci, variant in enumerate(variants):
        ax = fig.add_subplot(gs[0, ci])
        sub = df[df["probs_variant"] == variant]
        counts = (sub.groupby(["m2_model", "threshold_source"]).size()
                     .unstack(fill_value=0)
                     .reindex(index=m2_models, columns=_SOURCE_ORDER, fill_value=0))
        totals = counts.sum(axis=1).replace(0, 1)
        frac = counts.div(totals, axis=0)
        x_labels_disp = [_display_m2(m) for m in m2_models]
        xpos = np.arange(len(m2_models))
        bottom = np.zeros(len(m2_models))
        for src in _SOURCE_ORDER:
            vals = frac[src].values
            if vals.sum() == 0:
                continue
            bars = ax.bar(xpos, vals, bottom=bottom, color=_SOURCE_COLORS[src],
                          label=src, edgecolor="white", linewidth=0.3)
            # Annotate each visible segment
            for j, (bar, v) in enumerate(zip(bars, vals)):
                if v < 0.03:
                    continue
                cnt = int(counts.iloc[j][src])
                tot = int(totals.iloc[j])
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottom[j] + v / 2,
                        f"{cnt}/{tot}\n({100*v:.0f}%)",
                        ha="center", va="center", fontsize=7, color="black")
            bottom += vals
        ax.set_title(f"probs = {variant}", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fraction of configs")
        ax.set_xticks(xpos); ax.set_xticklabels(x_labels_disp, rotation=0, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # ── Bottom row: aggregated stacked bar across all m2 + all configs, per variant ──
    ax_agg = fig.add_subplot(gs[1, :])
    xw = np.arange(len(variants))
    bar_w = 0.4
    legend_rendered = False
    for vi, variant in enumerate(variants):
        sub = df[df["probs_variant"] == variant]
        counts = sub["threshold_source"].value_counts().reindex(_SOURCE_ORDER, fill_value=0)
        total = int(counts.sum()) or 1
        frac = counts / total
        bottom = 0.0
        for src in _SOURCE_ORDER:
            v = float(frac.get(src, 0.0))
            if v == 0:
                continue
            ax_agg.bar(xw[vi], v, bar_w, bottom=bottom, color=_SOURCE_COLORS[src],
                       label=src if not legend_rendered else None,
                       edgecolor="white", linewidth=0.3)
            if v >= 0.02:
                cnt = int(counts.get(src, 0))
                ax_agg.text(xw[vi], bottom + v / 2,
                            f"{cnt}/{total}  ({100*v:.1f}%)",
                            ha="center", va="center", fontsize=8, color="black")
            bottom += v
        legend_rendered = True
        # Success headline
        n_succ = int(counts.get("Utility-Opt", 0))
        ax_agg.text(xw[vi], 1.02, f"Utility-Opt: {n_succ}/{total}  ({100*n_succ/total:.1f}%)",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_agg.set_ylim(0, 1.12)
    ax_agg.set_xticks(xw)
    ax_agg.set_xticklabels([f"probs = {v}" for v in variants], fontsize=10)
    ax_agg.set_ylabel("Fraction of configs")
    ax_agg.set_title("Aggregate across all (m1, m2, direction, granularity)", fontsize=11)
    ax_agg.grid(axis="y", alpha=0.3)
    ax_agg.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="Threshold source")

    title = "Threshold-optimization outcome across stages" + _m1_title_suffix(df)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])

    # Save both names for backwards compatibility, though content is now unified
    out_main = Path(save_dir) / "calibration_threshold_opt_stats.png"
    _save(fig, out_main)
    # Keep the "aggregate" filename too but pointing at the same figure content
    out_agg = Path(save_dir) / "calibration_threshold_opt_stats_aggregate.png"
    # Re-save the same figure content (so both expected paths exist); regenerate because
    # matplotlib closed the first fig above.
    # Simplest: write a single-variant summary as the "aggregate" file.
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.2))
    xw2 = np.arange(len(variants))
    legend_rendered = False
    for vi, variant in enumerate(variants):
        sub = df[df["probs_variant"] == variant]
        counts = sub["threshold_source"].value_counts().reindex(_SOURCE_ORDER, fill_value=0)
        total = int(counts.sum()) or 1
        bottom = 0.0
        for src in _SOURCE_ORDER:
            v = float(counts.get(src, 0)) / total
            if v == 0:
                continue
            ax2.bar(xw2[vi], v, 0.45, bottom=bottom, color=_SOURCE_COLORS[src],
                    label=src if not legend_rendered else None,
                    edgecolor="white", linewidth=0.3)
            if v >= 0.02:
                cnt = int(counts.get(src, 0))
                ax2.text(xw2[vi], bottom + v / 2,
                         f"{cnt}/{total}  ({100*v:.1f}%)",
                         ha="center", va="center", fontsize=8, color="black")
            bottom += v
        legend_rendered = True
    ax2.set_xticks(xw2); ax2.set_xticklabels([f"probs = {v}" for v in variants])
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Fraction of configs")
    ax2.set_title("Overall threshold-optimization outcome" + _m1_title_suffix(df), fontsize=11)
    ax2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="Threshold source")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    _save(fig2, out_agg)
    return out_main, out_agg


__all__ = ["plot_split_tp_rate_heatmaps",
           "plot_degeneracy_triggers",
           "plot_val_opt_classification",
           "plot_val_opt_financial",
           "plot_threshold_optimization_stats"]
