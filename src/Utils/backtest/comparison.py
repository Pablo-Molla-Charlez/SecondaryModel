"""Comparison-table builders for per-gran, unified, and paradigm analyses."""

import json
import csv as csv_mod
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from Utils.utils import model_label as _model_label
from Utils.data import GRAN_ORDER

__all__ = ["GRAN_ORDER", "run_comparison", "run_paradigm_comparison"]


# ┏━━━━━━━━━━ Run Comparison ━━━━━━━━━━┓
def run_comparison(per_gran_dir: Path, unified_dir: Path, output_dir: Path = None):
    """Build stacked comparison tables for per-gran vs unified results."""
    if output_dir is None:
        output_dir = unified_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    unified_path = unified_dir / "unified_summary.json"
    if not unified_path.exists():
        print(f"ERROR: {unified_path} not found")
        return
    with open(unified_path) as f:
        unified_data = json.load(f)

    model_name = unified_data.get("model", "rf")
    mkey = model_name
    uni_direction = unified_data.get("direction", "up")
    uni_mode = unified_data.get("meta_label_mode", "tp")

    per_gran_results = {}
    for summary_path in sorted(per_gran_dir.glob("*/analysis_summary.json")):
        with open(summary_path) as f:
            data = json.load(f)
        if data.get("direction") != uni_direction or data.get("meta_label_mode") != uni_mode:
            continue
        gran = data.get("granularity", summary_path.parent.name.split("_")[0])
        per_gran_results[gran] = data

    def _g(d, *keys, default="—"):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return default
            if d is None:
                return default
        return d

    def _fmt(val, fmt_str=".1%"):
        if val == "—" or val is None:
            return "—"
        try:
            v = float(val)
            if fmt_str == ".1%":
                return f"{v:.1%}"
            if fmt_str == ".2f":
                return f"{v:.2f}"
            if fmt_str == "+.2f":
                return f"{v:+.2f}%"
            if fmt_str == "d":
                return f"{int(v)}"
            return f"{v:{fmt_str}}"
        except (ValueError, TypeError):
            return "—"

    def _pct(val):
        if val == "—" or val is None:
            return "—"
        try:
            return f"{float(val):.2f}%"
        except (ValueError, TypeError):
            return "—"

    def _val_row_pergran(data):
        t = _g(data, f"{mkey}_temporal_all_features", default={})
        v, vs = _g(t, "Val", default={}), _g(t, "Val_selective", default={})
        return [
            _fmt(_g(v, "coverage")),
            _fmt(_g(v, "risk")),
            _fmt(_g(v, "precision")),
            _fmt(_g(v, "baseline")),
            _fmt(_g(vs, "threshold"), ".2f"),
            _fmt(_g(vs, "coverage")),
            _fmt(_g(vs, "risk")),
            _fmt(_g(vs, "precision")),
        ]

    def _val_row_unified(data, gran):
        gd = _g(data, "per_gran", gran, default={})
        v, vs = _g(gd, "Val", default={}), _g(gd, "val_selective", default={})
        return [
            _fmt(_g(v, "coverage")),
            _fmt(_g(v, "risk")),
            _fmt(_g(v, "precision")),
            _fmt(_g(v, "baseline")),
            _fmt(_g(gd, "threshold"), ".2f"),
            _fmt(_g(vs, "coverage")),
            _fmt(_g(vs, "risk")),
            _fmt(_g(vs, "precision")),
        ]

    def _test_row_pergran(data):
        t = _g(data, f"{mkey}_temporal_all_features", default={})
        te, ts = _g(t, "Test", default={}), _g(t, "Test_selective", default={})
        return [
            _fmt(_g(te, "coverage")),
            _fmt(_g(te, "risk")),
            _fmt(_g(te, "precision")),
            _fmt(_g(te, "baseline")),
            _fmt(_g(ts, "threshold"), ".2f"),
            _fmt(_g(ts, "coverage")),
            _fmt(_g(ts, "risk")),
            _fmt(_g(ts, "precision")),
        ]

    def _test_row_unified(data, gran):
        gd = _g(data, "per_gran", gran, default={})
        te, ts = _g(gd, "Test", default={}), _g(gd, "test_selective", default={})
        return [
            _fmt(_g(te, "coverage")),
            _fmt(_g(te, "risk")),
            _fmt(_g(te, "precision")),
            _fmt(_g(te, "baseline")),
            _fmt(_g(gd, "threshold"), ".2f"),
            _fmt(_g(ts, "coverage")),
            _fmt(_g(ts, "risk")),
            _fmt(_g(ts, "precision")),
        ]

    def _bt_row_pergran(data):
        bt = _g(data, f"{mkey}_backtest_all_features", default={})
        return [
            _pct(_g(bt, "execution_rate")),
            _fmt(_g(bt, "n_total_trades"), "d"),
            _fmt(_g(bt, "n_m2_trades"), "d"),
            _fmt(_g(bt, "m2_total_return"), "+.2f"),
            _fmt(_g(bt, "m1_total_return"), "+.2f"),
            _fmt(_g(bt, "bh_total_return"), "+.2f"),
            _fmt(_g(bt, "m2_sharpe"), ".2f"),
            _fmt(_g(bt, "m1_sharpe"), ".2f"),
            _fmt(_g(bt, "m2_max_drawdown"), "+.2f"),
            _fmt(_g(bt, "m1_max_drawdown"), "+.2f"),
            _fmt(_g(bt, "bh_max_drawdown"), "+.2f"),
        ]

    def _bt_row_unified(data, gran):
        bt = _g(data, "per_gran", gran, "backtest", default={})
        return [
            _pct(_g(bt, "execution_rate")),
            _fmt(_g(bt, "n_total_trades"), "d"),
            _fmt(_g(bt, "n_m2_trades"), "d"),
            _fmt(_g(bt, "m2_total_return"), "+.2f"),
            _fmt(_g(bt, "m1_total_return"), "+.2f"),
            _fmt(_g(bt, "bh_total_return"), "+.2f"),
            _fmt(_g(bt, "m2_sharpe"), ".2f"),
            _fmt(_g(bt, "m1_sharpe"), ".2f"),
            _fmt(_g(bt, "m2_max_drawdown"), "+.2f"),
            _fmt(_g(bt, "m1_max_drawdown"), "+.2f"),
            _fmt(_g(bt, "bh_max_drawdown"), "+.2f"),
        ]

    row_labels, row_types = [], []
    val_rows, test_rows, bt_rows = [], [], []

    for gran in GRAN_ORDER:
        if gran in per_gran_results:
            row_labels.append(gran)
            row_types.append("sep")
            val_rows.append(_val_row_pergran(per_gran_results[gran]))
            test_rows.append(_test_row_pergran(per_gran_results[gran]))
            bt_rows.append(_bt_row_pergran(per_gran_results[gran]))
        if gran in unified_data.get("per_gran", {}):
            row_labels.append(f"{gran}_uni")
            row_types.append("uni")
            val_rows.append(_val_row_unified(unified_data, gran))
            test_rows.append(_test_row_unified(unified_data, gran))
            bt_rows.append(_bt_row_unified(unified_data, gran))

    if not row_labels:
        print("ERROR: No data found for comparison.")
        return

    COL_SEP = "#E0F7FA"
    COL_UNI = "#80DEEA"
    COL_WIN = "#2E7D32"
    COL_ALERT = "#C62828"
    COL_GRAN_EDGE = "#263238"
    COL_GRAN_TEXT = "white"
    HDR_ORANGE = "#E65100"
    HDR_BLUE = "#0D47A1"
    HDR_MAGENTA = "#880E4F"

    _COMPARE_COLS_VALTEST = {0: "higher", 1: "lower", 2: "higher", 5: "higher", 6: "lower", 7: "higher"}
    _COMPARE_COLS_BT = {0: "higher", 2: "higher", 3: "higher", 6: "higher", 8: "higher"}

    def _parse_pct(s):
        if s == "—" or s is None:
            return None
        s = str(s).strip().rstrip("%")
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def _build_winner_map(rows, compare_cols):
        winners = {}
        i = 0
        while i + 1 < len(rows):
            if row_types[i] == "sep" and row_types[i + 1] == "uni":
                for col, direction in compare_cols.items():
                    v0 = _parse_pct(rows[i][col])
                    v1 = _parse_pct(rows[i + 1][col])
                    if v0 is not None and v1 is not None and v0 != v1:
                        if direction == "higher":
                            win = i if v0 > v1 else i + 1
                        else:
                            win = i if v0 < v1 else i + 1
                        winners[(win, col)] = True
                i += 2
            else:
                i += 1
        return winners

    def _is_alert_cell(table_title: str, row_vals: list[str], col_idx: int) -> bool:
        if table_title == "Backtest":
            if col_idx in (3, 6):
                v = _parse_pct(row_vals[col_idx]) if col_idx < len(row_vals) else None
                return (v is not None) and (v < 0)
            return False
        if table_title in ("Validation", "Test"):
            if col_idx != 7:
                return False
            base = _parse_pct(row_vals[3]) if len(row_vals) > 3 else None
            wr_sel = _parse_pct(row_vals[7]) if len(row_vals) > 7 else None
            return (base is not None) and (wr_sel is not None) and (wr_sel < base)
        return False

    val_col_labels = ["Cov", "Risk", "WR", "Base", "Thr", "Cov", "Risk", "WR"]
    test_col_labels = ["Cov", "Risk", "WR", "Base", "Thr", "Cov", "Risk", "WR"]
    bt_col_labels = ["Exec%", "#M1", "#M2", "M2 Ret", "M1 Ret", "BH Ret", "M2 SR", "M1 SR", "M2 MDD", "M1 MDD", "BH MDD"]

    direction_label = uni_direction.upper()
    suptitle = f"Comparison: Per-Gran vs Unified — {_model_label(model_name)} {direction_label} {uni_mode.upper()}"

    n_rows = len(row_labels)
    _row_grans = [lbl.replace("_uni", "") for lbl in row_labels]
    _gran_groups = []
    gi = 0
    while gi < n_rows:
        g = _row_grans[gi]
        gj = gi + 1
        while gj < n_rows and _row_grans[gj] == g:
            gj += 1
        _gran_groups.append((gi, gj))
        gi = gj

    def _render_table(ax, col_labels, cell_data, title, hdr_color, sub_groups=None, winner_map=None):
        n_c = len(col_labels)
        full_data = [col_labels] + cell_data
        full_row_labels = [""] + row_labels
        header_rows = 1

        table = ax.table(
            cellText=full_data,
            rowLabels=full_row_labels,
            colWidths=[0.11] * n_c,
            cellLoc="center",
            loc="top",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(0.8, 1.5)

        for j in range(n_c):
            cell = table[0, j]
            cell.set_facecolor(hdr_color)
            cell.set_text_props(fontweight="bold", fontsize=8, color="white")
        rl = table[0, -1]
        rl.set_facecolor(hdr_color)
        rl.set_text_props(fontweight="bold", fontsize=8, color="white")

        if sub_groups:
            _deferred_labels.append((ax, table, sub_groups, hdr_color))
        _gran_badge_deferred.append((ax, table, hdr_color))

        for i in range(n_rows):
            is_uni = row_types[i] == "uni"
            bg_label = COL_UNI if is_uni else COL_SEP
            bg_data = "white" if is_uni else COL_SEP

            rl = table[header_rows + i, -1]
            rl.set_facecolor(bg_label)
            rl.set_text_props(fontweight="bold", fontsize=8)
            for j in range(n_c):
                cell = table[header_rows + i, j]
                if _is_alert_cell(title, cell_data[i], j):
                    cell.set_facecolor(COL_ALERT)
                    cell.set_text_props(fontsize=8, color="white", fontweight="bold")
                elif winner_map and (i, j) in winner_map:
                    cell.set_facecolor(COL_WIN)
                    cell.set_text_props(fontsize=8, color="white", fontweight="bold")
                else:
                    cell.set_facecolor(bg_data)
                    cell.set_text_props(fontsize=8)

        merge_cols = []
        if title in ["Validation", "Test"]:
            merge_cols = [3]
        elif title == "Backtest":
            merge_cols = [1, 4, 5, 7, 9, 10]

        if merge_cols:
            i = 0
            while i + 1 < n_rows:
                if row_types[i] == "sep" and row_types[i + 1] == "uni":
                    for c_idx in merge_cols:
                        top_cell = table[header_rows + i, c_idx]
                        bot_cell = table[header_rows + i + 1, c_idx]
                        val = top_cell.get_text().get_text()
                        top_cell.get_text().set_text("")
                        bot_cell.get_text().set_text("")
                        top_cell.set_facecolor("white")
                        bot_cell.set_facecolor("white")
                        top_cell.visible_edges = "LRT"
                        bot_cell.visible_edges = "LRB"
                        _deferred_base_merges.append((table, header_rows + i, header_rows + i + 1, c_idx, val))
                    i += 2
                else:
                    i += 1

        ax.set_title(title, fontsize=11, fontweight="bold", color=hdr_color, y=1.8)
        return table

    val_sub = [("Pre-Selective", 4), ("Selective", 4)]
    test_sub = [("Pre-Selective", 4), ("Selective", 4)]

    fig_w = 15
    row_h = 0.35
    table_h_val = 0.4 + (n_rows + 2) * row_h
    table_h_bt = 0.4 + (n_rows + 1) * row_h
    fig_h = table_h_bt + table_h_val + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 2, height_ratios=[table_h_bt, table_h_val], hspace=-0.05, wspace=0.01)
    ax_bt = fig.add_subplot(gs[0, :])
    ax_val = fig.add_subplot(gs[1, 0])
    ax_test = fig.add_subplot(gs[1, 1])

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.25)

    for ax in [ax_bt, ax_val, ax_test]:
        ax.axis("off")

    _deferred_labels = []
    _deferred_base_merges = []
    _gran_badge_deferred = []
    val_winners = _build_winner_map(val_rows, _COMPARE_COLS_VALTEST)
    test_winners = _build_winner_map(test_rows, _COMPARE_COLS_VALTEST)
    bt_winners = _build_winner_map(bt_rows, _COMPARE_COLS_BT)
    _render_table(ax_bt, bt_col_labels, bt_rows, "Backtest", HDR_MAGENTA, winner_map=bt_winners)
    _render_table(ax_val, val_col_labels, val_rows, "Validation", HDR_ORANGE, sub_groups=val_sub, winner_map=val_winners)
    _render_table(ax_test, test_col_labels, test_rows, "Test", HDR_BLUE, sub_groups=test_sub, winner_map=test_winners)

    fig.subplots_adjust(top=0.9)
    save_path = output_dir / f"comparison_table_{uni_direction}.png"

    if _deferred_labels or _deferred_base_merges or _gran_badge_deferred:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_w_px = fig.get_figwidth() * fig.dpi
        fig_h_px = fig.get_figheight() * fig.dpi

        for ax_ref, tbl, sgroups, color in _deferred_labels:
            del ax_ref
            sample_bb = tbl[0, 0].get_window_extent(renderer)
            row_h_px = sample_bb.height

            col_start = 0
            for label, span in sgroups:
                bb_left = tbl[0, col_start].get_window_extent(renderer)
                bb_right = tbl[0, col_start + span - 1].get_window_extent(renderer)
                x0_f = bb_left.x0 / fig_w_px
                x1_f = bb_right.x1 / fig_w_px
                y0_f = bb_left.y1 / fig_h_px
                h_f = (row_h_px * 0.7) / fig_h_px
                w_f = x1_f - x0_f

                rect = plt.Rectangle(
                    (x0_f, y0_f),
                    w_f,
                    h_f,
                    transform=fig.transFigure,
                    clip_on=False,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                )
                fig.patches.append(rect)
                fig.text(
                    x0_f + w_f / 2,
                    y0_f + h_f / 2,
                    label,
                    transform=fig.transFigure,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
                col_start += span

        for tbl, r_top, r_bot, col, val in _deferred_base_merges:
            bb_top = tbl[r_top, col].get_window_extent(renderer)
            bb_bot = tbl[r_bot, col].get_window_extent(renderer)
            x_center = (bb_top.x0 + bb_top.x1) / 2 / fig_w_px
            y_center = (bb_top.y1 + bb_bot.y0) / 2 / fig_h_px
            fig.text(x_center, y_center, val, transform=fig.transFigure, ha="center", va="center", fontsize=8)

        for _, tbl, badge_color in _gran_badge_deferred:
            for g_start, g_end in _gran_groups:
                gran_label = _row_grans[g_start]
                top_bb = tbl[1 + g_start, -1].get_window_extent(renderer)
                bot_bb = tbl[1 + g_end - 1, -1].get_window_extent(renderer)

                badge_h_px = top_bb.y1 - bot_bb.y0
                row_label_w_px = top_bb.x1 - top_bb.x0
                badge_w_px = min(max(row_label_w_px * 0.34, 14.0), 24.0)
                gap_px = 2.0

                x0_px = top_bb.x0 - gap_px - badge_w_px
                y0_px = bot_bb.y0

                x0_f = max(0.002, x0_px / fig_w_px)
                y0_f = y0_px / fig_h_px
                w_f = badge_w_px / fig_w_px
                h_f = badge_h_px / fig_h_px

                rect = plt.Rectangle(
                    (x0_f, y0_f),
                    w_f,
                    h_f,
                    transform=fig.transFigure,
                    clip_on=False,
                    facecolor=badge_color,
                    edgecolor=COL_GRAN_EDGE,
                    linewidth=0.9,
                )
                fig.patches.append(rect)
                fig.text(
                    x0_f + w_f / 2,
                    y0_f + h_f / 2,
                    gran_label,
                    transform=fig.transFigure,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=COL_GRAN_TEXT,
                    rotation=90,
                )

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison table saved: {save_path}")


# ┏━━━━━━━━━━ Run Paradigm (between Model types) Comparison ━━━━━━━━━━┓
def run_paradigm_comparison(dirs: list):
    """Compare result directories across paradigms with stacked summary tables."""
    dirs = [Path(d) for d in dirs]
    dir_labels = [d.name for d in dirs]
    short_labels = [
        d.replace("_7_fees", "").replace("_7_OCP_new", "+OCP").replace("_7_OCP_old", "+OCP_old").replace("_", " ")
        for d in dir_labels
    ]

    per_gran_data = {}
    unified_data = {}

    for p_idx, d in enumerate(dirs):
        for sp in sorted(d.glob("*/analysis_summary.json")):
            with open(sp) as f:
                data = json.load(f)
            direction = data.get("direction", "up")
            gran = data.get("granularity", sp.parent.name.split("_")[0])
            per_gran_data.setdefault(direction, {}).setdefault(gran, {})[p_idx] = data

        for usp in sorted(d.glob("unified_*/unified_summary.json")):
            with open(usp) as f:
                udata = json.load(f)
            direction = udata.get("direction", "up")
            unified_data.setdefault(direction, {})[p_idx] = udata

    if not per_gran_data:
        print("ERROR: No analysis_summary.json files found in any directory.")
        return

    def _g(d, *keys, default=None):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d if d is not None else default

    def _fmt(val, spec=".1%"):
        if val is None:
            return "—"
        try:
            v = float(val)
            if spec == ".1%":
                return f"{v:.1%}"
            if spec == "+.2f":
                return f"{v:+.2f}%"
            if spec == ".2f":
                return f"{v:.2f}"
            if spec == "d":
                return str(int(v))
            return f"{v:{spec}}"
        except (ValueError, TypeError):
            return "—"

    def _pct(val):
        if val is None:
            return "—"
        try:
            return f"{float(val):.2f}%"
        except Exception:
            return "—"

    def _val_row_pg(data):
        model = data.get("model", "rf")
        t = _g(data, f"{model}_temporal_all_features", default={})
        v, vs = _g(t, "Val", default={}), _g(t, "Val_selective", default={})
        return [
            _fmt(_g(v, "coverage")),
            _fmt(_g(v, "risk")),
            _fmt(_g(v, "precision")),
            _fmt(_g(v, "baseline")),
            _fmt(_g(vs, "threshold"), ".2f"),
            _fmt(_g(vs, "coverage")),
            _fmt(_g(vs, "risk")),
            _fmt(_g(vs, "precision")),
        ]

    def _test_row_pg(data):
        model = data.get("model", "rf")
        t = _g(data, f"{model}_temporal_all_features", default={})
        te, ts = _g(t, "Test", default={}), _g(t, "Test_selective", default={})
        return [
            _fmt(_g(te, "coverage")),
            _fmt(_g(te, "risk")),
            _fmt(_g(te, "precision")),
            _fmt(_g(te, "baseline")),
            _fmt(_g(ts, "threshold"), ".2f"),
            _fmt(_g(ts, "coverage")),
            _fmt(_g(ts, "risk")),
            _fmt(_g(ts, "precision")),
        ]

    def _bt_row_pg(data):
        model = data.get("model", "rf")
        bt = _g(data, f"{model}_backtest_all_features", default={})
        return [
            _pct(_g(bt, "execution_rate")),
            _fmt(_g(bt, "n_total_trades"), "d"),
            _fmt(_g(bt, "n_m2_trades"), "d"),
            _fmt(_g(bt, "m2_total_return"), "+.2f"),
            _fmt(_g(bt, "m1_total_return"), "+.2f"),
            _fmt(_g(bt, "bh_total_return"), "+.2f"),
            _fmt(_g(bt, "m2_sharpe"), ".2f"),
            _fmt(_g(bt, "m1_sharpe"), ".2f"),
            _fmt(_g(bt, "m2_max_drawdown"), "+.2f"),
            _fmt(_g(bt, "m1_max_drawdown"), "+.2f"),
            _fmt(_g(bt, "bh_max_drawdown"), "+.2f"),
        ]

    def _val_row_uni(udata, gran):
        gd = _g(udata, "per_gran", gran, default={})
        v, vs = _g(gd, "Val", default={}), _g(gd, "val_selective", default={})
        return [
            _fmt(_g(v, "coverage")),
            _fmt(_g(v, "risk")),
            _fmt(_g(v, "precision")),
            _fmt(_g(v, "baseline")),
            _fmt(_g(gd, "threshold"), ".2f"),
            _fmt(_g(vs, "coverage")),
            _fmt(_g(vs, "risk")),
            _fmt(_g(vs, "precision")),
        ]

    def _test_row_uni(udata, gran):
        gd = _g(udata, "per_gran", gran, default={})
        te, ts = _g(gd, "Test", default={}), _g(gd, "test_selective", default={})
        return [
            _fmt(_g(te, "coverage")),
            _fmt(_g(te, "risk")),
            _fmt(_g(te, "precision")),
            _fmt(_g(te, "baseline")),
            _fmt(_g(gd, "threshold"), ".2f"),
            _fmt(_g(ts, "coverage")),
            _fmt(_g(ts, "risk")),
            _fmt(_g(ts, "precision")),
        ]

    def _bt_row_uni(udata, gran):
        bt = _g(udata, "per_gran", gran, "backtest", default={})
        return [
            _pct(_g(bt, "execution_rate")),
            _fmt(_g(bt, "n_total_trades"), "d"),
            _fmt(_g(bt, "n_m2_trades"), "d"),
            _fmt(_g(bt, "m2_total_return"), "+.2f"),
            _fmt(_g(bt, "m1_total_return"), "+.2f"),
            _fmt(_g(bt, "bh_total_return"), "+.2f"),
            _fmt(_g(bt, "m2_sharpe"), ".2f"),
            _fmt(_g(bt, "m1_sharpe"), ".2f"),
            _fmt(_g(bt, "m2_max_drawdown"), "+.2f"),
            _fmt(_g(bt, "m1_max_drawdown"), "+.2f"),
            _fmt(_g(bt, "bh_max_drawdown"), "+.2f"),
        ]

    _PARADIGM_PAIRS = [
        ("#E3F2FD", "#BBDEFB"),
        ("#FFF3E0", "#FFE0B2"),
        ("#E8F5E9", "#C8E6C9"),
        ("#F3E5F5", "#E1BEE7"),
        ("#FFEBEE", "#FFCDD2"),
    ]
    COL_WIN = "#2E7D32"
    COL_ALERT = "#C62828"
    COL_GRAN_EDGE = "#263238"
    COL_GRAN_TEXT = "white"
    HDR_ORANGE = "#E65100"
    HDR_BLUE = "#0D47A1"
    HDR_MAG = "#880E4F"

    def _parse_pct(s):
        if s == "—" or s is None:
            return None
        s = str(s).strip().rstrip("%")
        try:
            return float(s)
        except Exception:
            return None

    def _is_alert_cell(table_title: str, row_vals: list[str], col_idx: int) -> bool:
        if table_title.startswith("Backtest"):
            if col_idx in (3, 6):
                v = _parse_pct(row_vals[col_idx]) if col_idx < len(row_vals) else None
                return (v is not None) and (v < 0)
            return False
        if table_title in ("Validation", "Test"):
            if col_idx != 7:
                return False
            base = _parse_pct(row_vals[3]) if len(row_vals) > 3 else None
            wr_sel = _parse_pct(row_vals[7]) if len(row_vals) > 7 else None
            return (base is not None) and (wr_sel is not None) and (wr_sel < base)
        return False

    for direction in sorted(per_gran_data.keys()):
        grans_data = per_gran_data[direction]
        uni_for_dir = unified_data.get(direction, {})

        row_labels = []
        row_p_idx = []
        row_is_uni = []
        val_rows, test_rows, bt_rows = [], [], []

        for gran in GRAN_ORDER:
            if gran not in grans_data:
                continue
            for p_idx in range(len(dir_labels)):
                if p_idx in grans_data[gran]:
                    data = grans_data[gran][p_idx]
                    row_labels.append(f"{gran} | {short_labels[p_idx]}")
                    row_p_idx.append(p_idx)
                    row_is_uni.append(False)
                    val_rows.append(_val_row_pg(data))
                    test_rows.append(_test_row_pg(data))
                    bt_rows.append(_bt_row_pg(data))

                if p_idx in uni_for_dir:
                    udata = uni_for_dir[p_idx]
                    if gran in _g(udata, "per_gran", default={}):
                        row_labels.append(f"{gran}_uni | {short_labels[p_idx]}")
                        row_p_idx.append(p_idx)
                        row_is_uni.append(True)
                        val_rows.append(_val_row_uni(udata, gran))
                        test_rows.append(_test_row_uni(udata, gran))
                        bt_rows.append(_bt_row_uni(udata, gran))

        if not row_labels:
            print(f"  No data for direction={direction}")
            continue

        n_rows = len(row_labels)

        def _build_winners(rows, compare_cols):
            winners = {}
            grans_seen = []
            for lbl in row_labels:
                g = lbl.split("|")[0].strip().replace("_uni", "")
                grans_seen.append(g)
            i = 0
            while i < n_rows:
                g = grans_seen[i]
                group = [j for j in range(i, n_rows) if grans_seen[j] == g]
                for col, d in compare_cols.items():
                    best_j, best_v = None, None
                    for j in group:
                        v = _parse_pct(rows[j][col])
                        if v is None:
                            continue
                        if best_v is None or (d == "higher" and v > best_v) or (d == "lower" and v < best_v):
                            best_v, best_j = v, j
                    if best_j is not None and len(group) > 1:
                        winners[(best_j, col)] = True
                i = group[-1] + 1
            return winners

        _COMPARE_VT = {0: "higher", 1: "lower", 2: "higher", 5: "higher", 6: "lower", 7: "higher"}
        _COMPARE_BT = {0: "higher", 2: "higher", 3: "higher", 6: "higher", 8: "higher"}

        val_winners = _build_winners(val_rows, _COMPARE_VT)
        test_winners = _build_winners(test_rows, _COMPARE_VT)
        bt_winners = _build_winners(bt_rows, _COMPARE_BT)

        val_col = ["Cov", "Risk", "WR", "Base", "Thr", "Cov", "Risk", "WR"]
        test_col = val_col[:]
        bt_col = ["Exec%", "#M1", "#M2", "M2 Ret", "M1 Ret", "BH Ret", "M2 SR", "M1 SR", "M2 MDD", "M1 MDD", "BH MDD"]

        _BT_SHARED_COLS = {1, 4, 5, 7, 9, 10}
        _VT_SHARED_COLS = {3}
        COL_MERGED = "#ECEFF1"

        _row_grans = [lbl.split("|")[0].strip().replace("_uni", "") for lbl in row_labels]

        _gran_groups = []
        i = 0
        while i < n_rows:
            g = _row_grans[i]
            j = i + 1
            while j < n_rows and _row_grans[j] == g:
                j += 1
            _gran_groups.append((i, j))
            i = j

        for g_start, g_end in _gran_groups:
            g_size = g_end - g_start
            mid = g_start + g_size // 2
            for row_i in range(g_start, g_end):
                if row_i != mid:
                    for col_idx in _BT_SHARED_COLS:
                        if col_idx < len(bt_rows[row_i]):
                            bt_rows[row_i][col_idx] = ""
                    for col_idx in _VT_SHARED_COLS:
                        if col_idx < len(val_rows[row_i]):
                            val_rows[row_i][col_idx] = ""
                        if col_idx < len(test_rows[row_i]):
                            test_rows[row_i][col_idx] = ""

        def _render_table(ax, col_labels, cell_data, title, hdr_color, sub_groups=None, winner_map=None, shared_cols=None):
            n_c = len(col_labels)
            full_data = [col_labels] + cell_data
            full_row_labels = [""] + row_labels

            table = ax.table(cellText=full_data, rowLabels=full_row_labels, colWidths=[0.11] * n_c, cellLoc="center", loc="top")
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(0.8, 1.4)

            for j in range(n_c):
                table[0, j].set_facecolor(hdr_color)
                table[0, j].set_text_props(fontweight="bold", fontsize=7, color="white")
            table[0, -1].set_facecolor(hdr_color)
            table[0, -1].set_text_props(fontweight="bold", fontsize=7, color="white")

            if sub_groups:
                _sg_deferred.append((ax, table, sub_groups, hdr_color))

            for i in range(n_rows):
                p = row_p_idx[i]
                is_u = row_is_uni[i]
                bg_pair = _PARADIGM_PAIRS[p % len(_PARADIGM_PAIRS)]
                bg = bg_pair[1] if is_u else bg_pair[0]

                rl = table[1 + i, -1]
                rl.set_facecolor(bg)
                rl.set_text_props(fontweight="bold", fontsize=7)

                for j in range(n_c):
                    cell = table[1 + i, j]
                    if _is_alert_cell(title, cell_data[i], j):
                        cell.set_facecolor(COL_ALERT)
                        cell.set_text_props(fontsize=7, color="white", fontweight="bold")
                    elif winner_map and (i, j) in winner_map:
                        cell.set_facecolor(COL_WIN)
                        cell.set_text_props(fontsize=7, color="white", fontweight="bold")
                    else:
                        cell.set_facecolor(bg)
                        cell.set_text_props(fontsize=7)

            if shared_cols:
                for g_start, g_end in _gran_groups:
                    g_size = g_end - g_start
                    if g_size <= 1:
                        continue
                    for col_idx in shared_cols:
                        if col_idx >= n_c:
                            continue
                        for k, row_i in enumerate(range(g_start, g_end)):
                            cell = table[1 + row_i, col_idx]
                            cell.set_facecolor(COL_MERGED)
                            cell.set_text_props(fontsize=7, color="#212121")
                            if k == 0:
                                cell.visible_edges = "TLR"
                            elif k == g_size - 1:
                                cell.visible_edges = "BLR"
                            else:
                                cell.visible_edges = "LR"

            _title_deferred.append((ax, table, title, hdr_color))
            _gran_badge_deferred.append((ax, table, hdr_color))
            return table

        row_h = 0.30
        table_h_val = 0.4 + (n_rows + 2) * row_h
        table_h_bt = 0.4 + (n_rows + 1) * row_h
        fig_h = table_h_bt + table_h_val + 1.2
        fig_w = 17

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(2, 2, height_ratios=[table_h_bt, table_h_val], hspace=0.08, wspace=0.01)
        ax_bt = fig.add_subplot(gs[0, :])
        ax_val = fig.add_subplot(gs[1, 0])
        ax_test = fig.add_subplot(gs[1, 1])

        dir_label = direction.upper()

        for ax in [ax_bt, ax_val, ax_test]:
            ax.axis("off")

        _sg_deferred = []
        _title_deferred = []
        _gran_badge_deferred = []
        val_sub = [("Pre-Selective", 4), ("Selective", 4)]
        test_sub = [("Pre-Selective", 4), ("Selective", 4)]

        _render_table(ax_bt, bt_col, bt_rows, f"Backtest — {dir_label}", HDR_MAG, winner_map=bt_winners, shared_cols=_BT_SHARED_COLS)
        _render_table(ax_val, val_col, val_rows, "Validation", HDR_ORANGE, sub_groups=val_sub, winner_map=val_winners, shared_cols=_VT_SHARED_COLS)
        _render_table(ax_test, test_col, test_rows, "Test", HDR_BLUE, sub_groups=test_sub, winner_map=test_winners, shared_cols=_VT_SHARED_COLS)

        fig.subplots_adjust(top=0.92)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_w_px = fig.get_figwidth() * fig.dpi
        fig_h_px = fig.get_figheight() * fig.dpi

        for ax_ref, tbl, title, hdr_color in _title_deferred:
            del ax_ref
            n_c_local = len([k for k in tbl.get_celld().keys() if k[0] == 0 and k[1] >= 0])
            if n_c_local == 0:
                continue
            bb_first = tbl[0, 0].get_window_extent(renderer)
            bb_last = tbl[0, n_c_local - 1].get_window_extent(renderer)
            x_center = (bb_first.x0 + bb_last.x1) / 2 / fig_w_px
            y_top = bb_first.y1 / fig_h_px + 0.015
            fig.text(
                x_center,
                y_top,
                title,
                transform=fig.transFigure,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=hdr_color,
            )

        if _sg_deferred:
            for ax_ref, tbl, sgroups, color in _sg_deferred:
                del ax_ref
                sample_bb = tbl[0, 0].get_window_extent(renderer)
                row_h_px = sample_bb.height

                col_start = 0
                for label, span in sgroups:
                    bb_left = tbl[0, col_start].get_window_extent(renderer)
                    bb_right = tbl[0, col_start + span - 1].get_window_extent(renderer)
                    x0_f = bb_left.x0 / fig_w_px
                    x1_f = bb_right.x1 / fig_w_px
                    y0_f = bb_left.y1 / fig_h_px
                    h_f = (row_h_px * 0.7) / fig_h_px
                    w_f = x1_f - x0_f

                    rect = plt.Rectangle(
                        (x0_f, y0_f),
                        w_f,
                        h_f,
                        transform=fig.transFigure,
                        clip_on=False,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=1.0,
                    )
                    fig.patches.append(rect)
                    fig.text(
                        x0_f + w_f / 2,
                        y0_f + h_f / 2,
                        label,
                        transform=fig.transFigure,
                        ha="center",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                        color="white",
                    )
                    col_start += span

        if _gran_badge_deferred:
            for _, tbl, badge_color in _gran_badge_deferred:
                for g_start, g_end in _gran_groups:
                    gran_label = _row_grans[g_start]
                    top_bb = tbl[1 + g_start, -1].get_window_extent(renderer)
                    bot_bb = tbl[1 + g_end - 1, -1].get_window_extent(renderer)

                    badge_h_px = top_bb.y1 - bot_bb.y0
                    row_label_w_px = top_bb.x1 - top_bb.x0
                    badge_w_px = min(max(row_label_w_px * 0.34, 14.0), 24.0)
                    gap_px = 2.0

                    x0_px = top_bb.x0 - gap_px - badge_w_px
                    y0_px = bot_bb.y0

                    x0_f = max(0.002, x0_px / fig_w_px)
                    y0_f = y0_px / fig_h_px
                    w_f = badge_w_px / fig_w_px
                    h_f = badge_h_px / fig_h_px

                    rect = plt.Rectangle(
                        (x0_f, y0_f),
                        w_f,
                        h_f,
                        transform=fig.transFigure,
                        clip_on=False,
                        facecolor=badge_color,
                        edgecolor=COL_GRAN_EDGE,
                        linewidth=0.9,
                    )
                    fig.patches.append(rect)
                    fig.text(
                        x0_f + w_f / 2,
                        y0_f + h_f / 2,
                        gran_label,
                        transform=fig.transFigure,
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        fontweight="bold",
                        color=COL_GRAN_TEXT,
                        rotation=90,
                    )

        save_path = dirs[0].parent / f"paradigm_comparison_{direction}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Paradigm comparison saved: {save_path}")

        csv_path = save_path.with_suffix(".csv")
        with open(csv_path, "w", newline="") as cf:
            writer = csv_mod.writer(cf)
            writer.writerow(["Row", "Type"] + bt_col + [""] + val_col + [""] + test_col)
            for i in range(n_rows):
                rtype = "unified" if row_is_uni[i] else "per-gran"
                writer.writerow([row_labels[i], rtype] + bt_rows[i] + [""] + val_rows[i] + [""] + test_rows[i])
        print(f"  CSV saved: {csv_path}")
