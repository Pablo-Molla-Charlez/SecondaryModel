"""CLI for the calibration-analysis package.

Usage:
    python -m Utils.calibration --study all
    python -m Utils.calibration --study split_tp_rate
    python -m Utils.calibration --study degeneracy
    python -m Utils.calibration --study val_opt
    python -m Utils.calibration --study thr_opt_stats
    python -m Utils.calibration --study all --m1 Kronos --m2 rf tabpfn tabicl

Outputs go to Output/Analysis/Calibration/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from Utils.calibration.analysis import (load_default_config,
                                         collect_split_tp_rates,
                                         collect_degeneracy_triggers,
                                         collect_val_opt_comparisons,
                                         collect_threshold_optimization_stats)
from Utils.calibration.plots import (plot_split_tp_rate_heatmaps,
                                      plot_degeneracy_triggers,
                                      plot_val_opt_classification,
                                      plot_val_opt_financial,
                                      plot_threshold_optimization_stats)


_STUDIES = ("split_tp_rate", "degeneracy", "val_opt", "thr_opt_stats", "all")


def main():
    parser = argparse.ArgumentParser(description="Calibration analysis")
    parser.add_argument("--study", choices=_STUDIES, default="all",
                        help="Which study to run (default: all)")
    parser.add_argument("--m1", nargs="+", default=["Kronos"],
                        help="M1 models to include (default: Kronos)")
    parser.add_argument("--m2", nargs="+", default=["rf", "tabpfn", "tabicl", "autogluon"],
                        help="M2 models to include (default: rf tabpfn tabicl autogluon)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (defaults to Secondary-Model/src/config.yaml)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Output dir (defaults to Output/Analysis/Calibration)")
    args = parser.parse_args()

    cfg = load_default_config(args.config)
    save_dir = Path(args.save_dir) if args.save_dir else (_SRC / "Output" / "Analysis" / "Calibration")
    save_dir.mkdir(parents=True, exist_ok=True)

    def _run_split_tp_rate():
        print("\n[1] Collecting per-split TP rates…")
        df = collect_split_tp_rates(cfg, m1_models=args.m1)
        if df.empty:
            print("  No data collected."); return
        df.sort_values(["m1_model", "direction", "granularity", "split"]).to_csv(
            save_dir / "calibration_split_tp_rate.csv", index=False)
        out = plot_split_tp_rate_heatmaps(df, save_dir)
        print(f"  Saved: {out}")

    def _run_degeneracy():
        print("\n[2] Collecting isotonic-degeneracy triggers on Val-Cal…")
        df = collect_degeneracy_triggers(m1_models=args.m1, m2_models=args.m2)
        if df.empty:
            print("  No data collected (no best_probs.npz found)."); return
        df.sort_values(["m2_model", "m1_model", "direction", "granularity"]).to_csv(
            save_dir / "calibration_degeneracy_triggers.csv", index=False)
        per_m2, agg = plot_degeneracy_triggers(df, save_dir)
        print(f"  Saved: {per_m2}\n         {agg}")

    def _run_val_opt():
        print("\n[3/4] Collecting Val-Opt 4-stage (raw/cal × τ=0.5/τ*) metrics…")
        df = collect_val_opt_comparisons(cfg, m1_models=args.m1, m2_models=args.m2)
        if df.empty:
            print("  No data collected."); return
        df.sort_values(["m2_model", "m1_model", "direction", "granularity", "stage"]).to_csv(
            save_dir / "calibration_val_opt_gain.csv", index=False)
        clf_per, clf_agg = plot_val_opt_classification(df, save_dir)
        fin_per, fin_agg = plot_val_opt_financial(df, save_dir)
        print(f"  Saved: {clf_per}\n         {clf_agg}")
        print(f"         {fin_per}\n         {fin_agg}")

    def _run_thr_opt_stats():
        print("\n[5] Collecting threshold-optimization success/failure stats…")
        df = collect_threshold_optimization_stats(cfg, m1_models=args.m1, m2_models=args.m2)
        if df.empty:
            print("  No data collected."); return
        df.sort_values(["m2_model", "m1_model", "direction", "granularity", "probs_variant"]).to_csv(
            save_dir / "calibration_threshold_opt_stats.csv", index=False)
        per_m2, agg = plot_threshold_optimization_stats(df, save_dir)
        print(f"  Saved: {per_m2}\n         {agg}")
        # Headline numbers
        n_total = len(df)
        n_succ  = int((df["threshold_source"] == "Utility-Opt").sum())
        print(f"\n  Headline: Utility-Opt picked in {n_succ}/{n_total} "
              f"({100.0*n_succ/max(n_total,1):.1f}%) of (m1×m2×direction×gran×variant) configs.")

    if args.study in ("split_tp_rate", "all"): _run_split_tp_rate()
    if args.study in ("degeneracy",    "all"): _run_degeneracy()
    if args.study in ("val_opt",       "all"): _run_val_opt()
    if args.study in ("thr_opt_stats", "all"): _run_thr_opt_stats()

    print(f"\nAll outputs in: {save_dir}")


if __name__ == "__main__":
    main()
