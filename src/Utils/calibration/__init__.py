"""Calibration analysis package.

Studies:
  (1) Per-split TP rates (train / val-cal / val-opt / test)
  (2) Isotonic-regression degeneracy triggers on Val-Cal
  (3) Val-Opt raw vs. calibrated classification + financial metrics
  (4) 4-stage comparison: (NoCal, NoOpt) / (Cal, NoOpt) / (NoCal, Opt) / (Cal, Opt)
  (5) Threshold-optimization success/fail statistics (Utility-Opt vs fallback stages)

Data sources:
  • rf / tabpfn / tabicl  → Output/{M1}/HPO/{m2}/{DIR}/{gran}/best_probs.npz
  • autogluon             → Output/{M1}/{m2}/{DIR}/Utility_Score/{gran}_training/final_model/best_probs.npz
  • Returns + labels      → cache .pt loaded once per (M1, direction), split with embargo
"""
from Utils.calibration.analysis import (collect_split_tp_rates,
                                        collect_degeneracy_triggers,
                                        collect_val_opt_comparisons,
                                        collect_threshold_optimization_stats)

__all__ = ["collect_split_tp_rates",
           "collect_degeneracy_triggers",
           "collect_val_opt_comparisons",
           "collect_threshold_optimization_stats"]
