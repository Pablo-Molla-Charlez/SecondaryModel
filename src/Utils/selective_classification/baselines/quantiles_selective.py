"""M1+PI — native quantile predictive-interval gate (Kronos).

Trains NO second-stage model. It reuses M1's own predictive spread: a narrow
forecast interval means M1 is confident, a wide one means it is not.

    width_t = (q90_t - q10_t) / close_t          (relative, comparable across assets)
    score_t = -width_t                           (narrower -> higher score)

The M1 signal is executed when ``score_t >= tau``, with tau chosen on validation by
the same risk-budget rule as every other gate.

Requires the Kronos CSVs to carry ``pred_q10`` / ``pred_q90`` (added by the
Monte-Carlo quantile patch in the Primary-Model pipeline). Kronos draws 30 sample
paths internally; the patch persists their percentiles instead of discarding them.
Slices lacking those columns are reported as skipped rather than silently dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.selective_classification.baselines._common import (
    DEFAULT_KRONOS_ROOT, GRANULARITIES, SIDES,
    discover_run_dir, list_assets, load_signal_frame,
    split_indices, subset, selective_metrics, pick_threshold, trade_returns,
    m1_baseline_precision, RISK_ALPHA,
)

# Column names written by the Kronos MC-quantile patch.
Q_LOW_COL = "pred_q10"
Q_HIGH_COL = "pred_q90"


def relative_interval_width(frame: pd.DataFrame, close: np.ndarray) -> np.ndarray:
    """(q90 - q10) / close, guarded against division by zero."""
    if Q_LOW_COL not in frame.columns or Q_HIGH_COL not in frame.columns:
        raise KeyError(
            f"Missing quantile columns '{Q_LOW_COL}'/'{Q_HIGH_COL}'. Re-run the Kronos "
            "pipeline with Monte-Carlo quantile persistence enabled."
        )
    lo = pd.to_numeric(frame[Q_LOW_COL], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(frame[Q_HIGH_COL], errors="coerce").to_numpy(dtype=float)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    width = np.maximum(hi - lo, 0.0)
    return width / np.maximum(np.abs(close), 1e-8)


def pi_scores(frame: pd.DataFrame, close: np.ndarray) -> np.ndarray:
    """Reliability score: negative relative width (narrower interval = more reliable)."""
    rel = relative_interval_width(frame, close)
    return np.nan_to_num(-rel, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)


def run_slice(run_dir: Path, asset: str, granularity: str, side: str,
              risk_alpha: float = RISK_ALPHA) -> dict:
    """Evaluate the M1+PI gate on one (asset, granularity, side) slice."""
    sf = load_signal_frame(run_dir, asset, granularity, side)
    idx_tr, idx_va, idx_te = split_indices(len(sf), sf.dates)
    val, test = subset(sf, idx_va), subset(sf, idx_te)

    # The gate only ever acts on rows where M1 actually fired a signal.
    va_mask, te_mask = val.m1_pred == 1, test.m1_pred == 1
    if not va_mask.any() or not te_mask.any():
        raise ValueError("No M1-positive rows in validation or test split.")

    va_scores = pi_scores(val.frame, val.close)[va_mask]
    te_scores = pi_scores(test.frame, test.close)[te_mask]
    va_y, te_y = val.meta_label[va_mask], test.meta_label[te_mask]

    sel = pick_threshold(va_y, va_scores, risk_alpha=risk_alpha)
    tau = sel["threshold"]
    te_ret = trade_returns(test.close, test.ground_truth, side)[te_mask]
    metrics = selective_metrics(te_y, te_scores >= tau, returns=te_ret)

    m1_prec = m1_baseline_precision(test)
    return {
        "method": "M1+PI", "asset": asset, "granularity": granularity, "side": side,
        "threshold": tau, "constraint_satisfied": sel["constraint_satisfied"],
        "val_precision": sel["val_precision"], "val_coverage": sel["val_coverage"],
        "precision": metrics["precision"], "risk": metrics["risk"],
        "coverage": metrics["coverage"], "selected_count": metrics["selected_count"],
        "n_signals": metrics["n_signals"], "m1_precision": m1_prec,
        "delta_precision_vs_m1": metrics["precision"] - m1_prec,
        "mean_return": metrics.get("mean_return", float("nan")),
        "total_return": metrics.get("total_return", float("nan")),
        # the same trades without any gate, so the gate's effect is readable
        "m1_mean_return": float(te_ret[np.isfinite(te_ret)].mean() * 100.0)
                          if te_ret.size else float("nan"),
        "mean_rel_width": float(np.mean(relative_interval_width(test.frame, test.close))),
        "skipped_reason": "",
    }


def run_all(kronos_root: Path, granularities, sides, horizon: int = 7,
            risk_alpha: float = RISK_ALPHA, assets=None) -> pd.DataFrame:
    rows = []
    for gran in granularities:
        try:
            run_dir = discover_run_dir(kronos_root, gran, horizon=horizon)
        except FileNotFoundError as exc:
            rows.append({"method": "M1+PI", "granularity": gran, "skipped_reason": str(exc)})
            continue
        for side in sides:
            for asset in (assets or list_assets(run_dir, side)):
                try:
                    rows.append(run_slice(run_dir, asset, gran, side, risk_alpha))
                except Exception as exc:
                    rows.append({"method": "M1+PI", "asset": asset, "granularity": gran,
                                 "side": side, "skipped_reason": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="M1+PI quantile predictive-interval gate (Kronos).")
    p.add_argument("--kronos-root", type=Path, default=DEFAULT_KRONOS_ROOT)
    p.add_argument("--granularities", nargs="+", default=list(GRANULARITIES))
    p.add_argument("--sides", nargs="+", default=list(SIDES))
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--risk-alpha", type=float, default=RISK_ALPHA)
    p.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    args = p.parse_args()

    df = run_all(args.kronos_root, args.granularities, args.sides,
                 horizon=args.horizon, risk_alpha=args.risk_alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    ok = df[df["skipped_reason"] == ""] if "skipped_reason" in df.columns else df
    print(f"[M1+PI] wrote {args.out}  ({len(ok)} ok / {len(df)} rows)")
    if not ok.empty:
        print(ok.groupby("granularity")[["precision", "coverage"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
