"""Random-coverage gate — the "is M2 actually selecting skilfully?" control.

An M2 model improves precision partly by simply trading LESS. This baseline removes
that confound: given an M2's realized coverage c, it selects a RANDOM c-fraction of
the same M1 signals, repeats the draw ``n_trials`` times, and reports the resulting
precision distribution.

The comparison is then a one-sided empirical test:

    p = P(random precision >= M2 precision)

A small p means M2's precision cannot be explained by reduced coverage alone — it is
picking the *right* signals, not merely fewer. This directly answers the reviewer
request for statistical significance and failure rates, and needs no model training:
it resamples decisions that already exist.

Coverage can be supplied explicitly (``--coverage``) or read from an M2 results CSV
carrying ``coverage``/``precision`` columns per (asset, granularity, side).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.selective_classification.baselines._common import (
    DEFAULT_KRONOS_ROOT, GRANULARITIES, SIDES,
    discover_run_dir, list_assets, load_signal_frame,
    split_indices, subset, m1_baseline_precision,
)

DEFAULT_TRIALS = 1000


def random_precision_distribution(y_true: np.ndarray, coverage: float,
                                  n_trials: int = DEFAULT_TRIALS,
                                  seed: int = 42) -> np.ndarray:
    """Precision of ``n_trials`` random selections at the given coverage.

    ``y_true`` is the meta label over M1-positive rows (1 = M1 was correct). Each
    trial draws ``round(coverage * n)`` rows uniformly WITHOUT replacement.
    """
    n = int(y_true.size)
    k = int(round(float(coverage) * n))
    if n == 0 or k <= 0:
        return np.array([], dtype=float)
    k = min(k, n)

    rng = np.random.default_rng(seed)
    out = np.empty(n_trials, dtype=float)
    for t in range(n_trials):
        pick = rng.choice(n, size=k, replace=False)
        out[t] = float(np.mean(y_true[pick] == 1))
    return out


def compare_to_m2(y_true: np.ndarray, coverage: float, m2_precision: float,
                  n_trials: int = DEFAULT_TRIALS, seed: int = 42) -> dict:
    """Empirical one-sided test of M2 precision against the random-gate null."""
    dist = random_precision_distribution(y_true, coverage, n_trials, seed)
    if dist.size == 0:
        return {"random_mean": np.nan, "random_std": np.nan, "random_p05": np.nan,
                "random_p95": np.nan, "p_value": np.nan, "n_trials": 0,
                "m2_beats_random_pct": np.nan}
    # +1 smoothing keeps p strictly positive (never claim p = 0 from finite trials).
    p_value = float((np.sum(dist >= m2_precision) + 1) / (dist.size + 1))
    return {
        "random_mean": float(np.mean(dist)), "random_std": float(np.std(dist, ddof=1)),
        "random_p05": float(np.percentile(dist, 5)), "random_p95": float(np.percentile(dist, 95)),
        "p_value": p_value, "n_trials": int(dist.size),
        "m2_beats_random_pct": float(np.mean(dist < m2_precision) * 100.0),
    }


def run_slice(run_dir: Path, asset: str, granularity: str, side: str,
              coverage: float, m2_precision: float, m2_model: str,
              n_trials: int = DEFAULT_TRIALS, seed: int = 42) -> dict:
    """Random-gate control for one slice at a given M2 operating point."""
    sf = load_signal_frame(run_dir, asset, granularity, side)
    _, _, idx_te = split_indices(len(sf), sf.dates)
    test = subset(sf, idx_te)

    mask = test.m1_pred == 1
    if not mask.any():
        raise ValueError("No M1-positive rows in the test split.")
    y_true = test.meta_label[mask]

    stats = compare_to_m2(y_true, coverage, m2_precision, n_trials, seed)
    m1_prec = m1_baseline_precision(test)
    return {
        "method": "Random", "m2_model": m2_model, "asset": asset,
        "granularity": granularity, "side": side,
        "target_coverage": float(coverage), "m2_precision": float(m2_precision),
        "m1_precision": m1_prec, "n_signals": int(y_true.size),
        "delta_m2_vs_random": float(m2_precision) - stats["random_mean"],
        **stats, "skipped_reason": "",
    }


def run_from_m2_results(kronos_root: Path, m2_results: pd.DataFrame,
                        horizon: int = 7, n_trials: int = DEFAULT_TRIALS,
                        seed: int = 42) -> pd.DataFrame:
    """Run the control for every row of an M2 results table.

    ``m2_results`` needs: asset, granularity, side, coverage, precision, and
    optionally m2_model (defaults to 'M2').
    """
    required = {"asset", "granularity", "side", "coverage", "precision"}
    missing = sorted(required.difference(m2_results.columns))
    if missing:
        raise KeyError(f"M2 results table is missing columns: {missing}")

    rows, run_cache = [], {}
    for _, r in m2_results.iterrows():
        gran = str(r["granularity"])
        try:
            if gran not in run_cache:
                run_cache[gran] = discover_run_dir(kronos_root, gran, horizon=horizon)
            rows.append(run_slice(
                run_cache[gran], str(r["asset"]), gran, str(r["side"]),
                float(r["coverage"]), float(r["precision"]),
                str(r.get("m2_model", "M2")), n_trials, seed))
        except Exception as exc:
            rows.append({"method": "Random", "asset": r.get("asset"), "granularity": gran,
                         "side": r.get("side"), "m2_model": str(r.get("m2_model", "M2")),
                         "skipped_reason": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def run_fixed_coverage(kronos_root: Path, granularities, sides, coverage: float,
                       horizon: int = 7, n_trials: int = DEFAULT_TRIALS,
                       seed: int = 42, assets=None) -> pd.DataFrame:
    """Random gate at one fixed coverage across the grid (no M2 table needed)."""
    rows = []
    for gran in granularities:
        try:
            run_dir = discover_run_dir(kronos_root, gran, horizon=horizon)
        except FileNotFoundError as exc:
            rows.append({"method": "Random", "granularity": gran, "skipped_reason": str(exc)})
            continue
        for side in sides:
            for asset in (assets or list_assets(run_dir, side)):
                try:
                    # No M2 reference point → compare against NaN, keep the distribution.
                    rows.append(run_slice(run_dir, asset, gran, side, coverage,
                                          np.nan, "fixed", n_trials, seed))
                except Exception as exc:
                    rows.append({"method": "Random", "asset": asset, "granularity": gran,
                                 "side": side, "skipped_reason": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Random-coverage control gate (Kronos).")
    p.add_argument("--kronos-root", type=Path, default=DEFAULT_KRONOS_ROOT)
    p.add_argument("--m2-results", type=Path, default=None,
                   help="CSV with asset,granularity,side,coverage,precision[,m2_model].")
    p.add_argument("--coverage", type=float, default=None,
                   help="Fixed coverage to test when --m2-results is not given.")
    p.add_argument("--granularities", nargs="+", default=list(GRANULARITIES))
    p.add_argument("--sides", nargs="+", default=list(SIDES))
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.m2_results is not None:
        df = run_from_m2_results(args.kronos_root, pd.read_csv(args.m2_results),
                                 args.horizon, args.n_trials, args.seed)
    elif args.coverage is not None:
        df = run_fixed_coverage(args.kronos_root, args.granularities, args.sides,
                                args.coverage, args.horizon, args.n_trials, args.seed)
    else:
        p.error("Provide either --m2-results or --coverage.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    ok = df[df["skipped_reason"] == ""] if "skipped_reason" in df.columns else df
    print(f"[Random] wrote {args.out}  ({len(ok)} ok / {len(df)} rows)")
    if not ok.empty and "p_value" in ok.columns:
        sig = int((ok["p_value"] < 0.05).sum())
        print(f"  M2 beats random at p<0.05 in {sig}/{len(ok)} slices")


if __name__ == "__main__":
    main()
