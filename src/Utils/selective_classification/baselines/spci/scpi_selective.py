"""M1+SPCI — Sequential Predictive Conformal Inference as a reliability gate.

Trimmed from the CTTS baseline runner and retargeted to this project's Kronos
meta-label CSVs (20 crypto assets x 8 granularities x {up, down}).

Unlike M1+PI (which reuses M1's own predictive spread), SPCI fits an INDEPENDENT
second-stage model: it predicts the h-step-ahead close and calibrates conformal
intervals from the sequence of realized residuals, so the interval adapts as recent
errors grow or shrink. Two gate scores are produced:

  reliability      = 1 / (1 + relative interval width)
                     "tighter conformal interval -> more reliable" (CTTS convention).
  directional      = (center - close) / width   for UP signals
                     (close - center) / width   for DOWN signals
                     "the conformal center supports M1's side, measured in units of
                     conformal uncertainty." Positive = agreement.

Both are emitted so either can be reported; the directional score is the more
informative one, since it can DISAGREE with M1 rather than only measure confidence.

Upstream dependency
-------------------
Requires the `hamrel-cxu/SPCI-code` source under ``vendor/SPCI_code/`` (see
README). That directory is intentionally untracked and must be supplied by an
authorized user; this module raises a clear error if it is absent.

Citation: Xu & Xie, "Sequential Predictive Conformal Inference for Time Series",
ICML 2023, PMLR 202:38707-38727.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from Utils.selective_classification.baselines._common import (
    trade_returns,
    DEFAULT_KRONOS_ROOT, GRANULARITIES, SIDES, RISK_ALPHA,
    SignalFrame, discover_run_dir, list_assets, load_signal_frame,
    split_indices, subset, selective_metrics, pick_threshold,
    m1_baseline_precision,
)

# The upstream hamrel-cxu/SPCI-code modules sit next to this file. They import each
# other by bare module name (``import utils_SPCI``, ``import data``), so this directory
# must be on sys.path before SPCI_class is imported.
VENDOR_ROOT = Path(__file__).resolve().parent
_VENDOR_MODULES = ("SPCI_class.py", "utils_SPCI.py", "data.py",
                   "utils_EnbPI.py", "PI_class_EnbPI.py")


def _import_spci():
    """Import the upstream SPCI class, with an actionable error if it is missing."""
    missing = [m for m in _VENDOR_MODULES if not (VENDOR_ROOT / m).is_file()]
    if missing:
        raise RuntimeError(
            f"SPCI upstream source is incomplete in {VENDOR_ROOT}; missing: {missing}. "
            "Restore them from hamrel-cxu/SPCI-code."
        )
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    try:
        from SPCI_class import SPCI_and_EnbPI  # noqa: WPS433 (deferred by design)
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            f"Missing a third-party dependency of SPCI ({exc.name}). Install them with:\n"
            f"    conda run -n CTTS python -m pip install -r {VENDOR_ROOT / 'requirements.txt'}"
        ) from exc
    return SPCI_and_EnbPI


# Feature columns fed to SPCI's regressor. Only those present AND non-empty are used;
# 'close' is mandatory (it anchors the level of the predicted series).
FORECAST_FEATURES = ("close", "prediction", "pred_q10", "pred_q90", "pred_std", "pred")


@contextlib.contextmanager
def _scalar_quantile_predictions():
    """Make ``RandomForestQuantileRegressor.predict`` return 1-D quantile vectors.

    Upstream ``compute_PIs_Ensemble_online`` does ``width_left[i] = curr_SigmaX *
    wid_all[i_star]``, assuming ``wid_all`` is one value per quantile. sklearn-quantile
    actually returns shape ``(n_quantiles, n_samples)``; since SPCI predicts a single
    residual row at a time, ``wid_all[i_star]`` is a length-1 ARRAY and the assignment
    raises "setting an array element with a sequence".

    Squeezing the sample axis when it is 1 restores the shape upstream expects. The
    vendored source is left untouched so it stays a faithful copy; the patch is scoped
    to this call and reverted on exit.
    """
    from sklearn_quantile import RandomForestQuantileRegressor

    original = RandomForestQuantileRegressor.predict

    def _patched(self, X, *a, **kw):
        out = np.asarray(original(self, X, *a, **kw))
        # (n_quantiles, 1) -> (n_quantiles,); leave genuine multi-row output alone.
        if out.ndim == 2 and out.shape[1] == 1:
            return out[:, 0]
        return out

    RandomForestQuantileRegressor.predict = _patched
    try:
        yield
    finally:
        RandomForestQuantileRegressor.predict = original


@dataclass
class SPCIIntervals:
    center: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    rel_width: np.ndarray
    reliability: np.ndarray
    interval_coverage: float
    mean_rel_width: float


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in FORECAST_FEATURES
            if c in frame.columns and pd.to_numeric(frame[c], errors="coerce").notna().any()]
    if "close" not in cols:
        raise ValueError("Column 'close' must be usable as an SPCI feature.")
    return cols


def prepare_feature_matrices(train_df: pd.DataFrame, pred_df: pd.DataFrame,
                             feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute then standardize on TRAIN statistics (no test leakage)."""
    train = train_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    pred = pred_df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
    medians = train.median(axis=0, skipna=True).fillna(0.0)
    train, pred = train.fillna(medians), pred.fillna(medians)
    mean = train.mean(axis=0)
    std = train.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    x_train = np.nan_to_num(((train - mean) / std).to_numpy(dtype=np.float32))
    x_pred = np.nan_to_num(((pred - mean) / std).to_numpy(dtype=np.float32))
    return x_train, x_pred


def fit_spci_intervals(train: SignalFrame, predict: SignalFrame,
                       feature_cols: list[str], args: argparse.Namespace,
                       seed: int = 42) -> SPCIIntervals:
    """Fit SPCI on ``train`` and produce conformal intervals over ``predict``."""
    import torch

    SPCI_and_EnbPI = _import_spci()

    if len(train) < 20:
        raise ValueError("SPCI requires at least 20 finite training rows.")
    if len(predict) == 0:
        raise ValueError("SPCI prediction split is empty.")
    past_window = min(int(args.past_window), len(train) - 2)
    if past_window < 10:
        raise ValueError(f"SPCI past_window too small after split adjustment: {past_window}.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    x_train, x_pred = prepare_feature_matrices(train.frame, predict.frame, feature_cols)
    spci = SPCI_and_EnbPI(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(x_pred, dtype=torch.float32),
        torch.tensor(train.ground_truth.astype(np.float32), dtype=torch.float32),
        torch.tensor(predict.ground_truth.astype(np.float32), dtype=torch.float32),
        fit_func=LinearRegression(),
    )
    spci.n_estimators = int(args.qrf_trees)
    spci.max_d = int(args.qrf_depth)
    spci.bins = int(args.qrf_bins)
    spci.criterion = "squared_error"
    spci.T1 = None if int(args.qrf_train_window) == 0 else int(args.qrf_train_window)

    stream = io.StringIO()
    ctx = contextlib.nullcontext() if args.verbose_spci else contextlib.redirect_stdout(stream)
    with ctx, _scalar_quantile_predictions():
        spci.fit_bootstrap_models_online_multistep(B=int(args.bootstrap_models),
                                                   fit_sigmaX=False, stride=1)
        spci.compute_PIs_Ensemble_online(alpha=float(args.conformal_alpha), stride=1,
                                         smallT=False, past_window=past_window,
                                         use_SPCI=True)

    pi = spci.PIs_Ensemble.copy()
    lower = pd.to_numeric(pi["lower"], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(pi["upper"], errors="coerce").to_numpy(dtype=float)
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    center = np.asarray(spci.Ensemble_pred_interval_centers, dtype=float)
    width = np.maximum(upper - lower, 0.0)
    rel_width = width / np.maximum(np.abs(predict.close), 1e-8)
    reliability = np.nan_to_num(1.0 / (1.0 + rel_width), nan=0.0, posinf=1.0, neginf=0.0)
    covered = (lower <= predict.ground_truth) & (predict.ground_truth <= upper)

    return SPCIIntervals(
        center=center, lower=lower, upper=upper, width=width, rel_width=rel_width,
        reliability=reliability,
        interval_coverage=float(np.mean(covered)) if covered.size else np.nan,
        mean_rel_width=float(np.mean(rel_width)) if rel_width.size else np.nan,
    )


def directional_margin_score(side: str, center: np.ndarray, close: np.ndarray,
                             width: np.ndarray) -> np.ndarray:
    """Positive when the conformal center supports the M1 side, scaled by width."""
    denom = np.maximum(np.asarray(width, dtype=float), 1e-8)
    center, close = np.asarray(center, dtype=float), np.asarray(close, dtype=float)
    score = (center - close) / denom if side.lower() == "up" else (close - center) / denom
    return np.nan_to_num(score, nan=-np.inf, posinf=np.inf, neginf=-np.inf)


def _concat(a: SignalFrame, b: SignalFrame) -> SignalFrame:
    """Chronological concatenation (train + val) for the final test-time refit."""
    return SignalFrame(
        asset=a.asset, granularity=a.granularity, side=a.side,
        frame=pd.concat([a.frame, b.frame], ignore_index=True),
        dates=np.concatenate([a.dates, b.dates]),
        close=np.concatenate([a.close, b.close]),
        prediction=np.concatenate([a.prediction, b.prediction]),
        ground_truth=np.concatenate([a.ground_truth, b.ground_truth]),
        m1_pred=np.concatenate([a.m1_pred, b.m1_pred]),
        lab=np.concatenate([a.lab, b.lab]),
        meta_label=np.concatenate([a.meta_label, b.meta_label]),
    )


def run_slice(run_dir: Path, asset: str, granularity: str, side: str,
              args: argparse.Namespace, score_kind: str = "directional") -> dict:
    """Evaluate the M1+SPCI gate on one (asset, granularity, side) slice."""
    sf = load_signal_frame(run_dir, asset, granularity, side)
    feature_cols = _feature_columns(sf.frame)
    idx_tr, idx_va, idx_te = split_indices(len(sf), sf.dates)
    train, val, test = subset(sf, idx_tr), subset(sf, idx_va), subset(sf, idx_te)

    va_mask, te_mask = val.m1_pred == 1, test.m1_pred == 1
    if not va_mask.any() or not te_mask.any():
        raise ValueError("No M1-positive rows in validation or test split.")

    # tau is chosen with intervals fitted on TRAIN only; the test intervals are
    # refit on TRAIN+VAL, so no test information reaches the threshold.
    val_iv = fit_spci_intervals(train, val, feature_cols, args, args.seed)
    test_iv = fit_spci_intervals(_concat(train, val), test, feature_cols, args, args.seed)

    def _scores(iv, split):
        return (directional_margin_score(side, iv.center, split.close, iv.width)
                if score_kind == "directional" else iv.reliability)

    va_scores = _scores(val_iv, val)[va_mask]
    te_scores = _scores(test_iv, test)[te_mask]
    va_y, te_y = val.meta_label[va_mask], test.meta_label[te_mask]

    sel = pick_threshold(va_y, va_scores, risk_alpha=args.risk_alpha)
    te_ret = trade_returns(test.close, test.ground_truth, side)[te_mask]
    metrics = selective_metrics(te_y, te_scores >= sel["threshold"], returns=te_ret)
    m1_prec = m1_baseline_precision(test)

    return {
        "method": f"M1+SPCI({score_kind})", "asset": asset, "granularity": granularity,
        "side": side, "threshold": sel["threshold"],
        "constraint_satisfied": sel["constraint_satisfied"],
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
        "conformal_interval_coverage": test_iv.interval_coverage,
        "mean_rel_width": test_iv.mean_rel_width, "skipped_reason": "",
    }


def run_all(kronos_root: Path, granularities, sides, args: argparse.Namespace,
            score_kind: str = "directional", assets=None) -> pd.DataFrame:
    rows = []
    for gran in granularities:
        try:
            run_dir = discover_run_dir(kronos_root, gran, horizon=args.horizon)
        except FileNotFoundError as exc:
            rows.append({"method": f"M1+SPCI({score_kind})", "granularity": gran,
                         "skipped_reason": str(exc)})
            continue
        for side in sides:
            for asset in (assets or list_assets(run_dir, side)):
                try:
                    rows.append(run_slice(run_dir, asset, gran, side, args, score_kind))
                except Exception as exc:
                    rows.append({"method": f"M1+SPCI({score_kind})", "asset": asset,
                                 "granularity": gran, "side": side,
                                 "skipped_reason": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1+SPCI conformal reliability gate (Kronos).")
    p.add_argument("--kronos-root", type=Path, default=DEFAULT_KRONOS_ROOT)
    p.add_argument("--granularities", nargs="+", default=list(GRANULARITIES))
    p.add_argument("--sides", nargs="+", default=list(SIDES))
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--score-kind", choices=["directional", "reliability"], default="directional")
    p.add_argument("--risk-alpha", type=float, default=RISK_ALPHA)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    # Upstream SPCI fitting parameters (defaults follow the CTTS baseline runner).
    p.add_argument("--conformal-alpha", type=float, default=0.20)
    p.add_argument("--bootstrap-models", type=int, default=10)
    p.add_argument("--past-window", type=int, default=100)
    p.add_argument("--qrf-trees", type=int, default=10)
    p.add_argument("--qrf-depth", type=int, default=2)
    p.add_argument("--qrf-bins", type=int, default=5)
    p.add_argument("--qrf-train-window", type=int, default=300)
    p.add_argument("--verbose-spci", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = run_all(args.kronos_root, args.granularities, args.sides, args, args.score_kind)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    ok = df[df["skipped_reason"] == ""] if "skipped_reason" in df.columns else df
    print(f"[M1+SPCI] wrote {args.out}  ({len(ok)} ok / {len(df)} rows)")
    if not ok.empty:
        print(ok.groupby("granularity")[["precision", "coverage"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
