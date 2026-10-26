"""Shared plumbing for the M2 baseline gates (Kronos-only).

Every baseline in this package answers the SAME question as an M2 model: given an
M1 signal, should we execute it? Each one produces a per-signal *reliability score*
and executes when the score clears a threshold tau chosen on validation. Only the
scoring rule differs, so the comparison isolates the scoring method.

Data source: the Kronos meta-label CSVs written by the Primary-Model pipeline,

    <KRONOS_ROOT>/pred_<h>_<gran>/<MARKET>/<label_kind>/<run_id>/<ASSET>_<side>.csv

with (at least) the columns ``date, close, prediction, ground_truth, pred, lab,
meta_label``. ``pred`` is M1's directional call (1 = signal fired), ``meta_label``
is the M2 target (1 = M1 was right), ``lab`` the realized direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ┏━━━━━━━━━━ Experiment grid (CIKM/Neurocomputing) ━━━━━━━━━━┓
GRANULARITIES = ("1d", "12h", "8h", "6h", "4h", "2h", "1h", "30m")
SIDES = ("up", "down")

# Chronological split fractions, matching the M2 pipeline convention.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Risk-budget policy: keep selective risk (1 - precision) at or below this on
# validation while maximizing coverage. Mirrors the M2 threshold rule.
RISK_ALPHA = 0.40
MIN_SELECTED = 10

DEFAULT_KRONOS_ROOT = Path(
    "/home/pablo/M2_DS/Primary-Model/src/data/dataset/m2/meta/kronos_predictions"
)


@dataclass
class SignalFrame:
    """One (asset, granularity, side) slice, already restricted to finite rows."""

    asset: str
    granularity: str
    side: str
    frame: pd.DataFrame
    dates: np.ndarray
    close: np.ndarray
    prediction: np.ndarray      # M1 point forecast of the h-step-ahead close
    ground_truth: np.ndarray    # realized h-step-ahead close
    m1_pred: np.ndarray         # 1 where M1 fired a signal on this side
    lab: np.ndarray             # realized direction label
    meta_label: np.ndarray      # M2 target: 1 = M1 was correct

    def __len__(self) -> int:
        return len(self.frame)


# ┏━━━━━━━━━━ Discovery ━━━━━━━━━━┓
def discover_run_dir(kronos_root: Path, granularity: str, horizon: int = 7,
                     market: str = "CRYPTO+CRYPTO", label_kind: str = "close_tp") -> Path:
    """Return the most recent run directory for a granularity.

    Layout: ``pred_<h>_<gran>/<market>/<label_kind>/<run_id>/``. Several run_ids may
    coexist (re-runs); the newest by mtime wins.
    """
    base = Path(kronos_root) / f"pred_{horizon}_{granularity}" / market / label_kind
    if not base.is_dir():
        raise FileNotFoundError(f"No Kronos predictions for granularity '{granularity}': {base}")
    runs = sorted((p for p in base.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run directories under {base}")
    return runs[0]


def list_assets(run_dir: Path, side: str) -> list[str]:
    """Asset symbols that have a CSV for this side, sorted."""
    suffix = f"_{side}.csv"
    return sorted(p.name[: -len(suffix)] for p in run_dir.glob(f"*{suffix}"))


# ┏━━━━━━━━━━ Loading ━━━━━━━━━━┓
REQUIRED_COLUMNS = ("date", "close", "ground_truth", "pred", "lab", "meta_label")


def load_signal_frame(run_dir: Path, asset: str, granularity: str, side: str) -> SignalFrame:
    """Load one CSV and drop rows without a usable label (the warm-up prefix)."""
    csv_path = run_dir / f"{asset}_{side}.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} is missing columns: {missing}")

    # The leading rows carry OHLCV only (no prediction yet) — drop them.
    numeric = ["close", "ground_truth", "pred", "lab", "meta_label"]
    vals = {c: pd.to_numeric(df[c], errors="coerce") for c in numeric}
    finite = np.ones(len(df), dtype=bool)
    for c in numeric:
        finite &= np.isfinite(vals[c].to_numpy(dtype=float))
    clean = df.loc[finite].reset_index(drop=True)
    if clean.empty:
        raise ValueError(f"No finite labelled rows in {csv_path}")

    pred_point = (pd.to_numeric(clean["prediction"], errors="coerce").to_numpy(dtype=float)
                  if "prediction" in clean.columns
                  else np.full(len(clean), np.nan))

    return SignalFrame(
        asset=asset,
        granularity=granularity,
        side=side,
        frame=clean,
        dates=pd.to_datetime(clean["date"]).to_numpy(),
        close=pd.to_numeric(clean["close"], errors="coerce").to_numpy(dtype=float),
        prediction=pred_point,
        ground_truth=pd.to_numeric(clean["ground_truth"], errors="coerce").to_numpy(dtype=float),
        m1_pred=pd.to_numeric(clean["pred"], errors="coerce").to_numpy(dtype=float).astype(int),
        lab=pd.to_numeric(clean["lab"], errors="coerce").to_numpy(dtype=float).astype(int),
        meta_label=pd.to_numeric(clean["meta_label"], errors="coerce").to_numpy(dtype=float).astype(int),
    )


# ┏━━━━━━━━━━ Chronological splits ━━━━━━━━━━┓
def _paper_split_dates() -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """train_end / val_end from the pipeline config, or None if unreadable."""
    import yaml
    cfg_path = Path(__file__).resolve().parents[3] / "config.yaml"
    try:
        sp = yaml.safe_load(cfg_path.read_text())["data"]["split"]
        return pd.Timestamp(sp["train_end"]), pd.Timestamp(sp["val_end"])
    except Exception:
        return None


def split_indices(n_samples: int, dates: np.ndarray | None = None,
                  embargo: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / val / test index arrays.

    When ``dates`` is given the split is made on the SAME calendar boundaries the
    rest of the pipeline uses, so a gate evaluated here is scored on exactly the
    held-out window the paper reports. A proportional 70/15/15 split would put
    each asset on its own window and make the comparison incommensurable with
    every other number in the paper. The proportional split survives only as a
    fallback for callers that cannot supply dates.
    """
    if dates is not None:
        bounds = _paper_split_dates()
        if bounds is not None:
            t_train_end, t_val_end = bounds
            d = pd.to_datetime(pd.Series(np.asarray(dates)))
            tr = np.flatnonzero((d <= t_train_end).to_numpy())
            va = np.flatnonzero(((d > t_train_end) & (d <= t_val_end)).to_numpy())
            te = np.flatnonzero((d > t_val_end).to_numpy())
            # The pipeline embargoes the first `horizon` bars after each split
            # boundary, since a trade opened there resolves inside the previous
            # split. Dropping them here too makes the gate score exactly the
            # trades the paper reports rather than a slightly larger set.
            if embargo > 0:
                va = va[embargo:] if va.size > embargo else va[:0]
                te = te[embargo:] if te.size > embargo else te[:0]
            return tr, va, te
    n_train = int(TRAIN_FRAC * n_samples)
    n_val = int(VAL_FRAC * n_samples)
    return (np.arange(0, n_train, dtype=int),
            np.arange(n_train, n_train + n_val, dtype=int),
            np.arange(n_train + n_val, n_samples, dtype=int))


def subset(sf: SignalFrame, idx: np.ndarray) -> SignalFrame:
    """Row-subset a SignalFrame by positional indices."""
    return SignalFrame(
        asset=sf.asset, granularity=sf.granularity, side=sf.side,
        frame=sf.frame.iloc[idx].reset_index(drop=True),
        dates=sf.dates[idx], close=sf.close[idx], prediction=sf.prediction[idx],
        ground_truth=sf.ground_truth[idx], m1_pred=sf.m1_pred[idx],
        lab=sf.lab[idx], meta_label=sf.meta_label[idx],
    )


# ┏━━━━━━━━━━ Selective metrics ━━━━━━━━━━┓
# Round-trip cost, matching the cash engine and Section "Meta Label Construction".
FEE = 0.002


def trade_returns(close: np.ndarray, ground_truth: np.ndarray, side: str,
                  fee: float = FEE) -> np.ndarray:
    """Net realized return of taking the M1 signal on each row.

    A ``down`` signal is a short, so it earns the fall; an ``up`` signal earns the
    rise. The round-trip cost is charged once per trade, exactly as the cash engine
    charges it, so these numbers are comparable to the returns reported elsewhere.
    """
    c = np.asarray(close, dtype=float)
    g = np.asarray(ground_truth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (g - c) / c
    raw = np.where(np.isfinite(raw), raw, 0.0)
    signed = -raw if str(side).lower().startswith("d") else raw
    return signed - fee


def selective_metrics(y_true: np.ndarray, selected: np.ndarray,
                      returns: np.ndarray | None = None) -> dict:
    """Precision / risk / coverage over the SELECTED subset, and its return.

    ``y_true`` is the meta label (1 = M1 correct) restricted to M1-positive rows;
    ``selected`` is the boolean gate decision on those same rows. Coverage is the
    fraction of M1 signals that survive the gate. When ``returns`` is supplied it
    carries the net realized return of each of those rows, and the mean and total
    over the selected subset are reported alongside.
    """
    n = int(y_true.size)
    k = int(selected.sum())
    base = {"precision": 0.0, "risk": 1.0, "coverage": 0.0,
            "selected_count": 0, "n_signals": n}
    if n == 0 or k == 0:
        if returns is not None:
            base.update({"mean_return": 0.0, "total_return": 0.0})
        return base
    precision = float(np.mean(y_true[selected] == 1))
    out = {"precision": precision, "risk": 1.0 - precision,
           "coverage": k / n, "selected_count": k, "n_signals": n}
    if returns is not None:
        r = np.asarray(returns, dtype=float)[selected]
        r = r[np.isfinite(r)]
        if r.size:
            # mean per trade, and the compounded total over the selected trades
            out["mean_return"] = float(r.mean() * 100.0)
            out["total_return"] = float((np.prod(1.0 + r) - 1.0) * 100.0)
        else:
            out["mean_return"] = 0.0
            out["total_return"] = 0.0
    return out


def pick_threshold(y_val: np.ndarray, scores_val: np.ndarray,
                   risk_alpha: float = RISK_ALPHA,
                   min_selected: int = MIN_SELECTED) -> dict:
    """Choose tau on validation: maximize coverage subject to risk <= risk_alpha.

    Falls back to the risk-minimizing threshold when the budget is unreachable, so
    the gate always returns a usable operating point (flagged via
    ``constraint_satisfied``).
    """
    if y_val.size == 0:
        return {"threshold": -np.inf, "constraint_satisfied": False,
                "val_precision": 0.0, "val_risk": 1.0, "val_coverage": 0.0}

    candidates = np.unique(np.concatenate([scores_val, [-np.inf]]))
    best = None
    fallback = None
    for tau in candidates:
        sel = scores_val >= tau
        if int(sel.sum()) < min_selected:
            continue
        m = selective_metrics(y_val, sel)
        # Best feasible = highest coverage within the risk budget.
        if m["risk"] <= risk_alpha:
            if best is None or m["coverage"] > best[1]["coverage"]:
                best = (tau, m)
        # Fallback = lowest risk regardless of budget.
        if fallback is None or m["risk"] < fallback[1]["risk"]:
            fallback = (tau, m)

    chosen, metrics, satisfied = (*best, True) if best is not None else (
        (*fallback, False) if fallback is not None else (-np.inf, selective_metrics(y_val, np.ones_like(y_val, bool)), False)
    )
    return {"threshold": float(chosen), "constraint_satisfied": bool(satisfied),
            "val_precision": metrics["precision"], "val_risk": metrics["risk"],
            "val_coverage": metrics["coverage"]}


def m1_baseline_precision(sf: SignalFrame) -> float:
    """Precision of taking EVERY M1 signal (the 100%-coverage reference)."""
    mask = sf.m1_pred == 1
    if not mask.any():
        return 0.0
    return float(np.mean(sf.meta_label[mask] == 1))
