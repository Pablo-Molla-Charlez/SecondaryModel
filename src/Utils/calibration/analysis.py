"""Calibration analysis — data collection (no plotting).

Each ``collect_*`` function returns a long-format ``pandas.DataFrame`` keyed by
(m1, m2, direction, granularity, ...). Plotting consumes these DataFrames.

Splits are computed with the exact 4-way embargo split used by
``kronos_tree.temporal_eval`` and ``Utils.edge`` — see
``Utils.ts_cross_validation.compute_embargo_splits``.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from Utils.ts_cross_validation import compute_embargo_splits
from Utils.selective_classification.calibration import calibrate_probabilities, _IdentityCalibrator
from Utils.selective_classification.thresholds import _find_best_utility_threshold


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Paths / defaults
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SRC_ROOT      = Path(__file__).resolve().parents[2]                         # Secondary-Model/src
_DEFAULT_CFG   = _SRC_ROOT / "config.yaml"
_OUTPUT_ROOT   = _SRC_ROOT / "Output"
_SAVE_DIR_DFLT = _OUTPUT_ROOT / "Analysis" / "Calibration"
_CANONICAL_GRANS = ["30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
_M1_MODELS_DEFAULT = ["Kronos", "Fincast", "Chronos2", "Tirex"]
_M2_MODELS_DEFAULT = ["rf", "tabpfn", "tabicl", "autogluon"]
_DIRECTIONS = ["UP", "DOWN"]

_DIR_RE = re.compile(r"_fee_(up|down)_", re.IGNORECASE)


def load_default_config(config_path: Optional[Path] = None) -> dict:
    """Load config.yaml (defaults to Secondary-Model/src/config.yaml)."""
    p = Path(config_path) if config_path else _DEFAULT_CFG
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cache discovery & split computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _discover_caches(m1_models: List[str]) -> Dict[Tuple[str, str], Path]:
    """Return {(m1_name, direction_upper): cache_path} for every .pt found."""
    out: Dict[Tuple[str, str], Path] = {}
    for m1 in m1_models:
        cdir = _OUTPUT_ROOT / m1 / "cache"
        if not cdir.is_dir():
            continue
        for pt in sorted(cdir.glob("*.pt")):
            m = _DIR_RE.search(pt.name)
            if m is None:
                continue
            out.setdefault((m1, m.group(1).upper()), pt)
    return out


def _load_cache(pt: Path):
    try:
        return torch.load(pt, weights_only=False, map_location="cpu")
    except Exception as e:
        warnings.warn(f"[calibration] Failed to load {pt.name}: {e}")
        return None


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def _compute_split_indices(sub: dict, cfg: dict, granularity: str):
    """Return (idx_train, idx_cal, idx_opt, idx_test) on the VALID-filtered timeline.

    Uses ``compute_embargo_splits`` — identical to the training pipeline.
    """
    labels = _to_numpy(sub["labels"]).ravel().astype(float)
    dates_all = sub["dates"]
    valid = ~np.isnan(labels)
    dates_valid = [dates_all[i] for i in range(len(dates_all)) if valid[i]]

    horizon    = int(cfg["data"]["load"]["forecast_horizon"])
    train_end  = cfg["data"]["split"]["train_end"]
    val_end    = cfg["data"]["split"]["val_end"]

    sp = compute_embargo_splits(dates_valid, train_end, val_end, horizon, granularity)
    return (sp["idx_train"], sp["idx_cal"], sp["idx_opt"], sp["idx_test"],
            valid, labels)


def _val_opt_returns(sub: dict, cfg: dict, granularity: str, direction: str):
    """Return (returns_val_opt, y_val_opt) for the Val-Opt window.

    Returns are flipped when direction == 'DOWN' so that positive return == profit.
    """
    idx_train, idx_cal, idx_opt, idx_test, valid, labels = _compute_split_indices(
        sub, cfg, granularity)
    returns_all = _to_numpy(sub["returns"]).ravel()
    returns_valid = returns_all[valid]
    rets_opt = returns_valid[idx_opt].copy()
    if direction.upper() == "DOWN":
        rets_opt = -rets_opt
    y_opt = labels[valid][idx_opt].astype(int)
    return rets_opt, y_opt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Best-probs discovery
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resolve_best_probs_path(m1: str, m2: str, direction: str, granularity: str) -> Optional[Path]:
    """rf/tabpfn/tabicl → HPO path; autogluon → Phase-1 final_model path."""
    dir_u = direction.upper()
    if m2 in ("rf", "tabpfn", "tabicl"):
        p = (_OUTPUT_ROOT / m1 / "HPO" / m2 / dir_u / granularity / "best_probs.npz")
        return p if p.exists() else None
    if m2 == "autogluon":
        p = (_OUTPUT_ROOT / m1 / m2 / dir_u / "Utility_Score"
             / f"{granularity}_training" / "final_model" / "best_probs.npz")
        return p if p.exists() else None
    return None


def _load_best_probs(m1: str, m2: str, direction: str, granularity: str) -> Optional[dict]:
    p = _resolve_best_probs_path(m1, m2, direction, granularity)
    if p is None:
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Study (1) — per-split TP rates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_split_tp_rates(cfg: dict,
                           m1_models: Optional[List[str]] = None) -> pd.DataFrame:
    """TP rate per (m1, direction, granularity, split) using the embargo 4-way split.

    Columns: m1_model, direction, granularity, split, n_tp, n_fp, n_total, tp_rate
    """
    m1_models = m1_models or _M1_MODELS_DEFAULT
    rows: List[dict] = []
    for (m1, direction), pt in _discover_caches(m1_models).items():
        ds = _load_cache(pt)
        if ds is None:
            continue
        grans = getattr(ds, "grans", None) or (list(ds.keys()) if isinstance(ds, dict) else [])
        for g in grans:
            if g not in _CANONICAL_GRANS:
                continue
            sub = ds.sub[g] if hasattr(ds, "sub") else ds[g]
            try:
                idx_train, idx_cal, idx_opt, idx_test, valid, labels = _compute_split_indices(
                    sub, cfg, g)
            except Exception as e:
                warnings.warn(f"[calibration] split failed for {m1}/{direction}/{g}: {e}")
                continue
            lab_valid = labels[valid].astype(int)
            for split_name, idx in [("train", idx_train),
                                     ("val_cal", idx_cal),
                                     ("val_opt", idx_opt),
                                     ("test", idx_test)]:
                if len(idx) == 0:
                    continue
                sub_lab = lab_valid[idx]
                n_tp = int((sub_lab == 1).sum())
                n_fp = int((sub_lab == 0).sum())
                rows.append({"m1_model":    m1,
                             "direction":   direction,
                             "granularity": g,
                             "split":       split_name,
                             "n_tp":        n_tp,
                             "n_fp":        n_fp,
                             "n_total":     n_tp + n_fp,
                             "tp_rate":     n_tp / max(n_tp + n_fp, 1)})
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Study (2) — isotonic degeneracy triggers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _evaluate_degeneracy(raw: np.ndarray,
                         y: np.ndarray,
                         min_unique_out: int = 5,
                         min_range_out: float = 0.10,
                         min_pos_frac_ratio: float = 0.25) -> dict:
    """Fit isotonic on (raw, y) and report the 3 degeneracy triggers independently.

    Mirrors ``calibrate_probabilities`` exactly so output matches production behavior.
    """
    from sklearn.isotonic import IsotonicRegression
    raw = np.asarray(raw, dtype=float)
    y   = np.asarray(y,   dtype=float)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
    cal = iso.predict(raw)
    n_unique = int(np.unique(np.round(cal, 6)).size)
    out_range = float(cal.max() - cal.min()) if cal.size else 0.0
    raw_pos_frac = float((raw >= 0.50).mean()) if raw.size else 0.0
    cal_pos_frac = float((cal >= 0.50).mean()) if cal.size else 0.0
    squashed = (raw_pos_frac >= 0.05 and cal_pos_frac < min_pos_frac_ratio * raw_pos_frac)

    trig_unique = n_unique < min_unique_out
    trig_range  = out_range < min_range_out
    trig_squash = bool(squashed)
    degenerate  = trig_unique or trig_range or trig_squash
    return {"n_unique":       n_unique,
            "out_range":      out_range,
            "raw_pos_frac":   raw_pos_frac,
            "cal_pos_frac":   cal_pos_frac,
            "trig_unique":    trig_unique,
            "trig_range":     trig_range,
            "trig_squash":    trig_squash,
            "degenerate":     bool(degenerate),
            "fell_back_to_identity": bool(degenerate)}


def collect_degeneracy_triggers(m1_models: Optional[List[str]] = None,
                                m2_models: Optional[List[str]] = None) -> pd.DataFrame:
    """Per (m1, m2, direction, gran): which of the 3 degeneracy triggers fired on Val-Cal.

    Uses ``cal_probs_raw`` + ``y_cal`` from ``best_probs.npz``.
    """
    m1_models = m1_models or _M1_MODELS_DEFAULT
    m2_models = m2_models or _M2_MODELS_DEFAULT
    rows: List[dict] = []
    for m1 in m1_models:
        for m2 in m2_models:
            for direction in _DIRECTIONS:
                for g in _CANONICAL_GRANS:
                    bp = _load_best_probs(m1, m2, direction, g)
                    if bp is None:
                        continue
                    cal_raw = bp.get("cal_probs_raw")
                    y_cal   = bp.get("y_cal")
                    if cal_raw is None or y_cal is None or cal_raw.size == 0:
                        continue
                    d = _evaluate_degeneracy(cal_raw, y_cal)
                    rows.append({"m1_model": m1, "m2_model": m2,
                                 "direction": direction, "granularity": g,
                                 **d})
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Studies (3) & (4) — Val-Opt classification + financial, raw vs cal; 4-stage comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_val_opt_metrics(probs: np.ndarray,
                             y:     np.ndarray,
                             returns: np.ndarray,
                             threshold: float,
                             fee: float) -> dict:
    """Classification + financial metrics at a given (probs, threshold) on Val-Opt.

    Classification: accuracy (@τ), winrate (=precision of preds), coverage,
    selective accuracy, selective winrate, selective coverage.
    Financial: mean_net_ret, mean_win_ret, mean_loss_ret  (over selected set).
    """
    probs = np.asarray(probs, dtype=float)
    y     = np.asarray(y, dtype=int)
    returns = np.asarray(returns, dtype=float)
    n = len(y)
    if n == 0:
        return {k: float("nan") for k in
                ("accuracy", "winrate", "coverage",
                 "sel_accuracy", "sel_winrate", "sel_coverage",
                 "mean_net_ret", "mean_win_ret", "mean_loss_ret",
                 "n_selected")}

    sel = probs >= threshold
    preds = sel.astype(int)
    n_sel = int(sel.sum())

    # Classification @τ  (treat selection as positive prediction)
    accuracy = float((preds == y).mean())
    coverage = n_sel / n
    winrate  = float(y[sel].mean()) if n_sel > 0 else float("nan")  # prec of preds

    # Selective (only evaluated where sel == 1)
    sel_coverage = coverage
    sel_accuracy = winrate                                           # on selected set
    sel_winrate  = winrate                                           # synonymous

    # Financial: returns already direction-flipped upstream, fee subtracted here
    net = returns[sel] - fee if n_sel > 0 else np.array([])
    mean_net = float(np.nanmean(net)) if net.size else float("nan")
    wins = net[net > 0] if net.size else np.array([])
    losses = net[net <= 0] if net.size else np.array([])
    mean_win = float(np.nanmean(wins)) if wins.size else float("nan")
    mean_loss = float(np.nanmean(losses)) if losses.size else float("nan")

    return {"accuracy":     accuracy,
            "winrate":      winrate,
            "coverage":     coverage,
            "sel_accuracy": sel_accuracy,
            "sel_winrate":  sel_winrate,
            "sel_coverage": sel_coverage,
            "mean_net_ret": mean_net,
            "mean_win_ret": mean_win,
            "mean_loss_ret": mean_loss,
            "n_selected":   n_sel}


def _m1_baseline_val_opt(sub: dict, cfg: dict, granularity: str) -> dict:
    """M1 accuracy and precision on raw (unfiltered) dates within Val-Opt window."""
    import pandas as _pd
    m1_pred = sub.get("m1_pred_labels") if isinstance(sub, dict) else getattr(sub, "m1_pred_labels", None)
    m1_true = sub.get("m1_true_labels") if isinstance(sub, dict) else getattr(sub, "m1_true_labels", None)
    m1_pred = _to_numpy(m1_pred)
    m1_true = _to_numpy(m1_true)
    dates_all = sub["dates"]
    if m1_pred is None or m1_true is None:
        return {"m1_acc": float("nan"), "m1_prec": float("nan")}

    # Val-Opt window in raw date space (before embargo): between cal_end and val_end minus embargo.
    # Use the same split and then map VALID indices back to raw dates.
    try:
        idx_train, idx_cal, idx_opt, idx_test, valid, labels = _compute_split_indices(sub, cfg, granularity)
    except Exception:
        return {"m1_acc": float("nan"), "m1_prec": float("nan")}
    dates_valid = [dates_all[i] for i in range(len(dates_all)) if valid[i]]
    if len(idx_opt) == 0:
        return {"m1_acc": float("nan"), "m1_prec": float("nan")}
    t_lo = _pd.Timestamp(dates_valid[idx_opt[0]])
    t_hi = _pd.Timestamp(dates_valid[idx_opt[-1]])
    idx_raw = [i for i, d in enumerate(dates_all) if t_lo <= _pd.Timestamp(d) <= t_hi]
    if not idx_raw:
        return {"m1_acc": float("nan"), "m1_prec": float("nan")}

    p = m1_pred[idx_raw].astype(float)
    t = m1_true[idx_raw].astype(float)
    ok = ~np.isnan(p) & ~np.isnan(t)
    if not ok.any():
        return {"m1_acc": float("nan"), "m1_prec": float("nan")}
    p = p[ok].astype(int); t = t[ok].astype(int)
    acc = float((p == t).mean())
    prec = float(t[p == 1].mean()) if (p == 1).any() else float("nan")
    return {"m1_acc": acc, "m1_prec": prec}


def _iter_m1_m2_dir_gran(m1_models, m2_models):
    for m1 in m1_models:
        for m2 in m2_models:
            for d in _DIRECTIONS:
                for g in _CANONICAL_GRANS:
                    yield m1, m2, d, g


def collect_val_opt_comparisons(cfg: dict,
                                m1_models: Optional[List[str]] = None,
                                m2_models: Optional[List[str]] = None) -> pd.DataFrame:
    """Tasks (3) and (4) rolled into one DataFrame.

    For each (m1, m2, direction, gran) we emit 4 rows, one per stage:
      stage ∈ {raw_tau05, cal_tau05, raw_opt, cal_opt}

    Columns: m1_model, m2_model, direction, granularity, stage, threshold,
             threshold_source, threshold_opt_success, m1_acc, m1_prec,
             accuracy, winrate, coverage, sel_accuracy, sel_winrate, sel_coverage,
             mean_net_ret, mean_win_ret, mean_loss_ret, n_selected.
    """
    m1_models = m1_models or _M1_MODELS_DEFAULT
    m2_models = m2_models or _M2_MODELS_DEFAULT
    fee = float(cfg["evaluation"]["fee_per_trade"])

    # Cache per (m1, direction) to avoid re-loading
    cache_map = _discover_caches(m1_models)
    ds_cache: Dict[Tuple[str, str], object] = {}
    def _get_ds(m1, direction):
        k = (m1, direction)
        if k not in ds_cache:
            pt = cache_map.get(k)
            ds_cache[k] = _load_cache(pt) if pt is not None else None
        return ds_cache[k]

    rows: List[dict] = []
    for m1, m2, direction, g in _iter_m1_m2_dir_gran(m1_models, m2_models):
        bp = _load_best_probs(m1, m2, direction, g)
        if bp is None:
            continue
        raw = bp["opt_probs_raw"]; cal = bp["opt_probs_cal"]; y = bp["y_opt"].astype(int)
        ds = _get_ds(m1, direction)
        if ds is None:
            continue
        sub = ds.sub[g] if hasattr(ds, "sub") else (ds[g] if isinstance(ds, dict) and g in ds else None)
        if sub is None:
            continue
        try:
            rets_opt, y_opt_from_cache = _val_opt_returns(sub, cfg, g, direction)
        except Exception as e:
            warnings.warn(f"[calibration] returns fetch failed {m1}/{m2}/{direction}/{g}: {e}")
            continue
        if rets_opt.shape[0] != raw.shape[0]:
            warnings.warn(f"[calibration] length mismatch {m1}/{m2}/{direction}/{g}: "
                          f"returns={rets_opt.shape[0]} probs={raw.shape[0]} — skipping")
            continue

        m1b = _m1_baseline_val_opt(sub, cfg, g)

        # Re-derive true isotonic calibration without the safety fallback.
        # bp["opt_probs_cal"] has the identity fallback baked in for degenerate configs,
        # making cal == raw and the Cal vs No-Cal comparison meaningless.
        # Fit pure isotonic unconditionally on Val-Cal, apply to Val-Opt.
        from sklearn.isotonic import IsotonicRegression
        _iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        _iso.fit(bp["cal_probs_raw"].astype(float), bp["y_cal"].astype(float))
        cal = _iso.predict(raw.astype(float))

        # Sweep τ* on raw and cal independently
        op_raw = _find_best_utility_threshold(raw, rets_opt, fee=fee, labels=y)
        op_cal = _find_best_utility_threshold(cal, rets_opt, fee=fee, labels=y)

        stages = [("raw_tau05", raw, 0.5, "Fixed-0.5", False),
                  ("cal_tau05", cal, 0.5, "Fixed-0.5", False),
                  ("raw_opt",   raw, float(op_raw["threshold"]), op_raw.get("threshold_source", "?"),
                   bool(op_raw.get("constraint_satisfied", False))),
                  ("cal_opt",   cal, float(op_cal["threshold"]), op_cal.get("threshold_source", "?"),
                   bool(op_cal.get("constraint_satisfied", False)))]

        for stage, probs_vec, thr, src, succ in stages:
            met = _compute_val_opt_metrics(probs_vec, y, rets_opt, thr, fee)
            rows.append({"m1_model": m1, "m2_model": m2,
                         "direction": direction, "granularity": g,
                         "stage": stage, "threshold": thr,
                         "threshold_source": src,
                         "threshold_opt_success": succ,
                         "m1_acc":  m1b["m1_acc"],
                         "m1_prec": m1b["m1_prec"],
                         **met})
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Study (5) — threshold-optimization success / failure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_threshold_optimization_stats(cfg: dict,
                                         m1_models: Optional[List[str]] = None,
                                         m2_models: Optional[List[str]] = None) -> pd.DataFrame:
    """Per (m1, m2, direction, gran, probs_variant): which threshold stage was selected.

    probs_variant ∈ {"raw", "cal"}.
    Stages (from ``_find_best_utility_threshold``):
        Utility-Opt | Precision-Coverage | Risk-Fallback | Baseline | Baseline-Override
    "success" == Utility-Opt (constraint_satisfied=True). Anything else is a fallback.
    """
    m1_models = m1_models or _M1_MODELS_DEFAULT
    m2_models = m2_models or _M2_MODELS_DEFAULT
    fee = float(cfg["evaluation"]["fee_per_trade"])
    cache_map = _discover_caches(m1_models)
    ds_cache: Dict[Tuple[str, str], object] = {}
    def _get_ds(m1, direction):
        k = (m1, direction)
        if k not in ds_cache:
            pt = cache_map.get(k)
            ds_cache[k] = _load_cache(pt) if pt is not None else None
        return ds_cache[k]

    rows: List[dict] = []
    for m1, m2, direction, g in _iter_m1_m2_dir_gran(m1_models, m2_models):
        bp = _load_best_probs(m1, m2, direction, g)
        if bp is None:
            continue
        raw = bp["opt_probs_raw"]
        y   = bp["y_opt"].astype(int)
        # Re-derive true isotonic calibration without the safety fallback (same fix as
        # collect_val_opt_comparisons — avoids cal == raw for degenerate configs).
        from sklearn.isotonic import IsotonicRegression
        _iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        _iso.fit(bp["cal_probs_raw"].astype(float), bp["y_cal"].astype(float))
        cal = _iso.predict(raw.astype(float))

        ds = _get_ds(m1, direction)
        if ds is None:
            continue
        sub = ds.sub[g] if hasattr(ds, "sub") else (ds[g] if isinstance(ds, dict) and g in ds else None)
        if sub is None:
            continue
        try:
            rets_opt, _ = _val_opt_returns(sub, cfg, g, direction)
        except Exception:
            continue
        if rets_opt.shape[0] != raw.shape[0]:
            continue

        for variant, probs_vec in (("raw", raw), ("cal", cal)):
            op = _find_best_utility_threshold(probs_vec, rets_opt, fee=fee, labels=y)
            src = op.get("threshold_source", "?")
            rows.append({"m1_model": m1, "m2_model": m2,
                         "direction": direction, "granularity": g,
                         "probs_variant": variant,
                         "threshold":        float(op["threshold"]),
                         "threshold_source": src,
                         "constraint_satisfied": bool(op.get("constraint_satisfied", False)),
                         "coverage":         float(op.get("coverage", 0.0)),
                         "precision":        float(op.get("precision", float("nan"))),
                         "mean_ret":         float(op.get("mean_ret", 0.0)),
                         "utility":          float(op.get("utility", 0.0)),
                         "success":          src == "Utility-Opt"})
    return pd.DataFrame(rows)


__all__ = ["load_default_config",
           "collect_split_tp_rates",
           "collect_degeneracy_triggers",
           "collect_val_opt_comparisons",
           "collect_threshold_optimization_stats"]
