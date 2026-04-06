"""Reusable OCP and SAOCP helpers for Kronos M2 analyses.

Includes three threshold-selection modes for online conformal prediction:

  1. "_run_saocp_online"  — Original SAOCP (optionally windowed).
  2. "_run_cost_deferral_online" — Cost-aware deferral (vanilla):
     dynamically re-optimises τ* on a rolling window by minimising
     an explicit expected cost  L(τ) = c_FP·FP + c_FN·FN + c_DEF·DEF.
  3. "_run_cost_deferral_online" with "mondrian=True" — Mondrian
     cost-aware deferral: computes separate τ* per volatility regime
     (low-vol / high-vol) using realized volatility of recent returns.

All functions return the same 4-tuple
    "(test_thresholds, test_approved, val_thresholds, conformal_stats)"
so they are drop-in replacements for one another.
"""
import numpy as np
import pandas as pd
from collections import deque
from Utils.utils import seed_everything


# ┏━━━━━━━━━━ Granularity → candles per day mapping ━━━━━━━━━━┓
_CANDLES_PER_DAY = {"1d": 1, "12h": 2, "8h": 3, "6h": 4, "4h": 6,
                    "2h": 12, "1h": 24, "30m": 48, "15m": 96, "5m": 288}


def _candles_per_day(granularity: str) -> int:
    """Return the number of candles per calendar day for a granularity string."""
    if granularity in _CANDLES_PER_DAY:
        return _CANDLES_PER_DAY[granularity]
    # Fallback: parse the string
    g = granularity.lower().strip()
    if g.endswith("m"):
        return max(1, 1440 // int(g[:-1]))
    if g.endswith("h"):
        return max(1, 24 // int(g[:-1]))
    if g.endswith("d"):
        return max(1, 1 // int(g[:-1]))
    return 1  # unknown → assume daily


def calib_window_for_gran(granularity: str, days: int = 25) -> int:
    """Auto-compute calibration window size (in candles) from granularity.

    Default: 25 calendar days worth of candles.
    """
    return days * _candles_per_day(granularity)


# ┏━━━━━━━━━━ SAOCP factory & feed helpers ━━━━━━━━━━┓
def _make_saocp(alpha: float):
    """Create a fresh SAOCP instance from the online_conformal library."""
    from online_conformal.saocp import SAOCP
    return SAOCP(model      = None, 
                 train_data = None,
                 max_scale  = 1.0, 
                 coverage   = 1.0 - alpha, 
                 horizon    = 1)


# ┏━━━━━━━━━━ SAOCP Update ━━━━━━━━━━┓
def _saocp_feed(saocp, score: float):
    """Feed a single conformity score into the SAOCP."""
    saocp.update(ground_truth=pd.Series([score]),
                 forecast=pd.Series([0.0]),
                 horizon=1)


# ┏━━━━━━━━━━ SAOCP Warm-up ━━━━━━━━━━┓
def _warm_saocp(saocp, scores):
    """Replay a sequence of conformity scores into a (fresh) SAOCP."""
    for s in scores:
        _saocp_feed(saocp, s)


# ═══════════════════════════════════════════════════════════════════════════
# Conformity score
# ═══════════════════════════════════════════════════════════════════════════

def _ocp_conformity_score(prob: float, label: int) -> float:
    """Conformity score: 1 - P(true class)."""
    return 1.0 - prob if label == 1 else prob

# ┏━━━━━━━━━━ Windowed SAOCP ━━━━━━━━━━┓
def _run_saocp_online(val_probs,
                      val_labels,
                      test_probs,
                      test_labels,
                      alpha: float          = 0.10,
                      test_dates            = None,
                      forecast_horizon: int = 1,
                      val_dates             = None,
                      calib_window: int | None = None):
    """Run SAOCP with validation warm-up followed by test-time adaptation.

    Parameters
    ----------
    calib_window : int or None
        If set, limits the calibration history the SAOCP sees:
        * Val warm-up is run on ALL val samples (so val_thresholds are
          identical to the no-window case for comparability), but at the
          transition to test the SAOCP is rebuilt from only the last
          "calib_window" conformity scores.
        * During test time the SAOCP is periodically reset and re-warmed
          from a rolling buffer of the last "calib_window" scores so
          that stale residuals wash out during regime shifts.
        If None (default) the original behaviour is preserved (all history
        accumulates indefinitely).
    """
    seed_everything(42)
    saocp = _make_saocp(alpha)

    # ┏━━━━━━━━━━ Validation warm-up ━━━━━━━━━━┓
    val_thresholds = np.zeros(len(val_probs))
    use_val_delay = (forecast_horizon > 1
                     and val_dates is not None
                     and len(val_dates) == len(val_probs))

    if use_val_delay:
        val_unique_dates = sorted(set(val_dates))
        val_date_to_candle = {d: k for k, d in enumerate(val_unique_dates)}
        val_sample_candle = np.array([val_date_to_candle[d] for d in val_dates])

    # ┏━━━━━━━━━━ Collect val scores for windowed rebuild at transition ━━━━━━━━━━┓
    val_scores: list[float] = []
    val_buffer = deque()
    for i in range(len(val_probs)):
        if use_val_delay:
            current_candle = val_sample_candle[i]
            while val_buffer and val_buffer[0][0] + forecast_horizon <= current_candle:
                _, delayed_score = val_buffer.popleft()
                val_scores.append(delayed_score)
                _saocp_feed(saocp, delayed_score)

        _, s_hat = saocp.predict(horizon=1)
        val_thresholds[i] = s_hat
        score = _ocp_conformity_score(val_probs[i], int(val_labels[i]))
        if use_val_delay:
            val_buffer.append((val_sample_candle[i], score))
        else:
            val_scores.append(score)
            _saocp_feed(saocp, score)

    while val_buffer:
        _, delayed_score = val_buffer.popleft()
        val_scores.append(delayed_score)
        _saocp_feed(saocp, delayed_score)

    # ┏━━━━━━━━━━ Windowed rebuild at val→test transition ━━━━━━━━━━┓
    if calib_window is not None:
        recent = deque(val_scores[-calib_window:], maxlen=calib_window)
        saocp = _make_saocp(alpha)
        _warm_saocp(saocp, recent)
        steps_since_reset = 0
    else:
        recent = None

    # ┏━━━━━━━━━━ Test-time adaptation ━━━━━━━━━━┓
    use_delay = (forecast_horizon > 1
                 and test_dates is not None
                 and len(test_dates) == len(test_probs))

    if use_delay:
        unique_dates = sorted(set(test_dates))
        date_to_candle = {d: k for k, d in enumerate(unique_dates)}
        sample_candle = np.array([date_to_candle[d] for d in test_dates])

    n_test = len(test_probs)
    test_thresholds = np.zeros(n_test)
    test_approved = np.zeros(n_test, dtype=bool)
    test_covered = np.zeros(n_test, dtype=bool)
    pred_sets = np.empty(n_test, dtype=object)
    n_set_1 = n_set_0 = n_set_both = n_set_empty = 0

    def _ingest_score(score: float):
        """Feed a score; if windowed, track + periodic reset."""
        nonlocal saocp, steps_since_reset
        _saocp_feed(saocp, score)
        if recent is not None:
            recent.append(score)
            steps_since_reset += 1
            if steps_since_reset >= calib_window:
                saocp = _make_saocp(alpha)
                _warm_saocp(saocp, recent)
                steps_since_reset = 0

    # Need steps_since_reset in scope even if no windowing
    if recent is None:
        steps_since_reset = 0

    update_buffer = deque()
    for i in range(n_test):
        if use_delay:
            current_candle = sample_candle[i]
            while update_buffer and update_buffer[0][0] + forecast_horizon <= current_candle:
                _, delayed_score = update_buffer.popleft()
                _ingest_score(delayed_score)

        _, s_hat = saocp.predict(horizon=1)
        test_thresholds[i] = s_hat
        p = test_probs[i]
        y = int(test_labels[i])
        tau = max(s_hat, 1.0 - s_hat)

        class1_in = p >= 1.0 - s_hat
        class0_in = p <= s_hat

        if class1_in and not class0_in:
            n_set_1 += 1
            pred_sets[i] = "{1}"
        elif class0_in and not class1_in:
            n_set_0 += 1
            pred_sets[i] = "{0}"
        elif class1_in and class0_in:
            n_set_both += 1
            pred_sets[i] = "{0,1}"
        else:
            n_set_empty += 1
            pred_sets[i] = "{}"

        test_approved[i] = p > tau
        test_covered[i] = class1_in if y == 1 else class0_in

        score = _ocp_conformity_score(p, y)
        if use_delay:
            update_buffer.append((sample_candle[i], score))
        else:
            _ingest_score(score)

    while update_buffer:
        _, delayed_score = update_buffer.popleft()
        _ingest_score(delayed_score)

    conformal_cov = float(test_covered.mean()) if n_test > 0 else 0.0
    conformal_stats = {"conformal_coverage": conformal_cov,
                       "n_set_1":    n_set_1,
                       "n_set_0":    n_set_0,
                       "n_set_both": n_set_both,
                       "n_set_empty": n_set_empty,
                       "covered":    test_covered,
                       "pred_sets":  pred_sets}

    return test_thresholds, test_approved, val_thresholds, conformal_stats


# ┏━━━━━━━━━━ Cost-Aware Deferral (Vanilla + Mondrian) ━━━━━━━━━━┓
_TAU_GRID = np.arange(0.50, 0.96, 0.01)  # search grid for deferral threshold


def _cost_grid_search(probs, labels, c_FP: float, c_FN: float, c_DEF: float,
                      tau_grid=None):
    """Find τ* that minimises empirical expected cost L(τ) on a sample.

    For each candidate τ:
      - defer if max(prob, 1-prob) < τ   (model is uncertain)
      - otherwise the sample is **traded** (approved by the filter)
      - cost = c_DEF  per deferred sample
             + c_FP   per traded sample where label == 0  (losing trade)
             + c_FN   per deferred sample where label == 1 (missed winning trade)

    Every non-deferred sample becomes a trade in the backtest regardless of
    whether prob >= 0.5, so any traded sample with label=0 is a losing trade.

    Returns (best_tau, best_cost, cost_curve).
    """
    if tau_grid is None:
        tau_grid = _TAU_GRID
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(probs)
    if n == 0:
        return 0.5, 0.0, np.zeros(len(tau_grid))

    certainty = np.maximum(probs, 1.0 - probs)  # u(x) = max(p, 1-p)

    best_tau = tau_grid[0]
    best_cost = np.inf
    cost_curve = np.zeros(len(tau_grid))

    for j, tau in enumerate(tau_grid):
        defer_mask = certainty < tau
        trade_mask = ~defer_mask

        cost = (c_DEF * defer_mask.sum()
                + c_FP * (trade_mask & (labels == 0)).sum()
                + c_FN * (defer_mask & (labels == 1)).sum()) / n
        cost_curve[j] = cost
        if cost < best_cost:
            best_cost = cost
            best_tau = tau

    return float(best_tau), float(best_cost), cost_curve


def _realized_volatility(returns, window: int = 20) -> np.ndarray:
    """Rolling realized volatility (std of returns) over a window.

    Returns array of same length with NaN for initial positions.
    """
    ret = np.asarray(returns, dtype=float)
    rv = np.full(len(ret), np.nan)
    for i in range(window, len(ret) + 1):
        rv[i - 1] = np.std(ret[i - window:i], ddof=1)
    return rv


def _run_cost_deferral_online(val_probs,
                              val_labels,
                              test_probs,
                              test_labels,
                              alpha: float          = 0.10,
                              test_dates            = None,
                              forecast_horizon: int = 1,
                              val_dates             = None,
                              calib_window: int     = 50,
                              # Cost-aware parameters
                              c_FP: float           = 10.0,
                              c_FN: float           = 0.0,
                              c_DEF: float          = 2.0,
                              # Mondrian parameters
                              mondrian: bool        = False,
                              test_returns          = None,
                              rv_window: int        = 20):
    """Cost-aware deferral with online SAOCP + dynamic τ* re-optimisation.

    The SAOCP adapts the conformal threshold α_t online (same as
    "_run_saocp_online" with windowing).  On top of that, every
    "calib_window" steps we re-run a grid search over τ on the rolling
    buffer to find the cost-minimising deferral threshold τ*.

    Mondrian variant ("mondrian=True"):
        Splits the rolling buffer into low-vol / high-vol regimes using
        realized volatility of "test_returns", and computes a separate
        τ* per regime.  Requires "test_returns" to be provided.

    Parameters
    ----------
    c_FP, c_FN, c_DEF : float
        Cost weights for false positive, false negative, and deferral.
    mondrian : bool
        If True, use volatility-regime-conditional τ*.
    test_returns : array-like or None
        Per-sample returns aligned with test_probs (required for Mondrian).
    rv_window : int
        Rolling window for realized volatility (default 20 bars).

    Returns
    -------
    Same 4-tuple as "_run_saocp_online".
    """
    seed_everything(42)
    saocp = _make_saocp(alpha)

    # ┏━━━━━━━━━━ Validation warm-up (same as _run_saocp_online) ━━━━━━━━━━┓
    val_thresholds = np.zeros(len(val_probs))
    use_val_delay = (forecast_horizon > 1
                     and val_dates is not None
                     and len(val_dates) == len(val_probs))

    if use_val_delay:
        val_unique_dates = sorted(set(val_dates))
        val_date_to_candle = {d: k for k, d in enumerate(val_unique_dates)}
        val_sample_candle = np.array([val_date_to_candle[d] for d in val_dates])

    val_scores: list[float] = []
    val_buffer_data: list[tuple] = []  # (prob, label, score) for cost search
    val_buffer = deque()
    for i in range(len(val_probs)):
        if use_val_delay:
            current_candle = val_sample_candle[i]
            while val_buffer and val_buffer[0][0] + forecast_horizon <= current_candle:
                _, delayed_score = val_buffer.popleft()
                val_scores.append(delayed_score)
                _saocp_feed(saocp, delayed_score)

        _, s_hat = saocp.predict(horizon=1)
        val_thresholds[i] = s_hat
        score = _ocp_conformity_score(val_probs[i], int(val_labels[i]))
        if use_val_delay:
            val_buffer.append((val_sample_candle[i], score))
        else:
            val_scores.append(score)
            _saocp_feed(saocp, score)
        val_buffer_data.append((float(val_probs[i]), int(val_labels[i]), score))

    while val_buffer:
        _, delayed_score = val_buffer.popleft()
        val_scores.append(delayed_score)
        _saocp_feed(saocp, delayed_score)

    # ┏━━━━━━━━━━ Windowed rebuild + initial τ* from val tail ━━━━━━━━━━┓
    recent_scores = deque(val_scores[-calib_window:], maxlen=calib_window)
    saocp = _make_saocp(alpha)
    _warm_saocp(saocp, recent_scores)
    steps_since_reset = 0

    # ┏━━━━━━━━━━ Rolling buffer of (prob, label) for cost re-optimisation ━━━━━━━━━━┓
    tail = val_buffer_data[-calib_window:]
    recent_pl = deque(tail, maxlen=calib_window)

    # ┏━━━━━━━━━━ Rolling buffer of returns for Mondrian RV (if applicable) ━━━━━━━━━━┓
    recent_returns: deque | None = None
    if mondrian and test_returns is not None:
        recent_returns = deque(maxlen=calib_window)

    # ┏━━━━━━━━━━ Initial τ* from val tail ━━━━━━━━━━┓
    buf_probs = np.array([x[0] for x in recent_pl])
    buf_labels = np.array([x[1] for x in recent_pl])
    current_tau, _, _ = _cost_grid_search(buf_probs, buf_labels, c_FP, c_FN, c_DEF)

    # ┏━━━━━━━━━━ For Mondrian: initial global τ* (no regime info from val returns) ━━━━━━━━━━┓
    tau_low_vol = current_tau
    tau_high_vol = current_tau

    # ┏━━━━━━━━━━ Test-time adaptation ━━━━━━━━━━┓
    use_delay = (forecast_horizon > 1
                 and test_dates is not None
                 and len(test_dates) == len(test_probs))
    if use_delay:
        unique_dates = sorted(set(test_dates))
        date_to_candle = {d: k for k, d in enumerate(unique_dates)}
        sample_candle = np.array([date_to_candle[d] for d in test_dates])

    n_test = len(test_probs)
    test_thresholds = np.zeros(n_test)
    test_approved = np.zeros(n_test, dtype=bool)
    test_covered = np.zeros(n_test, dtype=bool)
    pred_sets = np.empty(n_test, dtype=object)
    tau_trajectory = np.zeros(n_test)
    n_set_1 = n_set_0 = n_set_both = n_set_empty = 0

    # ┏━━━━━━━━━━ Mondrian diagnostics tracking ━━━━━━━━━━┓
    regime_assignments = np.full(n_test, -1, dtype=int)   # 0=low-vol, 1=high-vol, -1=unknown
    tau_low_trajectory = np.full(n_test, np.nan)
    tau_high_trajectory = np.full(n_test, np.nan)
    median_rv_trajectory = np.full(n_test, np.nan)

    # ┏━━━━━━━━━━ Pre-compute RV for the full test set (needed for regime assignment) ━━━━━━━━━━┓
    test_rv: np.ndarray | None = None
    if mondrian and test_returns is not None:
        test_rv = _realized_volatility(np.asarray(test_returns), window=rv_window)

    def _ingest_score(score: float, prob: float, label: int, ret_val: float | None = None):
        """Feed a score into SAOCP and rolling buffers.

        SAOCP is periodically reset every calib_window steps (because the
        library accumulates residuals with no way to forget old ones).
        τ* is re-optimised on every new score (the rolling deque IS the
        true sliding window — no reset needed).
        """
        nonlocal saocp, steps_since_reset
        nonlocal current_tau, tau_low_vol, tau_high_vol

        _saocp_feed(saocp, score)
        recent_scores.append(score)
        recent_pl.append((prob, label, score))
        if recent_returns is not None and ret_val is not None:
            recent_returns.append(ret_val)
        steps_since_reset += 1

        # ┏━━━━━━━━━━ Periodic SAOCP reset (library limitation workaround) ━━━━━━━━━━┓
        if steps_since_reset >= calib_window:
            saocp = _make_saocp(alpha)
            _warm_saocp(saocp, recent_scores)
            steps_since_reset = 0

        # ┏━━━━━━━━━━ Re-optimise τ* on rolling buffer (every new score) ━━━━━━━━━━┓
        buf_p = np.array([x[0] for x in recent_pl])
        buf_l = np.array([x[1] for x in recent_pl])
        current_tau, _, _ = _cost_grid_search(buf_p, buf_l, c_FP, c_FN, c_DEF)

        # ┏━━━━━━━━━━ Mondrian: separate τ* per volatility regime ━━━━━━━━━━┓
        if mondrian and recent_returns is not None and len(recent_returns) >= rv_window:
            rv_buf = _realized_volatility(np.array(recent_returns), window=rv_window)
            valid_rv = rv_buf[~np.isnan(rv_buf)]
            if len(valid_rv) > 0:
                median_rv = np.median(valid_rv)
                n_buf = len(recent_returns)
                pl_tail = list(recent_pl)[-n_buf:]
                rv_tail = rv_buf[-n_buf:]

                low_mask = rv_tail <= median_rv
                high_mask = rv_tail > median_rv

                # ┏━━━━━━━━━━ Low-vol regime ━━━━━━━━━━┓
                if (~np.isnan(rv_tail) & low_mask).sum() >= 10:
                    idx_low = np.where(~np.isnan(rv_tail) & low_mask)[0]
                    p_low = np.array([pl_tail[j][0] for j in idx_low])
                    l_low = np.array([pl_tail[j][1] for j in idx_low])
                    tau_low_vol, _, _ = _cost_grid_search(p_low, l_low, c_FP, c_FN, c_DEF)
                else:
                    tau_low_vol = current_tau

                # ┏━━━━━━━━━━ High-vol regime ━━━━━━━━━━┓
                if (~np.isnan(rv_tail) & high_mask).sum() >= 10:
                    idx_high = np.where(~np.isnan(rv_tail) & high_mask)[0]
                    p_high = np.array([pl_tail[j][0] for j in idx_high])
                    l_high = np.array([pl_tail[j][1] for j in idx_high])
                    tau_high_vol, _, _ = _cost_grid_search(p_high, l_high, c_FP, c_FN, c_DEF)
                else:
                    tau_high_vol = current_tau

    # ┏━━━━━━━━━━ Test-time adaptation ━━━━━━━━━━┓
    update_buffer = deque()
    for i in range(n_test):
        if use_delay:
            current_candle = sample_candle[i]
            while update_buffer and update_buffer[0][0] + forecast_horizon <= current_candle:
                buf_entry = update_buffer.popleft()
                _ingest_score(buf_entry[1], buf_entry[2], buf_entry[3], buf_entry[4])

        _, s_hat = saocp.predict(horizon=1)
        test_thresholds[i] = s_hat
        p = test_probs[i]
        y = int(test_labels[i])

        # ┏━━━━━━━━━━ Conformal prediction sets (same as standard SAOCP) ━━━━━━━━━━┓
        class1_in = p >= 1.0 - s_hat
        class0_in = p <= s_hat

        if class1_in and not class0_in:
            n_set_1 += 1
            pred_sets[i] = "{1}"
        elif class0_in and not class1_in:
            n_set_0 += 1
            pred_sets[i] = "{0}"
        elif class1_in and class0_in:
            n_set_both += 1
            pred_sets[i] = "{0,1}"
        else:
            n_set_empty += 1
            pred_sets[i] = "{}"

        test_covered[i] = class1_in if y == 1 else class0_in

        # ┏━━━━━━━━━━ Cost-aware deferral decision ━━━━━━━━━━┓
        certainty = max(p, 1.0 - p)
        if mondrian and test_rv is not None and not np.isnan(test_rv[i]):
            # ┏━━━━━━━━━━ Determine current regime from pre-computed RV ━━━━━━━━━━┓
            if recent_returns is not None and len(recent_returns) >= rv_window:
                rv_buf = _realized_volatility(np.array(recent_returns), window=rv_window)
                valid_rv = rv_buf[~np.isnan(rv_buf)]
                median_rv = np.median(valid_rv) if len(valid_rv) > 0 else test_rv[i]
            else:
                median_rv = test_rv[i]

            median_rv_trajectory[i] = median_rv
            if test_rv[i] <= median_rv:
                tau_i = tau_low_vol
                regime_assignments[i] = 0
            else:
                tau_i = tau_high_vol
                regime_assignments[i] = 1
        else:
            tau_i = current_tau

        tau_low_trajectory[i] = tau_low_vol
        tau_high_trajectory[i] = tau_high_vol
        tau_trajectory[i] = tau_i
        test_approved[i] = certainty >= tau_i  # NOT deferred

        # ┏━━━━━━━━━━ Compute conformity score and buffer ━━━━━━━━━━┓
        score = _ocp_conformity_score(p, y)
        ret_i = float(test_returns[i]) if test_returns is not None else None

        if use_delay:
            # (candle, score, prob, label, return)
            update_buffer.append((sample_candle[i], score, p, y, ret_i))
        else:
            _ingest_score(score, p, y, ret_i)

    while update_buffer:
        buf_entry = update_buffer.popleft()
        _ingest_score(buf_entry[1], buf_entry[2], buf_entry[3], buf_entry[4])

    # ┏━━━━━━━━━━ Build output ━━━━━━━━━━┓
    conformal_cov = float(test_covered.mean()) if n_test > 0 else 0.0
    conformal_stats = {"conformal_coverage": conformal_cov,
                       "n_set_1":        n_set_1,
                       "n_set_0":        n_set_0,
                       "n_set_both":     n_set_both,
                       "n_set_empty":    n_set_empty,
                       "covered":        test_covered,
                       "pred_sets":      pred_sets,
                       "tau_trajectory": tau_trajectory,
                       "cost_params":    {"c_FP": c_FP, "c_FN": c_FN, "c_DEF": c_DEF},
                       "mondrian":       mondrian}

    # ┏━━━━━━━━━━ Mondrian diagnostics (also for plain OCP-cost to evaluate regime potential) ━━━━━━━━━━┓
    if test_returns is not None:
        if test_rv is None:
            test_rv = _realized_volatility(np.asarray(test_returns), window=rv_window)
        # For non-Mondrian: assign regimes post-hoc using global median RV
        if not mondrian:
            valid_rv = test_rv[~np.isnan(test_rv)]
            if len(valid_rv) > 0:
                global_med = np.median(valid_rv)
                for i in range(n_test):
                    if not np.isnan(test_rv[i]):
                        regime_assignments[i] = 0 if test_rv[i] <= global_med else 1
                        median_rv_trajectory[i] = global_med
        
        # Compute per-regime win rates
        _labels = np.asarray(test_labels, dtype=int)
        _low = regime_assignments == 0
        _high = regime_assignments == 1
        low_wr = float(_labels[_low].mean()) if _low.sum() > 0 else float("nan")
        high_wr = float(_labels[_high].mean()) if _high.sum() > 0 else float("nan")
        overall_wr = float(_labels.mean()) if len(_labels) > 0 else float("nan")

        conformal_stats["mondrian_diag"] = {"test_rv":              test_rv,
                                            "regime_assignments":   regime_assignments,
                                            "tau_low_trajectory":   tau_low_trajectory if mondrian else None,
                                            "tau_high_trajectory":  tau_high_trajectory if mondrian else None,
                                            "median_rv_trajectory": median_rv_trajectory,
                                            "test_labels":          _labels,
                                            "is_mondrian":          mondrian}

        conformal_stats["regime_stats"] = {"n_low_vol":   int(_low.sum()),
                                           "n_high_vol":  int(_high.sum()),
                                           "wr_low_vol":  round(low_wr * 100, 2),
                                           "wr_high_vol": round(high_wr * 100, 2),
                                           "wr_overall":  round(overall_wr * 100, 2),
                                           "delta_wr_pp": round((low_wr - high_wr) * 100, 2) if not (np.isnan(low_wr) or np.isnan(high_wr)) else None}

    return test_thresholds, test_approved, val_thresholds, conformal_stats


# ┏━━━━━━━━━━ Threshold to Operating Point ━━━━━━━━━━┓
def _ocp_threshold_to_op(test_probs,
                         test_labels,
                         test_returns,
                         test_approved,
                         test_thresholds,
                         fee,
                         conformal_stats=None):
    """Build an operating-point dict from adaptive OCP selections."""
    del test_probs

    # ┏━━━━━━━━━━ Build output ━━━━━━━━━━┓
    n_total = len(test_labels)
    n_sel = int(test_approved.sum())
    if n_sel > 0:
        net_rets = test_returns[test_approved] - fee
        mu       = float(np.nanmean(net_rets))
        sigma    = float(np.nanstd(net_rets, ddof=1)) if n_sel > 1 else 0.0
        t_stat   = mu / sigma * np.sqrt(n_sel) if sigma > 0 else 0.0
        risk     = int((test_labels[test_approved] == 0).sum()) / n_sel
    else:
        mu, t_stat, risk = 0.0, 0.0, 0.0

    # ┏━━━━━━━━━━ For cost-deferral: use median of tau_trajectory if available ━━━━━━━━━━┓
    if conformal_stats is not None and "tau_trajectory" in conformal_stats:
        median_tau = float(np.median(conformal_stats["tau_trajectory"]))
    else:
        median_thr = float(np.median(test_thresholds))
        median_tau = max(median_thr, 1.0 - median_thr)

    # ┏━━━━━━━━━━ Determine threshold source label ━━━━━━━━━━┓
    if conformal_stats is not None and conformal_stats.get("mondrian"):
        source = "OCP-CostDeferral-Mondrian"
    elif conformal_stats is not None and "tau_trajectory" in conformal_stats:
        source = "OCP-CostDeferral"
    else:
        source = "OCP-SAOCP"

    # ┏━━━━━━━━━━ Build operating point ━━━━━━━━━━┓
    op = {"threshold":            median_tau,
          "coverage":             n_sel / n_total if n_total > 0 else 0.0,
          "risk":                 risk,
          "selected_count":       n_sel,
          "constraint_satisfied": True,
          "threshold_source":     source,
          "mean_ret":             mu,
          "t_stat":               t_stat}

    if conformal_stats is not None:
        op["conformal_coverage"] = conformal_stats.get("conformal_coverage", 0.0)
        op["n_set_1"]            = conformal_stats.get("n_set_1", 0)
        op["n_set_0"]            = conformal_stats.get("n_set_0", 0)
        op["n_set_both"]         = conformal_stats.get("n_set_both", 0)
        op["n_set_empty"]        = conformal_stats.get("n_set_empty", 0)
        if "cost_params" in conformal_stats:
            op["cost_params"] = conformal_stats["cost_params"]
        if "tau_trajectory" in conformal_stats:
            op["tau_median"] = median_tau
            op["tau_std"] = float(np.std(conformal_stats["tau_trajectory"]))
    return op


def plot_mondrian_diagnostics(conformal_stats, save_dir, gran_label="", thres_mode="OCP-cost-mondrian"):
    """Generate 3 diagnostic plots for Mondrian cost-aware deferral.

    Plots saved to save_dir:
      1. RV time series with median split line
      2. τ* per regime (low-vol vs high-vol) over time
      3. Rolling win rate per regime (does the split separate easy/hard?)

    Parameters
    ----------
    conformal_stats : dict
        Output from _run_cost_deferral_online with mondrian=True.
    save_dir : str or Path
        Directory to save the plot files.
    gran_label : str
        Granularity label for titles (e.g. "4h").
    thres_mode : str
        Threshold mode label for titles.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    diag = conformal_stats.get("mondrian_diag")
    if diag is None:
        return

    # ┏━━━━━━━━━━ Extract Information for realized volatility-based regime analysis ━━━━━━━━━━┓
    save_dir = Path(save_dir)
    test_rv      = diag["test_rv"]
    regimes      = diag["regime_assignments"]
    tau_low_t    = diag["tau_low_trajectory"]    # None for non-Mondrian
    tau_high_t   = diag["tau_high_trajectory"]   # None for non-Mondrian
    is_mondrian  = diag.get("is_mondrian", False)
    median_rv_t  = diag["median_rv_trajectory"]
    labels       = diag["test_labels"]
    tau_traj     = conformal_stats["tau_trajectory"]
    n = len(test_rv)
    idx = np.arange(n)

    cost_p = conformal_stats.get("cost_params", {})
    cost_tag = f"c_FP={cost_p.get('c_FP','?')}, c_DEF={cost_p.get('c_DEF','?')}"

    # ┏━━━━━━━━━━ Plot 1: RV time series with median split ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    valid = ~np.isnan(test_rv)
    ax.plot(idx[valid], test_rv[valid], color="#2980B9", linewidth=0.6, alpha=0.8, label="Realized Volatility")
    valid_med = ~np.isnan(median_rv_t)
    if valid_med.any():
        ax.plot(idx[valid_med], median_rv_t[valid_med], color="#E74C3C", linewidth=1.2,
                linestyle="--", alpha=0.9, label="Rolling median (split)")
    
    # ┏━━━━━━━━━━ Shade Regimes: Low vs High ━━━━━━━━━━┓
    rv_max = float(np.nanmax(test_rv)) if valid.any() else 1.0
    low = regimes == 0
    high = regimes == 1
    if low.any():
        ax.fill_between(idx, 0, rv_max, where=low, alpha=0.08, color="#27AE60", label="Low-vol regime")
    if high.any():
        ax.fill_between(idx, 0, rv_max, where=high, alpha=0.08, color="#E74C3C", label="High-vol regime")
    ax.set_xlabel("Test sample index", fontsize=10)
    ax.set_ylabel("Realized Volatility", fontsize=10)
    ax.set_title(f"Realized Volatility & Regime Assignment  |  {gran_label}  |  {thres_mode}\n{cost_tag}", fontsize=11, fontweight="bold", color="#2C3E50")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ┏━━━━━━━━━━ Save Plot ━━━━━━━━━━┓
    fig.savefig(str(save_dir / "mondrian_rv_regimes.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ┏━━━━━━━━━━ Plot 2: τ* per regime over time (only meaningful for Mondrian) ━━━━━━━━━━┓
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")
    ax.set_facecolor("#FAFAFA")
    if is_mondrian:
        ax.plot(idx, tau_traj, color="#8B008B", linewidth=0.4, alpha=0.25, label="τ* applied (per-sample)")
    else:
        ax.plot(idx, tau_traj, color="#8B008B", linewidth=0.8, alpha=0.7, label="τ* applied")
    if is_mondrian and tau_low_t is not None and tau_high_t is not None:
        valid_low = ~np.isnan(tau_low_t)
        valid_high = ~np.isnan(tau_high_t)
        if valid_high.any():
            ax.plot(idx[valid_high], tau_high_t[valid_high], color="#E74C3C", linewidth=1.2, alpha=0.9, label="τ* high-vol")
        if valid_low.any():
            ax.plot(idx[valid_low], tau_low_t[valid_low], color="#27AE60", linewidth=1.2, alpha=0.9,
                    linestyle="--", label="τ* low-vol")
    ax.axhline(y=0.5, color="#BDC3C7", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Test sample index", fontsize=10)
    ax.set_ylabel("Deferral threshold τ*", fontsize=10)
    title_2 = "Per-Regime τ* Evolution" if is_mondrian else "τ* Evolution (global, no regime split)"
    ax.set_title(f"{title_2}  |  {gran_label}  |  {thres_mode}\n{cost_tag}", fontsize=11, fontweight="bold", color="#2C3E50")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ┏━━━━━━━━━━ Save Plot ━━━━━━━━━━┓
    fig.savefig(str(save_dir / "mondrian_tau_regimes.png"), dpi=200, facecolor="white")
    plt.close(fig)

    # ┏━━━━━━━━━━ Plot 3: Rolling win rate per regime ━━━━━━━━━━┓
    rolling_w = min(200, n // 5) if n > 50 else max(10, n // 3)
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="white")
    ax.set_facecolor("#FAFAFA")

    # ┏━━━━━━━━━━ Overall Rolling Window Win-Rate ━━━━━━━━━━┓
    wins = (labels == 1).astype(float)
    overall_wr = pd.Series(wins).rolling(rolling_w, min_periods=20).mean()
    ax.plot(idx, overall_wr.values, color="#8B008B", linewidth=1.0, alpha=0.6, label=f"Overall WR (roll={rolling_w})")

    # ┏━━━━━━━━━━ Per-regime WR: compute only within regime windows ━━━━━━━━━━┓
    for regime_val, regime_name, color in [(0, "Low-vol", "#27AE60"), (1, "High-vol", "#E74C3C")]:
        mask = regimes == regime_val
        if mask.sum() < 20:
            continue
        # ┏━━━━━━━━━━ Rolling WR within regime samples only ━━━━━━━━━━┓
        regime_wins = np.where(mask, wins, np.nan)
        regime_wr = pd.Series(regime_wins).rolling(rolling_w, min_periods=20).mean()
        ax.plot(idx, regime_wr.values, color=color, linewidth=1.2, alpha=0.9, label=f"{regime_name} WR")

    # ┏━━━━━━━━━━ Summary stats in text box ━━━━━━━━━━┓
    low_mask = regimes == 0
    high_mask = regimes == 1
    low_wr = labels[low_mask].mean() * 100 if low_mask.sum() > 0 else 0
    high_wr = labels[high_mask].mean() * 100 if high_mask.sum() > 0 else 0
    overall_wr_val = labels.mean() * 100
    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())
    stats_text = (f"Low-vol: WR={low_wr:.1f}% (n={n_low})\n"
                  f"High-vol: WR={high_wr:.1f}% (n={n_high})\n"
                  f"Overall: WR={overall_wr_val:.1f}% (n={n})\n"
                  f"Δ WR = {low_wr - high_wr:+.1f}pp")
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=8, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#BDC3C7", alpha=0.9))

    ax.axhline(y=0.5, color="#BDC3C7", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Test sample index", fontsize=10)
    ax.set_ylabel("Win Rate", fontsize=10)
    ax.set_title(f"Win Rate by Volatility Regime  |  {gran_label}  |  {thres_mode}\n"
                 f"{cost_tag}  |  Does the regime split separate easy/hard periods?",
                 fontsize=11, fontweight="bold", color="#2C3E50")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0.2, 0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ┏━━━━━━━━━━ Save Plot ━━━━━━━━━━┓
    fig.savefig(str(save_dir / "mondrian_wr_regimes.png"), dpi=200, facecolor="white")
    plt.close(fig)


__all__ = [
    "_ocp_conformity_score",
    "_run_saocp_online",
    "_run_cost_deferral_online",
    "_ocp_threshold_to_op",
    "calib_window_for_gran",
    "_candles_per_day",
    "plot_mondrian_diagnostics",
]