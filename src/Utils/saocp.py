"""Reusable OCP and SAOCP helpers for Kronos M2 analyses."""
import numpy as np
import pandas as pd
from collections import deque
from Utils.utils import seed_everything


# ┏━━━━━━━━━━ OCP Conformity Score ━━━━━━━━━━┓
def _ocp_conformity_score(prob: float, label: int) -> float:
    """Conformity score: 1 - P(true class)."""
    return 1.0 - prob if label == 1 else prob


# ┏━━━━━━━━━━ Run SAOCP Online ━━━━━━━━━━┓
def _run_saocp_online(val_probs,
                      val_labels,
                      test_probs,
                      test_labels,
                      alpha: float          = 0.10,
                      test_dates            = None,
                      forecast_horizon: int = 1,
                      val_dates             = None):
    """Run SAOCP with validation warm-up followed by test-time adaptation."""
    seed_everything(42)
    from online_conformal.saocp import SAOCP

    # ┏━━━━━━━━━━ SAOCP Initialization ━━━━━━━━━━┓
    saocp = SAOCP(model      = None,
                  train_data = None,
                  max_scale  = 1.0,
                  coverage   = 1.0 - alpha,
                  horizon    = 1)

    # ┏━━━━━━━━━━ Validation Thresholds ━━━━━━━━━━┓
    val_thresholds = np.zeros(len(val_probs))
    use_val_delay = (forecast_horizon > 1 and 
                     val_dates is not None and 
                     len(val_dates) == len(val_probs))
    
    # ┏━━━━━━━━━━ Validation Delay ━━━━━━━━━━┓
    if use_val_delay:
        val_unique_dates = sorted(set(val_dates))
        val_date_to_candle = {d: k for k, d in enumerate(val_unique_dates)}
        val_sample_candle = np.array([val_date_to_candle[d] for d in val_dates])

    # ┏━━━━━━━━━━ Validation Buffer ━━━━━━━━━━┓
    val_buffer = deque()
    for i in range(len(val_probs)):
        if use_val_delay:
            current_candle = val_sample_candle[i]
            while val_buffer and val_buffer[0][0] + forecast_horizon <= current_candle:
                # ┏━━━━━━━━━━ Update SAOCP with Delayed Scores ━━━━━━━━━━┓
                _, delayed_score = val_buffer.popleft()
                saocp.update(ground_truth = pd.Series([delayed_score]),
                             forecast     = pd.Series([0.0]),
                             horizon      = 1)

        # ┏━━━━━━━━━━ Predict Threshold ━━━━━━━━━━┓
        _, s_hat = saocp.predict(horizon=1)
        val_thresholds[i] = s_hat
        
        # ┏━━━━━━━━━━ Compute Conformity Score ━━━━━━━━━━┓
        score = _ocp_conformity_score(val_probs[i], int(val_labels[i]))
        
        # ┏━━━━━━━━━━ Update SAOCP with Delayed Scores ━━━━━━━━━━┓
        if use_val_delay:
            val_buffer.append((val_sample_candle[i], score))
        else:
            saocp.update(ground_truth = pd.Series([score]),
                         forecast     = pd.Series([0.0]),
                         horizon      = 1)

    # ┏━━━━━━━━━━ Update SAOCP with Delayed Scores ━━━━━━━━━━┓
    while val_buffer:
        _, delayed_score = val_buffer.popleft()
        saocp.update(ground_truth = pd.Series([delayed_score]),
                     forecast     = pd.Series([0.0]),
                     horizon      = 1)

    # ┏━━━━━━━━━━ Test Delay ━━━━━━━━━━┓
    use_delay = (forecast_horizon > 1
                 and test_dates is not None
                 and len(test_dates) == len(test_probs))
    
    # ┏━━━━━━━━━━ Test Delay ━━━━━━━━━━┓
    if use_delay:
        unique_dates = sorted(set(test_dates))
        date_to_candle = {d: k for k, d in enumerate(unique_dates)}
        sample_candle = np.array([date_to_candle[d] for d in test_dates])

    # ┏━━━━━━━━━━ Test Thresholds Metrics ━━━━━━━━━━┓
    n_test = len(test_probs)
    test_thresholds = np.zeros(n_test)

    # ┏━━━━━━━━━━ Test Approved Metrics ━━━━━━━━━━┓
    test_approved = np.zeros(n_test, dtype=bool)
    test_covered = np.zeros(n_test, dtype=bool)
    pred_sets = np.empty(n_test, dtype=object)

    # ┏━━━━━━━━━━ Test Coverage Metrics ━━━━━━━━━━┓
    n_set_1 = 0
    n_set_0 = 0
    n_set_both = 0
    n_set_empty = 0

    update_buffer = deque()
    for i in range(n_test):
        if use_delay:
            current_candle = sample_candle[i]
            while update_buffer and update_buffer[0][0] + forecast_horizon <= current_candle:
                # ┏━━━━━━━━━━ Update SAOCP with Delayed Scores ━━━━━━━━━━┓
                _, delayed_score = update_buffer.popleft()
                saocp.update(ground_truth = pd.Series([delayed_score]),
                             forecast     = pd.Series([0.0]),
                             horizon      = 1)

        # ┏━━━━━━━━━━ Predict Threshold ━━━━━━━━━━┓
        _, s_hat = saocp.predict(horizon=1)
        test_thresholds[i] = s_hat
        p   = test_probs[i]
        y   = int(test_labels[i])
        tau = max(s_hat, 1.0 - s_hat)

        # ┏━━━━━━━━━━ Test Approved Metrics ━━━━━━━━━━┓
        class1_in = p >= 1.0 - s_hat
        class0_in = p <= s_hat

        # ┏━━━━━━━━━━ Prediction Set & Coverage Metrics ━━━━━━━━━━┓
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

        # ┏━━━━━━━━━━ Test Approved Metrics ━━━━━━━━━━┓
        test_approved[i] = p > tau
        test_covered[i]  = class1_in if y == 1 else class0_in

        # ┏━━━━━━━━━━ Test Coverage Metrics ━━━━━━━━━━┓
        score = _ocp_conformity_score(p, y)
        if use_delay:
            update_buffer.append((sample_candle[i], score))
        else:
            # ┏━━━━━━━━━━ Update SAOCP with Delayed Scores ━━━━━━━━━━┓
            saocp.update(ground_truth = pd.Series([score]),
                         forecast     = pd.Series([0.0]),
                         horizon      = 1)

    # ┏━━━━━━━━━━ Update SAOCP with Delayed Scores ━━━━━━━━━━┓
    while update_buffer:
        _, delayed_score = update_buffer.popleft()
        saocp.update(ground_truth = pd.Series([delayed_score]),
                     forecast     = pd.Series([0.0]),
                     horizon      = 1)

    # ┏━━━━━━━━━━ Conformal Coverage ━━━━━━━━━━┓
    conformal_cov = float(test_covered.mean()) if n_test > 0 else 0.0
    conformal_stats = {"conformal_coverage": conformal_cov,
                       "n_set_1": n_set_1,
                       "n_set_0": n_set_0,
                       "n_set_both": n_set_both,
                       "n_set_empty": n_set_empty,
                       "covered": test_covered,
                       "pred_sets": pred_sets}

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

    # ┏━━━━━━━━━━ Compute Metrics ━━━━━━━━━━┓
    n_total = len(test_labels)
    n_sel = int(test_approved.sum())
    if n_sel > 0:
        net_rets = test_returns[test_approved] - fee
        mu = float(np.nanmean(net_rets))
        sigma = float(np.nanstd(net_rets, ddof=1)) if n_sel > 1 else 0.0
        t_stat = mu / sigma * np.sqrt(n_sel) if sigma > 0 else 0.0
        risk = int((test_labels[test_approved] == 0).sum()) / n_sel
    else:
        mu, t_stat, risk = 0.0, 0.0, 0.0

    # ┏━━━━━━━━━━ Compute Threshold ━━━━━━━━━━┓
    median_thr = float(np.median(test_thresholds))
    median_tau = max(median_thr, 1.0 - median_thr)
    
    # ┏━━━━━━━━━━ Operating Point ━━━━━━━━━━┓
    op = {"threshold":            median_tau,
          "coverage":             n_sel / n_total if n_total > 0 else 0.0,
          "risk":                 risk,
          "selected_count":       n_sel,
          "constraint_satisfied": True,
          "threshold_source":     "OCP-SAOCP",
          "mean_ret":             mu,
          "t_stat":               t_stat}

    # ┏━━━━━━━━━━ Conformal Coverage ━━━━━━━━━━┓
    if conformal_stats is not None:
        op["conformal_coverage"] = conformal_stats["conformal_coverage"]
        op["n_set_1"]            = conformal_stats["n_set_1"]
        op["n_set_0"]            = conformal_stats["n_set_0"]
        op["n_set_both"]         = conformal_stats["n_set_both"]
        op["n_set_empty"]        = conformal_stats["n_set_empty"]
    return op


__all__ = [
    "_ocp_conformity_score",
    "_run_saocp_online",
    "_ocp_threshold_to_op",
]
