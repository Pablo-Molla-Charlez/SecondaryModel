"""Isotonic calibration with identity fallback for degenerate cases."""
import numpy as np


# ┏━━━━━━━━━━ Identity calibrator (fallback when isotonic degenerates) ━━━━━━━━━━┓
class _IdentityCalibrator:
    """No-op calibrator: predict(x) = clip(x, 0, 1). Used when isotonic collapses."""
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.clip(np.asarray(X, dtype=float), 0.0, 1.0)


# ┏━━━━━━━━━━ Calibrate Probabilities (Isotonic Regression + fallback) ━━━━━━━━━━┓
def calibrate_probabilities(val_probs: np.ndarray,
                            val_labels: np.ndarray,
                            test_probs: np.ndarray = None,
                            min_unique_out: int = 5,
                            min_range_out: float = 0.10,
                            min_pos_frac_ratio: float = 0.25):
    """Calibrate predicted probabilities using isotonic regression with safety fallback.

    Fits isotonic regression on (val_probs, val_labels). If the fitted calibrator
    is degenerate on its own training probs — fewer than ``min_unique_out`` distinct
    output values, or an output range smaller than ``min_range_out`` — the calibrator
    is replaced with an identity mapping so downstream threshold sweeps remain valid.
    This protects against small Cal sets where isotonic regression collapses all
    probabilities to a single value.

    Returns a dict with calibrated arrays and the fitted calibrator.
    """
    from sklearn.isotonic import IsotonicRegression

    # ┏━━━━━━━━━━ Format Change ━━━━━━━━━━┓
    val_probs  = np.asarray(val_probs, dtype=float)
    val_labels = np.asarray(val_labels, dtype=float)

    # ┏━━━━━━━━━━ Isotonic Regression ━━━━━━━━━━┓
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    # ┏━━━━━━━━━━ Fit Calibrator ━━━━━━━━━━┓
    iso.fit(val_probs, val_labels)

    # ┏━━━━━━━━━━ Predict Calibrated Values ━━━━━━━━━━┓
    val_cal = iso.predict(val_probs)

    # ┏━━━━━━━━━━ Check for Degeneracy ━━━━━━━━━━┓
    # Three failure modes:
    #   (1) too few distinct output values (collapse to constant-ish)
    #   (2) tiny output range
    #   (3) isotonic squashes almost everything below 0.5 while the raw model
    #       had a meaningful positive mass. Concretely, if raw had ≥5% of
    #       samples at p≥0.5 but calibrated retains < min_pos_frac_ratio × that,
    #       the calibrator has destroyed the decision boundary.
    n_unique = int(np.unique(np.round(val_cal, 6)).size)
    out_range = float(val_cal.max() - val_cal.min()) if val_cal.size else 0.0
    raw_pos_frac = float((val_probs >= 0.50).mean()) if val_probs.size else 0.0
    cal_pos_frac = float((val_cal   >= 0.50).mean()) if val_cal.size  else 0.0
    squashed = (raw_pos_frac >= 0.05 and cal_pos_frac < min_pos_frac_ratio * raw_pos_frac)
    degenerate = (n_unique < min_unique_out or out_range < min_range_out or squashed)
    if degenerate:
        reason = ("squash" if squashed else f"unique={n_unique}, range={out_range:.3f}")
        print(f"    [calibrate] WARNING: isotonic degenerated ({reason}, "
              f"raw_pos={raw_pos_frac:.3f}, cal_pos={cal_pos_frac:.3f}) "
              f"— falling back to identity (raw probs).")
        iso = _IdentityCalibrator()
        val_cal = iso.predict(val_probs)

    test_cal = None
    if test_probs is not None:
        test_probs = np.asarray(test_probs, dtype=float)
        test_cal = iso.predict(test_probs)

    return {"val_calibrated": val_cal,
            "test_calibrated": test_cal,
            "calibrator": iso}


__all__ = ["_IdentityCalibrator", "calibrate_probabilities"]
