"""Modified-BIC statistical hyperparameter selection for Jump Model regimes.

The selector scores Jump Models fitted on the BIC fit window (Train+Val; the
Test split is held out) with the modified BIC used for regime detection:

    BIC = (2 * (L_sat - L_candidate) + M * log(T)) / T

where ``L`` is the between-cluster sum of squares (BCSS), ``T`` is the number
of fit-window observations, and ``M`` is the model-family complexity penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from Utils.regime.model import JM_VARIANTS, FitArtifacts, is_sparse_variant

# A fitted candidate's scoring inputs: the typed FitArtifacts from
# RegimeJumpModel.fit_artifacts, or any mapping with the same keys (kept for
# lightweight test fixtures and external callers).
Artifacts = FitArtifacts | Mapping[str, object]

DEFAULT_K_GRID = (2, 3, 4, 5)
DEFAULT_LAMBDA_GRID = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
DEFAULT_BIC_K0 = 2
DEFAULT_BIC_A0 = None
DEFAULT_BIC_KBAR = 6
DEFAULT_BIC_ACTIVE_THRESHOLD = 1e-8


@dataclass
class BicScore:
    """Modified-BIC diagnostics for one fitted hyperparameter candidate."""

    variant: str
    n_components: int
    jump_penalty: float
    bic: float
    candidate_bcss: float
    saturated_bcss: float
    complexity: float
    n_jumps: int
    regime_counts: dict[int, int]
    average_regime_duration: float
    active_features: int
    n_features: int
    n_obs: int
    # Modified-BIC reference settings used for this candidate (k0/a0 reference
    # state & active-feature counts, and the active-feature weight threshold).
    bic_k0: int = DEFAULT_BIC_K0
    bic_a0: int | None = DEFAULT_BIC_A0
    bic_active_threshold: float = DEFAULT_BIC_ACTIVE_THRESHOLD
    # Sparse-JM max_feats target for this candidate (None for standard/
    # continuous); named "kappa" per the modified-BIC paper.
    kappa: float | None = None
    per_feature_bcss: dict[str, float] = field(default_factory=dict)

    @property
    def diagnostics(self) -> dict[str, float]:
        """Flat metric mapping derived from the fields above (kept for back-compat
        and meta.json serialization, so the numbers live in exactly one place)."""
        return {
            "bic": self.bic,
            "candidate_bcss": self.candidate_bcss,
            "saturated_bcss": self.saturated_bcss,
            "complexity": self.complexity,
            "n_jumps": self.n_jumps,
            "active_features": self.active_features,
            "n_features": self.n_features,
            "n_obs": self.n_obs,
            "K": self.n_components,
            "K0": self.bic_k0,
            "A0": self.bic_a0,
            "active_threshold": self.bic_active_threshold,
            "average_regime_duration": self.average_regime_duration,
        }

    def to_meta_dict(self) -> dict[str, Any]:
        """Serialize one candidate for the ``bic_scores`` array in meta.json.

        Shared by the per-asset and global trainers so the persisted schema lives
        in exactly one place.
        """
        return {
            "variant": self.variant,
            "K": self.n_components,
            "lambda": self.jump_penalty,
            "kappa": self.kappa,
            "bic": self.bic,
            "candidate_bcss": self.candidate_bcss,
            "saturated_bcss": self.saturated_bcss,
            "complexity": self.complexity,
            "n_jumps": self.n_jumps,
            "regime_counts": self.regime_counts,
            "average_regime_duration": self.average_regime_duration,
            "active_features": self.active_features,
            "n_features": self.n_features,
            "n_obs": self.n_obs,
            "diagnostics": self.diagnostics,
        }


class SparseBicReferenceError(ValueError):
    """Raised when a sparse BIC candidate is below the K0/A0 reference model."""

    def __init__(self,
                 message: str,
                 *,
                 reference_name: str,
                 candidate_value: int,
                 reference_value: int) -> None:
        super().__init__(message)
        self.reference_name = reference_name
        self.candidate_value = int(candidate_value)
        self.reference_value = int(reference_value)


class BicScoreList(list):
    """List of BIC scores plus grid-selection diagnostics."""

    def __init__(self,
                 scores: Sequence[BicScore] = (),
                 *,
                 skipped_reference_candidates: int = 0,
                 total_candidates: int = 0) -> None:
        super().__init__(scores)
        self.skipped_reference_candidates = int(skipped_reference_candidates)
        self.total_candidates = int(total_candidates)


def default_kappa_grid(n_features: int, size: int = 8) -> tuple[float, ...]:
    """Default sparse-JM kappa grid on [1, sqrt(P)]."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if size <= 0:
        raise ValueError("kappa grid size must be positive")
    upper = float(np.sqrt(n_features))
    if size == 1 or np.isclose(upper, 1.0):
        return (1.0,)
    return tuple(float(x) for x in np.linspace(1.0, upper, int(size)))


def default_bic_a0(n_features: int) -> int:
    """Default sparse-BIC reference active-feature count from feature width P."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    return int(max(2, min(int(n_features), round(0.25 * int(n_features)))))


def count_jumps(labels: pd.Series | np.ndarray,
                groups: pd.Series | np.ndarray | None = None) -> int:
    """Number of state transitions in a decoded state sequence."""
    ser = pd.Series(labels).dropna()
    arr = ser.to_numpy()
    if arr.size <= 1:
        return 0
    if groups is None:
        return int(np.sum(arr[1:] != arr[:-1]))
    label_index = pd.Series(labels).index
    if isinstance(groups, pd.Series):
        group_ser = groups.reindex(label_index)
    else:
        group_ser = pd.Series(groups, index=label_index)
    group_ser = group_ser.reindex(ser.index)
    group_arr = group_ser.to_numpy()
    same_group = group_arr[1:] == group_arr[:-1]
    return int(np.sum((arr[1:] != arr[:-1]) & same_group))


def active_feature_count(feature_weights: pd.Series | np.ndarray,
                         threshold: float = DEFAULT_BIC_ACTIVE_THRESHOLD) -> int:
    """Count nonzero sparse weights under the active-feature threshold."""
    weights = np.asarray(feature_weights, dtype=float)
    return int(np.sum(np.abs(weights) > float(threshold)))


def _normalize_proba_rows(proba: pd.DataFrame | np.ndarray,
                          index: pd.Index) -> pd.DataFrame:
    """Row-normalize a probability frame so each bar's regime probs sum to 1."""
    out = pd.DataFrame(proba, index=index).astype(float)
    if out.empty:
        raise ValueError("Cannot compute BCSS with empty probabilities")
    row_sum = out.sum(axis=1).replace(0.0, np.nan)
    return out.div(row_sum, axis=0).fillna(0.0)


def bcss_by_feature(X_scaled: pd.DataFrame,
                    proba: pd.DataFrame | np.ndarray) -> pd.Series:
    """Between-cluster sum of squares per feature."""
    X = pd.DataFrame(X_scaled).astype(float)
    if X.empty:
        raise ValueError("Cannot compute BCSS for empty features")
    P = _normalize_proba_rows(proba, X.index)
    if len(P) != len(X):
        raise ValueError("features and probabilities must have the same number of rows")

    x_arr = X.to_numpy(dtype=float)
    p_arr = P.to_numpy(dtype=float)
    counts = p_arr.sum(axis=0)
    centers = np.zeros((p_arr.shape[1], x_arr.shape[1]), dtype=float)
    valid = counts > 0
    centers[valid] = (p_arr[:, valid].T @ x_arr) / counts[valid, None]
    mean = x_arr.mean(axis=0)
    bcss = counts @ ((centers - mean) ** 2)
    bcss[np.abs(bcss) <= 1e-12] = 0.0
    return pd.Series(bcss, index=X.columns, dtype=float)


def artifact_bcss(artifacts: Artifacts) -> float:
    """BCSS for a fitted candidate's artifacts.

    Standard and continuous models use all features with unit weights. Sparse
    models pass their learned SJM weights, so inactive features do not contribute
    to ``L_candidate``.
    """
    art = FitArtifacts.from_mapping(artifacts)
    X_scaled = pd.DataFrame(art.X_scaled)
    weights = pd.Series(art.feature_weights, index=X_scaled.columns, dtype=float)
    per_feature = bcss_by_feature(X_scaled, art.proba)
    return float((per_feature * weights.reindex(per_feature.index).fillna(0.0)).sum())


def _bic_penalty_terms(*,
                       variant: str,
                       n_features: int,
                       n_components: int,
                       n_jumps: int,
                       k0: int,
                       a0: int | None,
                       feature_weights: pd.Series,
                       active_threshold: float) -> tuple[int, int, float]:
    """Active-feature reference, active-feature count, and the BIC complexity term.

    Sparse variants penalize the ``(k0, a0)`` reference cells plus the extra
    regimes and active features; dense variants use the full ``K * P`` parameter
    count. Both add the jump count ``J``. Raises ``SparseBicReferenceError`` when a
    sparse candidate falls below its ``(k0, a0)`` reference.
    """
    if not is_sparse_variant(variant):
        return int(n_features), int(n_features), float(n_components * n_features + n_jumps)

    a0_ref = default_bic_a0(n_features) if a0 is None else int(a0)
    if a0_ref <= 0:
        raise ValueError("a0 must be positive")
    active = active_feature_count(feature_weights, active_threshold)
    if n_components < k0:
        raise SparseBicReferenceError(
            f"sparse BIC candidate K={n_components} is below reference k0={k0}",
            reference_name="k0",
            candidate_value=n_components,
            reference_value=int(k0),
        )
    if active < a0_ref:
        raise SparseBicReferenceError(
            f"sparse BIC candidate active_features={active} is below reference a0={a0_ref}",
            reference_name="a0",
            candidate_value=active,
            reference_value=int(a0_ref),
        )
    complexity = float(k0 * a0_ref + a0_ref * (n_components - k0) + k0 * (active - a0_ref) + n_jumps)
    return a0_ref, active, complexity


def modified_bic_score(candidate_artifacts: Artifacts,
                       *,
                       saturated_bcss: float,
                       variant: str,
                       k0: int = DEFAULT_BIC_K0,
                       a0: int | None = DEFAULT_BIC_A0,
                       active_threshold: float = DEFAULT_BIC_ACTIVE_THRESHOLD) -> BicScore:
    """Compute the modified BIC for one fitted JM/CJM/SJM candidate."""
    if variant not in JM_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choices: {JM_VARIANTS}")
    if k0 <= 0:
        raise ValueError("k0 must be positive")

    art = FitArtifacts.from_mapping(candidate_artifacts)
    X_scaled = pd.DataFrame(art.X_scaled)
    labels = pd.Series(art.labels).reindex(X_scaled.index).dropna().astype(int)
    if X_scaled.empty or labels.empty:
        raise ValueError("BIC requires non-empty train features and labels")

    T = int(len(X_scaled))
    P = int(X_scaled.shape[1])
    K = int(art.n_components if art.n_components is not None else labels.nunique())
    feature_weights = pd.Series(art.feature_weights, index=X_scaled.columns, dtype=float)
    candidate_per_feature = bcss_by_feature(X_scaled, art.proba)
    candidate_bcss = float((candidate_per_feature * feature_weights).sum())
    J = count_jumps(labels, art.groups)
    counts = {int(k): int(v) for k, v in labels.value_counts().sort_index().items()}

    A0, A, complexity = _bic_penalty_terms(
        variant=variant,
        n_features=P,
        n_components=K,
        n_jumps=J,
        k0=k0,
        a0=a0,
        feature_weights=feature_weights,
        active_threshold=active_threshold,
    )

    bic = float((2.0 * (float(saturated_bcss) - candidate_bcss) + complexity * np.log(T)) / T)
    avg_duration = float(T / max(J + 1, 1))
    return BicScore(
        variant=variant,
        n_components=K,
        jump_penalty=float(art.jump_penalty if art.jump_penalty is not None else np.nan),
        kappa=(float(art.max_feats) if art.max_feats is not None else None),
        bic=bic,
        candidate_bcss=candidate_bcss,
        saturated_bcss=float(saturated_bcss),
        complexity=complexity,
        n_jumps=int(J),
        regime_counts=counts,
        average_regime_duration=avg_duration,
        active_features=int(A),
        n_features=int(P),
        n_obs=int(T),
        bic_k0=int(k0),
        bic_a0=int(A0),
        bic_active_threshold=float(active_threshold),
        per_feature_bcss={str(k): float(v) for k, v in candidate_per_feature.items()},
    )


def _resolve_bic_kappas(*,
                        variant: str,
                        kappa_grid: Sequence[float] | None,
                        k_grid: Sequence[int],
                        k0: int,
                        a0: int | None) -> tuple[float | None, ...]:
    """Validate the BIC grid request and return the kappa values to sweep.

    Dense variants sweep a single ``None`` kappa; sparse variants sweep the
    supplied ``kappa_grid`` and must satisfy the ``(k0, a0)`` reference bounds.
    """
    if variant not in JM_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choices: {JM_VARIANTS}")
    if not is_sparse_variant(variant):
        if kappa_grid is not None:
            raise ValueError("kappa_grid is only valid for the sparse variants")
        return (None,)

    kappas = tuple(kappa_grid) if kappa_grid is not None else ()
    if not kappas:
        raise ValueError("the sparse variants require at least one kappa value")
    if k0 <= 0:
        raise ValueError("k0 must be positive")
    if a0 is not None and int(a0) <= 0:
        raise ValueError("a0 must be positive")
    min_k = min(int(k) for k in k_grid)
    if min_k < int(k0):
        raise ValueError(f"k_grid contains K={min_k} below sparse BIC reference k0={k0}")
    return kappas


def select_bic_hyperparameters(
    fit_candidate_fn: Callable[[int, float, float | None], Artifacts],
    *,
    variant: str,
    saturated_bcss: float,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    kappa_grid: Sequence[float] | None = None,
    k0: int = DEFAULT_BIC_K0,
    a0: int | None = DEFAULT_BIC_A0,
    active_threshold: float = DEFAULT_BIC_ACTIVE_THRESHOLD,
) -> BicScoreList:
    """Fit and score every candidate in the requested BIC grid."""
    kappas = _resolve_bic_kappas(
        variant=variant, kappa_grid=kappa_grid, k_grid=k_grid, k0=k0, a0=a0
    )

    scores: list[BicScore] = []
    skipped_reference_candidates = 0
    reference_values: dict[str, set[int]] = {"k0": set(), "a0": set()}
    total_candidates = 0
    for K in k_grid:
        for lam in lambda_grid:
            for kappa in kappas:
                total_candidates += 1
                artifacts = fit_candidate_fn(int(K), float(lam), None if kappa is None else float(kappa))
                try:
                    score = modified_bic_score(
                        artifacts,
                        saturated_bcss=saturated_bcss,
                        variant=variant,
                        k0=k0,
                        a0=a0,
                        active_threshold=active_threshold,
                    )
                except SparseBicReferenceError as exc:
                    skipped_reference_candidates += 1
                    reference_values.setdefault(exc.reference_name, set()).add(exc.reference_value)
                    continue
                scores.append(score)
    if not scores and skipped_reference_candidates:
        refs = []
        for name in ("k0", "a0"):
            values = sorted(reference_values.get(name, ()))
            if values:
                refs.append(f"{name.upper()}={values[0] if len(values) == 1 else values}")
        ref_msg = f" ({', '.join(refs)})" if refs else ""
        raise ValueError(
            "No valid sparse BIC candidates after skipping "
            f"{skipped_reference_candidates}/{total_candidates} candidates below the sparse "
            f"BIC reference{ref_msg}. Try lowering --bic-a0 or using a higher --kappa-grid."
        )
    return BicScoreList(
        scores,
        skipped_reference_candidates=skipped_reference_candidates,
        total_candidates=total_candidates,
    )


def best_bic_candidate(scores: Sequence[BicScore]) -> BicScore:
    """Select the candidate with the lowest BIC."""
    if not scores:
        raise ValueError("No BIC scores to select from")
    return min(scores, key=lambda s: s.bic)


__all__ = [
    "DEFAULT_K_GRID",
    "DEFAULT_LAMBDA_GRID",
    "DEFAULT_BIC_K0",
    "DEFAULT_BIC_A0",
    "DEFAULT_BIC_KBAR",
    "DEFAULT_BIC_ACTIVE_THRESHOLD",
    "BicScore",
    "BicScoreList",
    "SparseBicReferenceError",
    "default_bic_a0",
    "default_kappa_grid",
    "count_jumps",
    "active_feature_count",
    "bcss_by_feature",
    "artifact_bcss",
    "modified_bic_score",
    "select_bic_hyperparameters",
    "best_bic_candidate",
]
