"""Modified-BIC hyperparameter selection helpers shared by the JM entrypoints.

In the upstream repo these five helpers live inside ``train_jump_model`` — the
per-asset Stage 0 CLI — and ``train_global_jump_model`` imports them from there.
This repo only needs the *global* pooled fit (one JM per granularity, labels per
asset), so copying the 1200-line per-asset entrypoint just to reach five helpers
would drag in the whole per-asset orchestrator. They are extracted here verbatim
instead, keeping the global trainer's import surface to what fitting requires.

Kept identical to upstream so a future diff against ``train_jump_model`` stays
trivial; only the imports were rewritten to point at this package.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from Utils.regime.model import RegimeJumpModel, is_sparse_variant, variant_is_continuous
from Utils.regime.stat_tuning import (
    DEFAULT_K_GRID,
    artifact_bcss,
    best_bic_candidate,
    default_kappa_grid,
    select_bic_hyperparameters,
)

# Continuous JM variants cap K; the hard-label families use the full DEFAULT_K_GRID.
MAX_CONTINUOUS_K = 4
DEFAULT_CONTINUOUS_K_GRID = (2, 3)


def _hard_bic_selection_variant(variant: str) -> str:
    """Return the hard-label variant used for modified-BIC selection."""
    if variant == "continuous":
        return "standard"
    if variant == "sparse_continuous":
        return "sparse"
    return variant


def _resolve_bic_k_grid(k_grid: Sequence[int] | None, *, variant: str) -> tuple[int, ...]:
    """Resolve the candidate K grid, enforcing the continuous-family K cap."""
    if k_grid is None:
        grid = DEFAULT_CONTINUOUS_K_GRID if variant_is_continuous(variant) else tuple(DEFAULT_K_GRID)
    else:
        grid = tuple(int(x) for x in k_grid)
    if not grid:
        raise ValueError("k_grid must contain at least one K")
    if variant_is_continuous(variant):
        too_high = [k for k in grid if k > MAX_CONTINUOUS_K]
        if too_high:
            raise ValueError(
                f"continuous variants support K <= {MAX_CONTINUOUS_K}; "
                f"got k_grid={list(grid)}"
            )
    return grid


def _validate_continuous_k_cap(variant: str, k: int) -> None:
    if variant_is_continuous(variant) and int(k) > MAX_CONTINUOUS_K:
        raise ValueError(
            f"continuous variants support K <= {MAX_CONTINUOUS_K}; got K={int(k)}"
        )


def _log_bic_selection(bic_scores,
                       *,
                       chosen_k: int,
                       chosen_lambda: float,
                       chosen_max_feats: float | None,
                       variant: str) -> None:
    """Print the per-candidate modified-BIC table, marking the chosen candidate."""
    skipped = int(getattr(bic_scores, "skipped_reference_candidates", 0))
    total = int(getattr(bic_scores, "total_candidates", len(bic_scores) + skipped))
    if skipped:
        print(f"[regime] sparse BIC skipped {skipped}/{total} candidates below K0/A0 reference")
    for s in bic_scores:
        marker = (
            " <== best"
            if s.n_components == chosen_k
            and s.jump_penalty == chosen_lambda
            and (not is_sparse_variant(variant) or s.kappa == chosen_max_feats)
            else ""
        )
        kappa_msg = f"  kappa={s.kappa:g}" if s.kappa is not None else ""
        print(f"  K={s.n_components}  λ={s.jump_penalty:>7g}{kappa_msg}  "
              f"BIC={s.bic:+.6f}  jumps={s.n_jumps:>4d}  A={s.active_features}"
              f"{marker}")


def _select_by_modified_bic(*,
                            variant: str,
                            feats_fit: pd.DataFrame,
                            ret_fit: pd.Series,
                            groups_fit: pd.Series | np.ndarray | None = None,
                            fit_p: int,
                            resolved_max_feats: float | None,
                            k_grid: Sequence[int],
                            lambda_grid: Sequence[float],
                            kappa_grid: Sequence[float] | None,
                            kappa_grid_size: int,
                            bic_k0: int,
                            bic_a0: int | None,
                            bic_kbar: int,
                            bic_active_threshold: float,
                            verbose: bool) -> tuple[int, float, float | None, list]:
    """Fit the requested grid on the BIC fit window (train+val) and pick the
    lowest modified-BIC candidate. The test split is never seen here.

    Returns ``(chosen_k, chosen_lambda, chosen_max_feats, bic_scores)``.
    """
    candidate_k_grid = tuple(int(x) for x in k_grid)
    candidate_lambda_grid = tuple(float(x) for x in lambda_grid)
    if not candidate_k_grid:
        raise ValueError("k_grid must contain at least one K")
    if not candidate_lambda_grid:
        raise ValueError("lambda_grid must contain at least one lambda")
    if is_sparse_variant(variant):
        candidate_kappa_grid = (
            tuple(float(x) for x in kappa_grid)
            if kappa_grid is not None
            else default_kappa_grid(fit_p, size=kappa_grid_size)
        )
        saturated_max_feats = float(np.sqrt(fit_p))
    else:
        candidate_kappa_grid = None
        saturated_max_feats = resolved_max_feats

    saturated = RegimeJumpModel(
        variant=variant,
        n_components=bic_kbar,
        jump_penalty=0.0,
        max_feats=saturated_max_feats,
    )
    saturated.fit(feats_fit, ret_fit, groups=groups_fit)
    saturated_bcss = artifact_bcss(saturated.fit_artifacts(feats_fit, groups=groups_fit))

    def _fit_candidate(candidate_k: int, lam: float, kappa: float | None):
        jm = RegimeJumpModel(
            variant=variant,
            n_components=candidate_k,
            jump_penalty=lam,
            max_feats=(float(kappa) if kappa is not None else resolved_max_feats),
        )
        jm.fit(feats_fit, ret_fit, groups=groups_fit)
        return jm.fit_artifacts(feats_fit, groups=groups_fit)

    if verbose:
        print(f"[regime] selecting by modified BIC on train+val | "
              f"variant={variant} K={list(candidate_k_grid)} λ={list(candidate_lambda_grid)}"
              f"{f' kappa={list(candidate_kappa_grid)}' if is_sparse_variant(variant) else ''}")
    bic_scores = select_bic_hyperparameters(
        _fit_candidate,
        variant=variant,
        saturated_bcss=saturated_bcss,
        k_grid=candidate_k_grid,
        lambda_grid=candidate_lambda_grid,
        kappa_grid=candidate_kappa_grid,
        k0=bic_k0,
        a0=bic_a0,
        active_threshold=bic_active_threshold,
    )
    best = best_bic_candidate(bic_scores)
    chosen_k = best.n_components
    chosen_lambda = best.jump_penalty
    chosen_max_feats = float(best.kappa) if best.kappa is not None else resolved_max_feats
    if verbose:
        _log_bic_selection(
            bic_scores,
            chosen_k=chosen_k,
            chosen_lambda=chosen_lambda,
            chosen_max_feats=chosen_max_feats,
            variant=variant,
        )
    return chosen_k, chosen_lambda, chosen_max_feats, bic_scores




__all__ = [
    "MAX_CONTINUOUS_K",
    "DEFAULT_CONTINUOUS_K_GRID",
    "_hard_bic_selection_variant",
    "_resolve_bic_k_grid",
    "_validate_continuous_k_cap",
    "_log_bic_selection",
    "_select_by_modified_bic",
]
