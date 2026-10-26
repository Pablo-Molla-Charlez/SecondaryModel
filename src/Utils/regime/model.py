"""Jump Model wrapper for regime detection (Stage 0).

Thin, config-driven wrapper over the `jumpmodels` package that exposes the three
variants the project wants behind one interface:

  • "standard"          → JumpModel(cont=False)        — hard regime labels
  • "continuous"        → JumpModel(cont=True)         — soft regime probabilities (CJM)
  • "sparse"            → SparseJumpModel(cont=False)  — feature-selecting JM, hard labels
  • "sparse_continuous" → SparseJumpModel(cont=True)   — feature-selecting JM, soft probabilities

The two sparse variants are identical for feature selection (max_feats, kappa grid,
sparse BIC, learned weights) and differ ONLY in `cont`: use ``is_sparse_variant`` to
branch on the sparse family and ``variant_is_continuous`` to decide soft vs hard.

K (n_components) is configurable (K=2 calm/stress, K=3 bull/sideways/bear).

Design choices baked in:
  • Features are standardized before fitting — the JM clustering loss weights
    every feature equally, so unscaled features would let the largest-variance
    one dominate. We use the package's StandardScalerPD.
  • States are sorted by cumulative return (`sort_by="cumret"`) so regime IDs are
    interpretable and STABLE across refits: regime 0 = lowest-cumret (bear-like),
    regime K-1 = highest-cumret (bull-like). This matters because downstream M2
    features and Stage-2 conditioning reference regimes by integer ID.
  • Causal (online) inference is provided separately from in-sample fit. The
    in-sample `.predict()` labels peek at the whole history and are for
    EXPLORATION / visualization / choosing K & λ only. Labels that feed M2 or
    Stage-2 conditioning must be the causal ones (`predict_online`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

from jumpmodels.jump import (
    JumpModel,
    check_X_with_feat_weights,
    do_E_step,
    empirical_trans_mx,
    raise_JM_labels_to_proba,
    raise_JM_proba_to_df,
    reduce_proba_to_labels,
    valid_feat_weights,
    weighted_mean_cluster,
)
from jumpmodels.sparse_jump import SparseJumpModel
from jumpmodels.sparse_jump import compute_BCSS, solve_lasso
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd, compute_num_shifts
from jumpmodels.utils.cluster import is_same_clustering
from jumpmodels.utils.validation import check_2d_array, raise_arr_to_pd_obj

# Variant names selectable from config / CLI.
JM_VARIANTS = ("standard", "continuous", "sparse", "sparse_continuous")


def is_sparse_variant(variant: str) -> bool:
    """True for the feature-selecting (SparseJumpModel) variants, hard or soft."""
    return variant in ("sparse", "sparse_continuous")


def variant_is_continuous(variant: str) -> bool:
    """True for variants that emit soft probabilities (cont=True)."""
    return variant in ("continuous", "sparse_continuous")


def _group_values(groups: pd.Series | np.ndarray | list | tuple | None,
                  index: pd.Index | None,
                  n: int) -> np.ndarray | None:
    """Return a 1D group vector aligned to ``n`` rows, or None."""
    if groups is None:
        return None
    if isinstance(groups, pd.Series):
        values = groups.reindex(index).to_numpy() if index is not None else groups.to_numpy()
    else:
        values = np.asarray(groups)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"groups must be a 1D vector with {n} entries")
    return values


def _group_slices(groups: np.ndarray | None, n: int) -> list[slice]:
    """Contiguous blocks in the input sequence; boundaries reset the E-step."""
    if groups is None:
        return [slice(0, n)]
    if n == 0:
        return []
    cuts = np.flatnonzero(groups[1:] != groups[:-1]) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, n]
    return [slice(int(a), int(b)) for a, b in zip(starts, stops)]


def _grouped_do_E_step(X: np.ndarray,
                       centers_: np.ndarray,
                       penalty_mx: np.ndarray,
                       prob_vecs: np.ndarray | None,
                       groups: np.ndarray | None,
                       *,
                       return_value_mx: bool = False):
    """Run the JM E-step independently inside each contiguous group block."""
    blocks = _group_slices(groups, len(X))
    if return_value_mx:
        parts = [
            do_E_step(X[block], centers_, penalty_mx, prob_vecs,
                      return_value_mx=True)
            for block in blocks
        ]
        return np.vstack(parts) if parts else np.empty((0, len(centers_)))

    proba_parts, label_parts = [], []
    total_val = 0.0
    for block in blocks:
        proba, labels, val = do_E_step(X[block], centers_, penalty_mx, prob_vecs)
        proba_parts.append(proba)
        label_parts.append(labels)
        total_val += float(val)
    if not proba_parts:
        return (
            np.empty((0, len(centers_))),
            np.empty((0,), dtype=int),
            0.0,
        )
    return np.vstack(proba_parts), np.concatenate(label_parts), float(total_val)


def _empirical_trans_mx_grouped(labels_: pd.Series | np.ndarray,
                                groups: np.ndarray | None,
                                n_components: int) -> np.ndarray:
    """Transition matrix that never counts transitions across group boundaries."""
    if groups is None:
        return empirical_trans_mx(labels_, n_components=n_components)
    labels = pd.Series(labels_).to_numpy(dtype=int)
    count_mx = np.zeros((n_components, n_components), dtype=int)
    for block in _group_slices(groups, len(labels)):
        block_labels = labels[block]
        if len(block_labels) <= 1:
            continue
        for state in range(n_components):
            next_states = block_labels[1:][block_labels[:-1] == state]
            states, counts = np.unique(next_states, return_counts=True)
            count_mx[state, states.astype(int)] += counts
    denom = count_mx.sum(axis=1, keepdims=True)
    return np.divide(
        count_mx.astype(float),
        denom,
        out=np.full(count_mx.shape, np.nan, dtype=float),
        where=denom != 0,
    )


@dataclass
class RegimeFitResult:
    """Everything produced by a single JM fit, ready to persist."""
    labels: pd.Series                    # hard regime id per bar (in-sample)
    proba: pd.DataFrame                  # soft regime probabilities per bar (in-sample)
    n_shifts: int                        # number of regime changes (sanity metric)
    variant: str
    n_components: int
    jump_penalty: float
    feature_set: str
    feature_names: list[str] = field(default_factory=list)


@dataclass
class FitArtifacts:
    """Scoring inputs for a fitted JM/CJM/SJM, consumed by stat_tuning's BIC.

    A typed payload — replacing what used to be a bare dict — so the
    model→stat_tuning scoring contract is explicit. ``feature_weights`` are
    unit weights for non-sparse variants and the learned SJM weights for sparse;
    ``centers`` are return-ordered cluster centers (or None if the estimator does
    not expose them); ``max_feats`` is set only for the sparse variant.
    """
    X_scaled: pd.DataFrame
    labels: pd.Series
    proba: pd.DataFrame
    feature_weights: pd.Series
    feature_names: list[str]
    n_components: int
    jump_penalty: float
    centers: np.ndarray | None = None
    max_feats: float | None = None
    groups: pd.Series | np.ndarray | None = None

    @classmethod
    def from_mapping(cls, source: "FitArtifacts | Mapping") -> "FitArtifacts":
        """Coerce a FitArtifacts (returned as-is) or a plain mapping into one.

        Lets lightweight callers/fixtures pass a dict of the same keys while the
        scoring code works against the typed dataclass.
        """
        if isinstance(source, cls):
            return source
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in source.items() if k in names})


class RegimeJumpModel:
    """Config-driven Jump Model for regime labelling.

    Args:
        variant: one of {"standard", "continuous", "sparse", "sparse_continuous"}.
        n_components: number of regimes K (2 or 3 typically).
        jump_penalty: λ — the persistence penalty. Larger ⇒ fewer, longer regimes.
        max_feats: only for the sparse variant — soft cap on selected features.
                   If None, resolves to sqrt(number of train features) at fit time.
        random_state / max_iter / n_init: passed through to jumpmodels.
    """

    def __init__(self,
                 variant: str = "continuous",
                 n_components: int = 2,
                 jump_penalty: float = 50.0,
                 max_feats: float | None = None,
                 random_state: int = 0,
                 max_iter: int = 30,
                 n_init: int = 10) -> None:
        if variant not in JM_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choices: {JM_VARIANTS}")
        self.variant = variant
        self.n_components = int(n_components)
        self.jump_penalty = float(jump_penalty)
        self.max_feats = None if max_feats is None else float(max_feats)
        self.random_state = random_state
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)

        # The estimator is constructed in fit(), once max_feats is resolved for
        # the sparse variant — keeping all estimator-construction in one place.
        self._jm = None
        self._clipper: DataClipperStd | None = None
        self._scaler: StandardScalerPD | None = None
        self._feature_names: list[str] = []
        # Identity remap until fit() reorders regimes by mean return.
        self._remap = np.arange(self.n_components, dtype=int)

    def _make_estimator(self):
        cont = variant_is_continuous(self.variant)
        if is_sparse_variant(self.variant):
            if self.max_feats is None:
                raise ValueError("max_feats must be resolved before building a sparse JM")
            return SparseJumpModel(
                n_components=self.n_components,
                max_feats=self.max_feats,
                jump_penalty=self.jump_penalty,
                cont=cont,                 # sparse can also be continuous; default hard
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_init_jm=self.n_init,
            )
        return JumpModel(
            n_components=self.n_components,
            jump_penalty=self.jump_penalty,
            cont=cont,
            random_state=self.random_state,
            max_iter=max(self.max_iter, 1000),  # JumpModel default 1000 iters
            n_init=self.n_init,
        )

    def _fit_jump_estimator_grouped(self,
                                    jm: JumpModel,
                                    X: pd.DataFrame,
                                    ret_ser: pd.Series | None,
                                    groups: np.ndarray | None,
                                    *,
                                    feat_weights=None,
                                    sort_by: str | None = "cumret") -> JumpModel:
        """Fit a JumpModel while resetting the E-step at group boundaries."""
        valid_feat_weights(feat_weights)
        X_arr = check_X_with_feat_weights(X, feat_weights)
        jm.feat_weights = feat_weights
        n_c = jm.n_components
        jump_penalty_mx = jm.check_jump_penalty_mx()
        init_centers_values = jm.init_centers(X_arr)
        best_val = np.inf
        best_idx = 0
        best_res = {"labels_": None}

        for n_init_, centers_ in enumerate(init_centers_values):
            labels_pre, val_pre = None, np.inf
            proba_, labels_, val_ = _grouped_do_E_step(
                X_arr, centers_, jump_penalty_mx, jm.prob_vecs, groups)
            num_iter = 0
            while (
                num_iter < jm.max_iter
                and (not is_same_clustering(labels_, labels_pre))
                and val_pre - val_ > jm.tol
            ):
                num_iter += 1
                labels_pre, val_pre = labels_, val_
                centers_ = weighted_mean_cluster(X_arr, proba_)
                proba_, labels_, val_ = _grouped_do_E_step(
                    X_arr, centers_, jump_penalty_mx, jm.prob_vecs, groups)
            if jm.verbose:
                print(f"{n_init_}-th init. val: {val_}")
            if (not is_same_clustering(best_res["labels_"], labels_)) and val_ < best_val:
                best_idx = n_init_
                best_val = val_
                best_res = {
                    "centers_": centers_,
                    "labels_": labels_,
                    "proba_": proba_,
                }

        jm.val_ = best_val
        if jm.verbose:
            print(f"{best_idx}-th init has the best value: {best_val}.")
        # For grouped fits, duplicate timestamps across assets make jumpmodels'
        # Series alignment ambiguous. The wrapper applies its own pooled
        # per-asset online-return remap after fit, so no internal financial sort
        # is needed here.
        jm.centers_ = best_res["centers_"]
        jm.proba_ = raise_JM_proba_to_df(best_res["proba_"], X)
        jm.labels_ = reduce_proba_to_labels(jm.proba_)
        jm.transmat_ = _empirical_trans_mx_grouped(jm.labels_, groups, n_components=n_c)
        return jm

    def _fit_sparse_estimator_grouped(self,
                                      sjm: SparseJumpModel,
                                      X: pd.DataFrame,
                                      ret_ser: pd.Series | None,
                                      groups: np.ndarray | None,
                                      *,
                                      sort_by: str | None = "cumret") -> SparseJumpModel:
        """Sparse JM fit loop using the grouped JumpModel base fit."""
        X_arr = check_2d_array(X)
        sjm.n_features_all = X_arr.shape[1]
        jm = sjm.init_jm()
        norm_ub = np.sqrt(sjm.max_feats)
        w_old = np.ones(sjm.n_features_all) * 2
        w = np.ones(sjm.n_features_all) / np.sqrt(sjm.n_features_all)
        n_iter = 0
        centers_unweighted = None

        while (
            n_iter < sjm.max_iter
            and np.linalg.norm(w - w_old, 1) / np.linalg.norm(w_old, 1) > sjm.tol_w
        ):
            n_iter += 1
            w_old = w
            feat_weights = np.sqrt(w)
            if n_iter > 1 and centers_unweighted is not None:
                jm.centers_ = centers_unweighted * feat_weights
            self._fit_jump_estimator_grouped(
                jm, X, ret_ser, groups, feat_weights=feat_weights, sort_by=sort_by)
            centers_unweighted = weighted_mean_cluster(X_arr, jm.proba_)
            bcss = compute_BCSS(X_arr, jm.proba_, centers_unweighted)
            if (bcss <= 0).all():
                sjm.print_log(n_iter, bcss, w)
                break
            w = solve_lasso(bcss / bcss.max(), norm_ub)
            sjm.print_log(n_iter, bcss, w)

        sjm.w = raise_arr_to_pd_obj(w, X, index_key="columns")
        sjm.feat_weights = raise_arr_to_pd_obj(jm.feat_weights, X, index_key="columns")
        sjm.centers_ = jm.centers_
        sjm.labels_ = jm.labels_
        sjm.proba_ = jm.proba_
        if ret_ser is not None and hasattr(jm, "ret_"):
            sjm.ret_ = jm.ret_
            sjm.vol_ = jm.vol_
        return sjm

    def _preprocess_fit(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit clipper+scaler on X and return the processed frame.

        Matches the jumpmodels reference workflow:
            scalar.fit_transform(clipper.fit_transform(X_train))
        DataClipperStd(mul=3) winsorizes each feature to ±3σ BEFORE standardizing
        — important for crypto's fat tails, so a few extreme bars don't distort
        the standardization or the cluster centers.
        """
        self._clipper = DataClipperStd(mul=3.0)
        self._scaler = StandardScalerPD()
        return self._scaler.fit_transform(self._clipper.fit_transform(X))

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted clipper+scaler to new X (transform only)."""
        if self._scaler is None or self._clipper is None:
            raise ValueError("RegimeJumpModel is not fitted yet. Call fit() first.")
        return self._scaler.transform(self._clipper.transform(X))

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return features after the fitted clipping/scaling pipeline."""
        return self._preprocess(X)

    # ── fit ───────────────────────────────────────────────────────────────
    def fit(self,
            X: pd.DataFrame,
            ret_ser: pd.Series,
            groups: pd.Series | np.ndarray | list | tuple | None = None) -> "RegimeJumpModel":
        """Fit on standardized features, then relabel regimes by mean return.

        We do NOT trust jumpmodels' own state ordering: its `sort_by="cumret"`
        uses a NEGATED criterion (`-ret*freq`), so it puts the highest-return
        cluster at id 0 — the opposite of an intuitive bear→bull ramp, and it is
        weighted by cluster frequency (so a huge calm cluster can outrank a small
        strongly-trending one). Instead we relabel deterministically by each
        regime's MEAN RETURN so the mapping is data-earned and stable:
            id 0      = lowest  mean return  (most bear-like)
            id K-1    = highest mean return  (most bull-like)

        CRITICAL: the remap is derived from the CAUSAL (online) labels — the same
        label sequence that downstream code consumes (M2 / Stage 2 / plots). The
        in-sample `.predict()` labels differ from the online ones, so ordering by
        in-sample mean return does NOT guarantee the online labels come out
        ascending. Deriving from online labels keeps the saved/plotted labels
        correctly ordered.

        Args:
            X: feature frame indexed by datetime (from regime.features).
            ret_ser: per-bar return series aligned to X.index, used to order
                     regimes by realized mean return.
        """
        self._feature_names = list(X.columns)
        if is_sparse_variant(self.variant) and self.max_feats is None:
            self.max_feats = float(np.sqrt(len(self._feature_names)))
        self._jm = self._make_estimator()
        X_scaled = self._preprocess_fit(X)        # clip ±3σ then standardize
        ret_aligned = ret_ser.reindex(X.index)
        group_values = _group_values(groups, X.index, len(X))
        if group_values is None:
            self._jm.fit(X_scaled, ret_ser=ret_aligned, sort_by="cumret")
        elif is_sparse_variant(self.variant):
            self._fit_sparse_estimator_grouped(
                self._jm, X_scaled, ret_aligned, group_values, sort_by="cumret")
        else:
            self._fit_jump_estimator_grouped(
                self._jm, X_scaled, ret_aligned, group_values, sort_by="cumret")

        # ── Build the relabel permutation from CAUSAL (online) mean returns ──
        # raw_online comes straight from the underlying estimator (no remap), so
        # the permutation below is derived from untouched labels.
        if group_values is None:
            raw_online = np.asarray(self._jm.predict_online(X_scaled)).astype(int)
        else:
            raw_online = self._predict_raw_labels(
                X_scaled, group_values, online=True).astype(int)
        self._remap, mean_ret = self._return_ordered_remap(raw_online, ret_aligned.values)
        self._regime_mean_ret = {int(self._remap[r]): float(mean_ret[r])
                                 for r in range(self.n_components) if not np.isnan(mean_ret[r])}
        return self

    def _return_ordered_remap(self, raw_labels: np.ndarray,
                              ret_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(remap, mean_ret)`` ordering raw regime ids by mean return.

        ``remap[raw_id] = new_id`` so id 0 is the lowest-mean-return (most
        bearish) regime and id K-1 the highest. Empty regimes (NaN mean) sort
        last (treated as +inf) so they take the top ids harmlessly.
        """
        k = self.n_components
        mean_ret = np.array([
            np.nanmean(ret_vals[raw_labels == r]) if np.any(raw_labels == r) else np.nan
            for r in range(k)
        ])
        order = np.argsort(np.where(np.isnan(mean_ret), np.inf, mean_ret))
        remap = np.empty(k, dtype=int)
        for new_id, raw_id in enumerate(order):
            remap[raw_id] = new_id
        return remap, mean_ret

    # Translate raw jumpmodels ids to return-ordered ids so regime 0 is bearish.
    def _inverse_remap(self) -> np.ndarray:
        inv = np.empty(self.n_components, dtype=int)
        for raw_id, new_id in enumerate(self._remap):
            inv[new_id] = raw_id
        return inv

    def _apply_remap_labels(self, raw: np.ndarray) -> np.ndarray:
        return self._remap[raw]

    def _apply_remap_proba(self, proba: np.ndarray) -> np.ndarray:
        out = np.empty_like(proba)
        inv = self._inverse_remap()  # new column new_id ← raw column inv[new_id]
        for new_id in range(self.n_components):
            out[:, new_id] = proba[:, inv[new_id]]
        return out

    def _base_predict_estimator(self):
        if self._jm is None:
            raise ValueError("RegimeJumpModel is not fitted yet. Call fit() first.")
        return self._jm.jm_ins if is_sparse_variant(self.variant) else self._jm

    def _predict_raw_proba(self,
                           X_scaled: pd.DataFrame,
                           groups: np.ndarray | None = None,
                           *,
                           online: bool = False) -> np.ndarray:
        """Underlying estimator probabilities before return-order remapping."""
        if groups is None:
            method = (
                self._jm.predict_proba_online
                if online
                else self._jm.predict_proba
            )
            return np.asarray(method(X_scaled), dtype=float)

        estimator = self._base_predict_estimator()
        X_arr = estimator.check_X_predict_func(X_scaled)
        if online:
            value_mx = _grouped_do_E_step(
                X_arr, estimator.centers_, estimator.jump_penalty_mx,
                estimator.prob_vecs, groups, return_value_mx=True)
            labels = value_mx.argmin(axis=1)
            return raise_JM_labels_to_proba(labels, estimator.n_components, estimator.prob_vecs)
        proba, _, _ = _grouped_do_E_step(
            X_arr, estimator.centers_, estimator.jump_penalty_mx,
            estimator.prob_vecs, groups)
        return proba

    def _predict_raw_labels(self,
                            X_scaled: pd.DataFrame,
                            groups: np.ndarray | None = None,
                            *,
                            online: bool = False) -> np.ndarray:
        proba = self._predict_raw_proba(X_scaled, groups, online=online)
        return np.asarray(proba.argmax(axis=1), dtype=int)

    # ── in-sample (non-causal) inference — exploration only ───────────────
    def predict(self,
                X: pd.DataFrame,
                groups: pd.Series | np.ndarray | list | tuple | None = None) -> pd.Series:
        X_scaled = self._preprocess(X)
        group_values = _group_values(groups, X.index, len(X))
        labels = self._apply_remap_labels(
            self._predict_raw_labels(X_scaled, group_values, online=False))
        return pd.Series(labels, index=X.index, name="regime")

    def predict_proba(self,
                      X: pd.DataFrame,
                      groups: pd.Series | np.ndarray | list | tuple | None = None) -> pd.DataFrame:
        X_scaled = self._preprocess(X)
        group_values = _group_values(groups, X.index, len(X))
        proba = self._apply_remap_proba(
            self._predict_raw_proba(X_scaled, group_values, online=False))
        cols = [f"p_regime_{k}" for k in range(proba.shape[1])]
        return pd.DataFrame(proba, index=X.index, columns=cols)

    # ── causal (online) inference — the labels that feed M2 / Stage 2 ─────
    def predict_online(self,
                       X: pd.DataFrame,
                       groups: pd.Series | np.ndarray | list | tuple | None = None) -> pd.Series:
        """Causal hard labels: label at t uses only data ≤ t."""
        X_scaled = self._preprocess(X)
        group_values = _group_values(groups, X.index, len(X))
        labels = self._apply_remap_labels(
            self._predict_raw_labels(X_scaled, group_values, online=True))
        return pd.Series(labels, index=X.index, name="regime")

    def predict_proba_online(self,
                             X: pd.DataFrame,
                             groups: pd.Series | np.ndarray | list | tuple | None = None) -> pd.DataFrame:
        """Causal soft probabilities: probs at t use only data ≤ t."""
        X_scaled = self._preprocess(X)
        group_values = _group_values(groups, X.index, len(X))
        proba = self._apply_remap_proba(
            self._predict_raw_proba(X_scaled, group_values, online=True))
        cols = [f"p_regime_{k}" for k in range(proba.shape[1])]
        return pd.DataFrame(proba, index=X.index, columns=cols)

    # ── convenience: fit + in-sample labels + sanity metric ───────────────
    def fit_label(self,
                  X: pd.DataFrame,
                  ret_ser: pd.Series,
                  feature_set: str = "nystrup",
                  groups: pd.Series | np.ndarray | list | tuple | None = None) -> RegimeFitResult:
        """Fit and return in-sample labels/proba + the regime-shift count.

        The shift count is a key sanity check: a good JM produces few,
        persistent shifts. >50 shifts over ~4 years ⇒ λ too low.
        """
        self.fit(X, ret_ser, groups=groups)
        labels = self.predict(X, groups=groups)
        proba = self.predict_proba(X, groups=groups)
        group_values = _group_values(groups, X.index, len(X))
        if group_values is None:
            n_shifts = int(compute_num_shifts(labels.values))
        else:
            n_shifts = int(sum(
                compute_num_shifts(labels.iloc[block].values)
                for block in _group_slices(group_values, len(labels))
            ))
        return RegimeFitResult(
            labels=labels,
            proba=proba,
            n_shifts=n_shifts,
            variant=self.variant,
            n_components=self.n_components,
            jump_penalty=self.jump_penalty,
            feature_set=feature_set,
            feature_names=list(self._feature_names),
        )

    @property
    def selected_features(self) -> list[tuple[str, float]] | None:
        """For the sparse variant, the feature weights it learned (else None)."""
        w = getattr(self._jm, "feat_weights", None)
        if w is None:
            return None
        return list(zip(self._feature_names, np.asarray(w, dtype=float)))

    def fit_artifacts(self,
                      X: pd.DataFrame,
                      groups: pd.Series | np.ndarray | list | tuple | None = None) -> FitArtifacts:
        """Artifacts needed to score a fitted JM/CJM/SJM without private access elsewhere."""
        X_scaled = self.transform_features(X)
        group_values = _group_values(groups, X.index, len(X))
        labels = self.predict(X, groups=groups)
        proba = self.predict_proba(X, groups=groups)
        if is_sparse_variant(self.variant):
            weights = getattr(self._jm, "w", None)
            if weights is None:
                weights = getattr(self._jm, "feat_weights", None)
            if weights is None:
                raise ValueError("Fitted sparse JM does not expose feature weights")
            feature_weights = pd.Series(np.asarray(weights, dtype=float),
                                        index=self._feature_names)
        else:
            feature_weights = pd.Series(np.ones(len(self._feature_names), dtype=float),
                                        index=self._feature_names)
        raw_centers = getattr(self._jm, "centers_", None)
        centers = None
        if raw_centers is not None:
            raw_centers = np.asarray(raw_centers, dtype=float)
            centers = raw_centers[self._inverse_remap()]
        return FitArtifacts(
            X_scaled=X_scaled,
            labels=labels,
            proba=proba,
            centers=centers,
            feature_weights=feature_weights,
            feature_names=list(self._feature_names),
            n_components=int(self.n_components),
            jump_penalty=float(self.jump_penalty),
            max_feats=float(self.max_feats) if is_sparse_variant(self.variant) else None,
            groups=(np.asarray(group_values).copy() if group_values is not None else None),
        )


__all__ = ["RegimeJumpModel", "RegimeFitResult", "FitArtifacts", "JM_VARIANTS",
           "is_sparse_variant", "variant_is_continuous"]
