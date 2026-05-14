"""Feature analysis and plotting helpers extracted from kronos_tree.py."""

import json
import warnings
import re
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from Utils.analysis.analysis_meta_labels import plot_asset_correlation
from sklearn.metrics import accuracy_score, fbeta_score, matthews_corrcoef, precision_recall_fscore_support


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MDA / SHAP / LIME  FEATURE RANKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_RNG = np.random.default_rng(42)


__all__ = ["compute_classification_metrics",
           "extract_time_features",
           "mda_rank",
           "shap_rank",
           "lime_rank",
           "combine_rankings",
           "run_feature_selection",
           "compute_asset_correlation"]




# ┏━━━━━━━━━━ Time Features ━━━━━━━━━━┓
def extract_time_features(timestamps: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    """
    Extract 5 temporal features from timestamps.
    
    Returns:
        dict with keys: minute, hour, dow, dom, month
    """
    return {'minute': (timestamps.hour * 60 + timestamps.minute).values.astype(np.int64),
            'hour': timestamps.hour.values.astype(np.int64),
            'dow': timestamps.dayofweek.values.astype(np.int64),
            'dom': (timestamps.day - 1).values.astype(np.int64),
            'month': (timestamps.month - 1).values.astype(np.int64)}


# ┏━━━━━━━━━━ Classification Metrics ━━━━━━━━━━┓
def compute_classification_metrics(targets: np.ndarray, preds: np.ndarray) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1-Score, F-Beta(0.9), and MCC.
    
    Args:
        targets: Ground truth labels (numpy array)
        preds: Predicted labels (numpy array)
        
    Returns:
        dict: Metrics dictionary
    """
    # ┏━━━━━━━━━━ Determine average mode ━━━━━━━━━━┓
    unique_labels = np.unique(targets)
    is_multiclass = len(unique_labels) > 2 or (len(unique_labels) == 2 and not np.array_equal(sorted(unique_labels), [0, 1]))
    avg_mode = 'weighted' if is_multiclass else 'macro'
    
    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average=avg_mode, zero_division=0)
    
    # Custom beta focus (for consistency, use same avg_mode)
    fbeta = fbeta_score(targets, preds, beta=0.3, average=avg_mode, zero_division=0)
    mcc = matthews_corrcoef(targets, preds)
    
    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fbeta': fbeta,
            'mcc': mcc}


# ┏━━━━━━━━━━ MDA (Mean Decrease in Accuracy) ━━━━━━━━━━┓
def mda_rank(model,
             X_train: pd.DataFrame, y_train: np.ndarray,
             X_val: pd.DataFrame,   y_val: np.ndarray,
             save_dir: Path,
             verbose: bool = False) -> dict:
    """Mean Decrease in Accuracy — permutation importance on validation set.

    For each feature, shuffle it in X_val, predict, measure accuracy drop.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator (must have .fit / .predict).
    X_train, y_train : training data (model is re-fit here for safety).
    X_val, y_val     : held-out validation data used for scoring.
    save_dir         : directory for bar-chart PNG + CSV.

    Returns
    -------
    dict : {feature_name: delta_accuracy} (higher = more important).
    """
    # ┏━━━━━━━━━━ Get columns and scale data ━━━━━━━━━━┓
    cols = list(X_train.columns)
    scaler = StandardScaler()

    # ┏━━━━━━━━━━ Fit model and calculate base accuracy ━━━━━━━━━━┓
    Xt = scaler.fit_transform(X_train.values)
    Xv = scaler.transform(X_val.values)
    model.fit(Xt, y_train)
    base_acc = accuracy_score(y_val, model.predict(Xv))

    # ┏━━━━━━━━━━ Compute MDA drops ━━━━━━━━━━┓
    drops = {}
    for i, c in enumerate(cols):
        Xp = Xv.copy()
        Xp[:, i] = _RNG.permutation(Xp[:, i])
        acc_p = accuracy_score(y_val, model.predict(Xp))
        drops[c] = base_acc - acc_p

    # ┏━━━━━━━━━━ Sort and Save ━━━━━━━━━━┓
    s = pd.Series(drops).sort_values(ascending=False)
    s.to_csv(save_dir / "fs_importance_MDA.csv")

    # ┏━━━━━━━━━━ Plot ━━━━━━━━━━┓
    try:
        fig, ax = plt.subplots(figsize=(8, max(3, min(12, 0.25 * len(s)))))
        s.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#3498db", edgecolor="k", linewidth=0.4)
        ax.set_title("MDA — Mean Decrease in Accuracy (Val)")
        ax.set_xlabel("ΔACC")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        fig.savefig(save_dir / "fs_bar_MDA.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    if verbose:
        print(f"  [MDA] baseline_ACC={base_acc:.4f} | top: {dict(s.head(5))}")
    return drops


# ┏━━━━━━━━━━ SHAP Feature Ranking ━━━━━━━━━━┓
def shap_rank(model,
              X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame,
              save_dir: Path,
              verbose: bool = False) -> dict:
    """SHAP-based feature ranking using TreeExplainer for tree models.

    Computes mean |SHAP value| per feature on the validation set.

    Returns
    -------
    dict : {feature_name: mean_abs_shap} (higher = more important).
    """
    try:
        import shap
    except ImportError:
        warnings.warn("[SHAP] `shap` package not installed — skipping SHAP ranking.")
        return {}

    # ┏━━━━━━━━━━ Get columns and scale data ━━━━━━━━━━┓
    cols = list(X_train.columns)
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train.values)
    Xv = scaler.transform(X_val.values)

    # ┏━━━━━━━━━━ Fit model ━━━━━━━━━━┓
    model.fit(Xt, y_train)

    # ┏━━━━━━━━━━ SHAP Explainer (Tree preferred, then generic) ━━━━━━━━━━┓
    try:
        # ┏━━━━━━━━━━ High-performance tree-based explainer (RF, XGB, etc.) ━━━━━━━━━━┓
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xv)
    except Exception:
        # ┏━━━━━━━━━━ Generic fallback for ensembles/complex models (like AutoGluon or custom) ━━━━━━━━━━┓
        explainer = shap.Explainer(model.predict, Xt)
        sv_obj = explainer(Xv)
        sv = sv_obj.values

    # ┏━━━━━━━━━━ Handle multi-class SHAP output ━━━━━━━━━━┓
    if isinstance(sv, list):
        sv = sv[1] if len(sv) >= 2 else sv[0]
    elif hasattr(sv, "ndim") and sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv.mean(axis=2)

    # ┏━━━━━━━━━━ Compute SHAP absolute values ━━━━━━━━━━┓
    shap_abs = np.mean(np.abs(sv), axis=0)
    scores = dict(zip(cols, shap_abs.tolist()))

    # ┏━━━━━━━━━━ Save CSV + plots ━━━━━━━━━━┓
    s = pd.Series(scores).sort_values(ascending=False)
    s.to_csv(save_dir / "fs_importance_SHAP.csv")

    try:
        # ┏━━━━━━━━━━ Beeswarm plot ━━━━━━━━━━┓
        shap.summary_plot(sv, 
                          pd.DataFrame(Xv, columns=cols), 
                          show = False,
                          max_display = min(20, len(cols)))
        plt.title("SHAP Beeswarm — Global Feature Impact (Val)", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(save_dir / "fs_beeswarm_SHAP.png", dpi=150)
        plt.close()

        # ┏━━━━━━━━━━ Bar plot ━━━━━━━━━━┓
        shap.summary_plot(sv, 
                          pd.DataFrame(Xv, columns=cols), 
                          show = False,
                          plot_type = "bar", 
                          max_display = min(20, len(cols)))
        plt.title("SHAP — Mean |SHAP Value| (Val)", fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(save_dir / "fs_bar_SHAP.png", dpi=150)
        plt.close()
    except Exception:
        pass

    if verbose:
        print(f"  [SHAP] top: {dict(s.head(5))}")
    return scores


# ┏━━━━━━━━━━ LIME Feature Ranking ━━━━━━━━━━┓
def lime_rank(model,
              X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame,
              save_dir: Path,
              num_samples: int = 2000,
              num_explanations: int = 256,
              class_idx: int = 1,
              verbose: bool = False) -> dict:
    """LIME-based feature ranking — average |local contribution| across val samples.

    For each explained sample, LIME fits a local linear model around the point
    using perturbed neighbours, and the coefficients become local contributions.

    Returns
    -------
    dict : {feature_name: mean_abs_contribution} (higher = more important).
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        warnings.warn("[LIME] `lime` package not installed — skipping LIME ranking.")
        return {}

    # ┏━━━━━━━━━━ Get columns and scale data ━━━━━━━━━━┓
    cols = list(X_train.columns)
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train.values)
    Xv = scaler.transform(X_val.values)

    # ┏━━━━━━━━━━ Fit model ━━━━━━━━━━┓
    model.fit(Xt, y_train)

    # ┏━━━━━━━━━━ LIME Explainer ━━━━━━━━━━┓
    explainer = LimeTabularExplainer(Xt, 
                                     feature_names         = cols, 
                                     class_names           = ["0", "1"],
                                     discretize_continuous = True, 
                                     random_state          = 42, 
                                     verbose               = False)

    # ┏━━━━━━━━━━ Select samples to explain ━━━━━━━━━━┓
    n = min(len(Xv), num_explanations)
    idx = _RNG.choice(len(Xv), size=n, replace=False)
    contrib = {c: [] for c in cols}

    # ┏━━━━━━━━━━ Predict probabilities ━━━━━━━━━━┓
    def _predict_proba(x):
        try:
            return model.predict_proba(x)
        except Exception:
            y = model.predict(x).ravel()
            return np.vstack([1 - y, y]).T

    # ┏━━━━━━━━━━ Compute feature contributions ━━━━━━━━━━┓
    for i in idx:
        try:
            exp = explainer.explain_instance(Xv[i], 
                                             _predict_proba,
                                             num_features = len(cols), 
                                             num_samples  = num_samples, 
                                             labels       = [class_idx])
            m = exp.as_map().get(class_idx, [])
            for fid, w in m:
                contrib[cols[fid]].append(abs(w))
        except Exception:
            continue

    # ┏━━━━━━━━━━ Save CSV + plot ━━━━━━━━━━┓
    scores = {c: float(np.mean(v)) if v else 0.0 for c, v in contrib.items()}
    s = pd.Series(scores).sort_values(ascending=False)
    s.to_csv(save_dir / "fs_importance_LIME.csv")

    try:
        fig, ax = plt.subplots(figsize=(8, max(3, min(12, 0.25 * len(s)))))
        s.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#e67e22", edgecolor="k", linewidth=0.4)
        ax.set_title("LIME — Mean |Local Contribution| (Val)")
        ax.set_xlabel("Mean |contrib|")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        fig.savefig(save_dir / "fs_bar_LIME.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    if verbose:
        print(f"  [LIME] top: {dict(s.head(5))}")
    return scores



# ┏━━━━━━━━━━ Normalised-rank ensemble ━━━━━━━━━━┓
def combine_rankings(all_scores: Dict[str, dict],
                     save_dir: Optional[Path] = None) -> pd.Series:
    """Combine multiple {feature: score} dicts via normalised-rank ensemble.

    Each method's scores are converted to ranks (1 = best), then normalised
    to (0, 1].  The final score per feature is the average across methods.

    Returns
    -------
    pd.Series : combined score per feature, sorted descending (best first).
    """
    # ┏━━━━━━━━━━ Handle empty input ━━━━━━━━━━┓
    if not all_scores:
        return pd.Series(dtype=float)

    # ┏━━━━━━━━━━ Get all features and initialize combined scores ━━━━━━━━━━┓
    all_feats = sorted(set().union(*[s.keys() for s in all_scores.values()]))
    combined = pd.Series(0.0, index=all_feats)

    # ┏━━━━━━━━━━ Iterate over each method's scores ━━━━━━━━━━┓
    for name, scores in all_scores.items():
        if not scores:
            continue
        s = pd.Series(scores)
        r = s.rank(ascending=False, method="min")        # 1 = best
        r = (r.max() - r + 1) / r.max()                  # normalise to (0, 1]
        r = r.reindex(all_feats).fillna(0.0)
        combined += r

    # ┏━━━━━━━━━━ Normalize the combined scores ━━━━━━━━━━┓
    n_methods = sum(1 for s in all_scores.values() if s)
    if n_methods > 0:
        combined /= n_methods

    # ┏━━━━━━━━━━ Sort the combined scores ━━━━━━━━━━┓
    combined = combined.sort_values(ascending=False)

    # ┏━━━━━━━━━━ Save CSV + plot ━━━━━━━━━━┓
    if save_dir is not None:
        combined.to_csv(save_dir / "fs_combined_rank_score.csv")
        try:
            fig, ax = plt.subplots(figsize=(8, max(3, min(12, 0.25 * len(combined)))))
            combined.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#9b59b6", edgecolor="k", linewidth=0.4)
            ax.set_title("Combined FS Score (Normalised-Rank Ensemble)")
            ax.set_xlabel("Score (avg normalised rank)")
            ax.set_ylabel("Feature")
            plt.tight_layout()
            fig.savefig(save_dir / "fs_combined_bar.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

    return combined



# ┏━━━━━━━━━━ OG Feature Selection ━━━━━━━━━━┓
def run_feature_selection(df_train: pd.DataFrame,
                          labels_train: np.ndarray,
                          df_val: pd.DataFrame,
                          labels_val: np.ndarray,
                          save_dir: Path,
                          model_builder,
                          model_name: str = "rf",
                          fs_methods: Optional[List[str]] = None,
                          top_k: Optional[int] = None,
                          verbose: bool = False) -> dict:
    """Run the full MDA/SHAP/LIME feature selection pipeline.

    Parameters
    ----------
    df_train, labels_train : training features + labels.
    df_val, labels_val     : validation features + labels (temporal split).
    save_dir               : output directory for plots/CSVs.
    model_builder          : callable(model_name, n_samples, cw_ratio) -> model.
    model_name             : model type for builder (default "rf").
    fs_methods             : subset of ["mda", "shap", "lime"]. None = all three.
    top_k                  : max features to select. None = all (ranked only).
    verbose                : print progress.

    Returns
    -------
    dict with keys:
        "selected_features" : list of top-k feature names (or all if top_k=None)
        "combined_scores"   : pd.Series (full ranking)
        "per_method_scores" : {method_name: {feature: score}}
    """
    # ┏━━━━━━━━━━ Handle empty input ━━━━━━━━━━┓
    if fs_methods is None:
        fs_methods = ["mda", "shap", "lime"]
    fs_methods = [m.lower().strip() for m in fs_methods]

    # ┏━━━━━━━━━━ Create save directory ━━━━━━━━━━┓
    save_dir.mkdir(parents=True, exist_ok=True)

    # ┏━━━━━━━━━━ Calculate class weights ━━━━━━━━━━┓
    n_pos = int((labels_train == 1).sum())
    n_neg = int((labels_train == 0).sum())
    cw_ratio = n_neg / max(n_pos, 1)

    # ┏━━━━━━━━━━ Run MDA feature selection ━━━━━━━━━━┓
    per_method: Dict[str, dict] = {}
    if "mda" in fs_methods:
        model = model_builder(model_name, len(labels_train), cw_ratio)
        per_method["mda"] = mda_rank(model, 
                                     df_train, labels_train,
                                     df_val,   labels_val, 
                                     save_dir, verbose=verbose)

    # ┏━━━━━━━━━━ Run SHAP feature selection ━━━━━━━━━━┓
    if "shap" in fs_methods:
        model = model_builder(model_name, len(labels_train), cw_ratio)
        per_method["shap"] = shap_rank(model, 
                                       df_train, labels_train,
                                       df_val, save_dir, verbose=verbose)

    # ┏━━━━━━━━━━ Run LIME feature selection ━━━━━━━━━━┓
    if "lime" in fs_methods:
        model = model_builder(model_name, len(labels_train), cw_ratio)
        per_method["lime"] = lime_rank(model, df_train, labels_train,
                                       df_val, save_dir, verbose=verbose)

    # ┏━━━━━━━━━━ Filter out empty method results ━━━━━━━━━━┓
    per_method = {k: v for k, v in per_method.items() if v}

    # ┏━━━━━━━━━━ Handle empty method results ━━━━━━━━━━┓
    if not per_method:
        warnings.warn("[run_feature_selection] No methods produced scores.")
        return {"selected_features": list(df_train.columns),
                "combined_scores": pd.Series(dtype=float),
                "per_method_scores": {}}

    # ┏━━━━━━━━━━ Combine rankings ━━━━━━━━━━┓
    combined = combine_rankings(per_method, save_dir=save_dir)

    # ┏━━━━━━━━━━ Select top-k ━━━━━━━━━━┓
    if top_k is not None and top_k > 0:
        selected = list(combined.index[:top_k])
    else:
        selected = list(combined.index)

    # ┏━━━━━━━━━━ Save summary ━━━━━━━━━━┓
    summary = {"methods_used": list(per_method.keys()),
               "n_features_scored": len(combined),
               "top_k": top_k,
               "selected_features": selected}
      
    with open(save_dir / "fs_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ┏━━━━━━━━━━ Print summary ━━━━━━━━━━┓
    print(f"  [FS] Methods: {list(per_method.keys())} | "
          f"Scored {len(combined)} features | Selected top-{len(selected)}")
    for i, feat in enumerate(selected[:10]):
        print(f"    {i+1}. {feat:25s}  score={combined[feat]:.4f}")

    return {"selected_features": selected,
            "combined_scores": combined,
            "per_method_scores": per_method}


"""
All four following functions are part of a cross-asset return correlation diagnostic — 
a one-off analysis that justifies training M2 on all assets together 
(rather than per-asset models) by showing assets share systematic risk factors.
"""

# ┏━━━━━━━━━━ Build return matrix ━━━━━━━━━━┓
def _build_return_matrix(dataset: dict) -> pd.DataFrame:
    """Pivot dataset returns into a (date x asset) DataFrame.

    Each cell is the window-level return for that asset at that timestamp.
    Dates are sorted chronologically; assets with fewer observations than
    the most-observed asset are left as NaN for those timestamps.

    Parameters
    ----------
    dataset : dict
        Multi-asset cache dict with keys ``returns``, ``asset_ids``,
        ``asset_map``, ``dates`` (output of ``prepare_multi_asset_dataset``).

    Returns
    -------
    pd.DataFrame
        Shape (T, A) — rows are timestamps, columns are asset names.
    """

    # ┏━━━━━━━━━━ Extract Data ━━━━━━━━━━┓
    returns   = dataset["returns"]
    asset_ids = dataset["asset_ids"]
    dates     = dataset["dates"]
    asset_map = dataset["asset_map"]   # id → name

    # ┏━━━━━━━━━━ Convert to numpy ━━━━━━━━━━┓
    if isinstance(returns, torch.Tensor):
        returns = returns.numpy()
    if isinstance(asset_ids, torch.Tensor):
        asset_ids = asset_ids.numpy().astype(int)

    # ┏━━━━━━━━━━ Create DataFrame ━━━━━━━━━━┓
    df = pd.DataFrame({"date":   pd.to_datetime(dates),
                       "asset":  [asset_map.get(int(a), str(a)) for a in asset_ids],
                       "return": returns})

    # ┏━━━━━━━━━━ Pivot DataFrame ━━━━━━━━━━┓
    pivot = (df.pivot_table(index="date", columns="asset", values="return", aggfunc="mean").sort_index())
    pivot.columns.name = None
    return pivot


# ┏━━━━━━━━━━ Permutation significance ━━━━━━━━━━┓
def _permutation_significance(pivot: pd.DataFrame, n_permutations: int = 5_000, seed: int = 42) -> dict:
    """Test whether the mean pairwise Pearson correlation is significantly > 0.

    Null distribution: for each iteration, independently permute every asset's
    return time series (breaking temporal alignment across assets), recompute
    the full Pearson correlation matrix, and record the mean off-diagonal
    correlation. Repeating this *n_permutations* times builds the null
    distribution under H0 = "assets are uncorrelated".

    Parameters
    ----------
    pivot : pd.DataFrame
        (T x A) return matrix from :func:`_build_return_matrix`, used both to
        compute the observed correlation and to generate the null distribution.

    Returns
    -------
    dict with keys ``observed_mean_r``, ``p_value``, ``null_mean``,
    ``null_std``, ``z_score``.
    """
    # ┏━━━━━━━━━━ Drop rows with any NaN so correlation is well-defined ━━━━━━━━━━┓
    mat = pivot.dropna().values          # shape (T, A)
    T, A = mat.shape
    off_diag = ~np.eye(A, dtype=bool)

    # ┏━━━━━━━━━━ Observed mean pairwise Pearson correlation ━━━━━━━━━━┓
    observed = float(np.corrcoef(mat, rowvar=False)[off_diag].mean())

    # ┏━━━━━━━━━━ Build null distribution by permuting each asset independently ━━━━━━━━━━┓
    rng = np.random.default_rng(seed)
    null_means = np.empty(n_permutations)
    for k in range(n_permutations):
        # ┏━━━━━━━━━━ Shuffle every asset's time axis independently ━━━━━━━━━━┓
        shuffled = np.empty_like(mat)
        for a in range(A):
            shuffled[:, a] = rng.permutation(mat[:, a])
        null_means[k] = float(np.corrcoef(shuffled, rowvar=False)[off_diag].mean())

    # ┏━━━━━━━━━━ One-sided p-value: fraction of null >= observed ━━━━━━━━━━┓
    p_value = float((null_means >= observed).mean())

    return {"observed_mean_r": observed,
            "p_value":         p_value,
            "null_mean":       float(null_means.mean()),
            "null_std":        float(null_means.std()),
            "z_score":         float((observed - null_means.mean()) / (null_means.std() + 1e-12))}


# ┏━━━━━━━━━━ Lead-lag matrix ━━━━━━━━━━┓
def _lead_lag_matrix(pivot: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """Compute peak cross-correlation lag between every ordered asset pair.

    Entry [i, j] is the lag l ∈ [-max_lag, max_lag] at which
    corr(asset_i_t, asset_j_{t+l}) is maximised (absolute value).
    A negative value means asset j leads asset i.

    Parameters
    ----------
    pivot : pd.DataFrame
        (T x A) return matrix from :func:`_build_return_matrix`.
    max_lag : int
        Maximum bar offset to search in each direction.

    Returns
    -------
    pd.DataFrame
        Shape (A, A) — integer lag values.
    """
    # ┏━━━━━━━━━━ Extract Data ━━━━━━━━━━┓
    assets = list(pivot.columns)
    A = len(assets)
    T = len(pivot)
    ret = pivot.values
    lag_matrix = pd.DataFrame(np.zeros((A, A), dtype=float), index=assets, columns=assets)

    # ┏━━━━━━━━━━ Compute Peak Cross-Correlation Lag ━━━━━━━━━━┓
    for i, ai in enumerate(assets):
        for j, aj in enumerate(assets):
            if i == j:
                continue
            best_abs_r, best_lag = -np.inf, 0
            xi, xj = ret[:, i], ret[:, j]
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    a, b = xi[-lag:], xj[:T + lag]
                elif lag > 0:
                    a, b = xi[:T - lag], xj[lag:]
                else:
                    a, b = xi, xj
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.sum() < 10:
                    continue
                r = np.corrcoef(a[valid], b[valid])[0, 1]
                if abs(r) > best_abs_r:
                    best_abs_r, best_lag = abs(r), lag
            lag_matrix.loc[ai, aj] = best_lag

    return lag_matrix


# ┏━━━━━━━━━━ Compute Asset Correlation ━━━━━━━━━━┓
def compute_asset_correlation(dataset: dict,
                              save_dir: Path,
                              gran: str = "",
                              direction: str = "",
                              n_permutations: int = 5_000,
                              max_lag: int = 3,
                              min_overlap: int = 50) -> dict:
    """Quantify and visualise cross-asset return correlation.

    One-off diagnostic that justifies training M2 on all assets simultaneously
    by demonstrating:

    1. **Pearson & Spearman correlation matrices** are non-trivially positive —
       assets share systematic crypto-market risk factors.
    2. **Permutation test** confirms the mean pairwise correlation is
       significantly above the null (p < 0.05).
    3. **Hierarchical clustering** groups assets into regime clusters, showing
       the joint dataset does not introduce contradictory labels.
    4. **Lead-lag analysis** reveals directional information flow between
       assets, supporting the use of cross-asset features.

    Usage
    -----
        from Utils import _load_multi_cache               # or torch.load directly
        from Utils.feature_selection import compute_asset_correlation
        from pathlib import Path

        multi = _load_multi_cache(Path("Output/.../multi_7_fee_up_*.pt"))
        dataset = multi.sub["4h"]
        results = compute_asset_correlation(dataset, Path("Output/Analysis/corr"), gran="4h", direction="up")

    Parameters
    ----------
    dataset : dict
        Multi-asset cache dict with keys ``returns``, ``asset_ids``,
        ``asset_map``, ``dates`` (output of ``prepare_multi_asset_dataset``).
    save_dir : Path
        Directory where PNG figures and a JSON summary are written.
    gran : str
        Granularity label for plot titles / filenames (e.g. ``"4h"``).
    direction : str
        Direction label (``"up"`` / ``"down"``).
    n_permutations : int
        Permutation-test draws.
    max_lag : int
        Maximum bar offset for the lead-lag search.
    min_overlap : int
        Minimum non-NaN observations required to retain an asset.

    Returns
    -------
    dict with keys:
        - ``pearson``        — (AxA) Pearson correlation DataFrame
        - ``spearman``       — (AxA) Spearman correlation DataFrame
        - ``lag``            — (AxA) peak cross-correlation lag DataFrame
        - ``significance``   — permutation-test results dict
        - ``n_assets``       — int
        - ``n_observations`` — int
        - ``summary_path``   — :class:`pathlib.Path` to the saved JSON summary
    """
    # ┏━━━━━━━━━━ Save Directory ━━━━━━━━━━┓
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ┏━━━━━━━━━━ Build return matrix ━━━━━━━━━━┓
    pivot = _build_return_matrix(dataset)
    pivot = pivot.loc[:, pivot.notna().sum() >= min_overlap]
    pivot = pivot.dropna(how="all")
    n_assets = pivot.shape[1]
    n_obs    = pivot.shape[0]

    # ┏━━━━━━━━━━ Check if there are enough assets ━━━━━━━━━━┓
    if n_assets < 2:
        raise ValueError(f"Need ≥2 assets with ≥{min_overlap} observations; got {n_assets}.")

    ret_filled = pivot.fillna(0.0)

    # ┏━━━━━━━━━━ Pearson correlation ━━━━━━━━━━┓
    pearson_corr = ret_filled.corr(method="pearson")

    # ┏━━━━━━━━━━ Spearman correlation ━━━━━━━━━━┓
    sp_values, _ = spearmanr(ret_filled.values)
    if n_assets == 2:
        sp_values = np.array([[1.0, float(sp_values)], [float(sp_values), 1.0]])
    spearman_corr = pd.DataFrame(sp_values, index=pivot.columns, columns=pivot.columns)

    # ┏━━━━━━━━━━ Permutation significance test (on Pearson) ━━━━━━━━━━┓
    sig = _permutation_significance(pivot, n_permutations=n_permutations)

    # ┏━━━━━━━━━━ Lead-lag matrix ━━━━━━━━━━┓
    lag_mat = _lead_lag_matrix(pivot, max_lag=max_lag)

    # ┏━━━━━━━━━━ Summary statistics ━━━━━━━━━━┓
    off_diag = ~np.eye(n_assets, dtype=bool)
    p_vals   = pearson_corr.values[off_diag]
    s_vals   = spearman_corr.values[off_diag]

    summary = {
        "gran":           gran,
        "direction":      direction,
        "n_assets":       int(n_assets),
        "n_observations": int(n_obs),
        "pearson": {
            "mean":         float(np.nanmean(p_vals)),
            "median":       float(np.nanmedian(p_vals)),
            "min":          float(np.nanmin(p_vals)),
            "max":          float(np.nanmax(p_vals)),
            "pct_positive": float((p_vals > 0).mean()),
        },
        "spearman": {
            "mean":         float(np.nanmean(s_vals)),
            "median":       float(np.nanmedian(s_vals)),
            "min":          float(np.nanmin(s_vals)),
            "max":          float(np.nanmax(s_vals)),
            "pct_positive": float((s_vals > 0).mean()),
        },
        "permutation_test": sig,
    }

    # ┏━━━━━━━━━━ Save summary ━━━━━━━━━━┓
    summary_path = save_dir / "asset_correlation_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # ━━━━━━━━━━ Console digest ━━━━━━━━━━┓
    tag = f"{gran} {direction}".strip() or "all"
    print(f"\n[asset_correlation] {tag} | {n_assets} assets × {n_obs} bars")
    print(f"  Pearson  : mean={summary['pearson']['mean']:+.3f}  "
          f"median={summary['pearson']['median']:+.3f}  "
          f"{summary['pearson']['pct_positive']*100:.0f}% positive pairs")
    print(f"  Spearman : mean={summary['spearman']['mean']:+.3f}  "
          f"median={summary['spearman']['median']:+.3f}")
    print(f"  Perm-test: mean_r={sig['observed_mean_r']:+.3f}  "
          f"p={sig['p_value']:.4f}  z={sig['z_score']:.2f}")

    # ┏━━━━━━━━━━ Plots ━━━━━━━━━━┓
    plot_asset_correlation(pearson_corr  = pearson_corr,
                           spearman_corr = spearman_corr,
                           lag_matrix    = lag_mat,
                           pivot         = pivot,
                           sig           = sig,
                           save_dir      = save_dir,
                           gran          = gran,
                           direction     = direction)

    return {"pearson":        pearson_corr,
            "spearman":       spearman_corr,
            "lag":            lag_mat,
            "significance":   sig,
            "n_assets":       n_assets,
            "n_observations": n_obs,
            "summary_path":   summary_path}

