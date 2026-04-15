"""Feature analysis and plotting helpers extracted from kronos_tree.py."""

import warnings
import re
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats
from xgboost import XGBClassifier
from typing import Dict, List, Optional, Any, Union


from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             precision_recall_fscore_support, confusion_matrix, 
                             ConfusionMatrixDisplay, fbeta_score, matthews_corrcoef)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from Utils.utils import model_label
from Utils.data import _get_from_dataset, get_dynamic_ret_limits, MultiGranDataset


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rank Aggregation → Top-K feature selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORR_THRESHOLD = 0.85
TOP_K = 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MDA / SHAP / LIME  FEATURE RANKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_RNG = np.random.default_rng(42)


__all__ = [
    "plot_correlation_heatmap",
    "plot_pointbiserial",
    "plot_class_distributions",
    "plot_mutual_information",
    "plot_tree_importance",
    "_plot_prob_distribution",
    "plot_class_distribution",
    "plot_meta_label_returns_histogram",
    "plot_prediction_returns_histogram",
    "plot_m1_prediction_returns_histogram",
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "extract_time_features",
    "plot_temporal_risk_coverage_curve",
    "plot_ocp_threshold_evolution",
    "compute_top_features",
    "mda_rank",
    "shap_rank",
    "lime_rank",
    "combine_rankings",
    "run_feature_selection",
    "plot_selective_return_distribution",
]




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


# ┏━━━━━━━━━━ Rank Features ━━━━━━━━━━┓
def _rank_dict(scores: dict, higher_is_better: bool = True) -> dict:
    """Rank features by score. Rank 1 = best."""
    sorted_feats = sorted(scores.keys(), key=lambda f: scores[f], reverse=higher_is_better)
    return {f: rank + 1 for rank, f in enumerate(sorted_feats)}



# ┏━━━━━━━━━━ Dominant Metric ━━━━━━━━━━┓
def _dominant_metric(feat: str, ranks: dict) -> str:
    """Return which metric this feature ranks best in."""
    best_metric = min(ranks, key=lambda m: ranks[m].get(feat, 999))
    return best_metric



# ┏━━━━━━━━━━ Compute Top Features ━━━━━━━━━━┓
def compute_top_features(pb_scores: dict, mi_scores: dict, rf_scores: dict,
                         corr_matrix: pd.DataFrame, save_dir: Path) -> dict:
    """
    Rank aggregation across metrics + correlation-aware deduplication.
    Returns the top-K dict for inclusion in analysis_summary.json.
    """
    features = list(pb_scores.keys())

    # ┏━━━━━━━━━━ Rank Features ━━━━━━━━━━┓
    ranks = {"point_biserial": _rank_dict(pb_scores, higher_is_better=True),
             "mutual_info":    _rank_dict(mi_scores, higher_is_better=True),
             "rf_importance":  _rank_dict(rf_scores, higher_is_better=True)}

    # ┏━━━━━━━━━━ Average Rank per Feature ━━━━━━━━━━┓
    avg_ranks = {}
    for f in features:
        r_pb = ranks["point_biserial"].get(f, len(features))
        r_mi = ranks["mutual_info"].get(f, len(features))
        r_rf = ranks["rf_importance"].get(f, len(features))
        avg_ranks[f] = (r_pb + r_mi + r_rf) / 3.0

    # ┏━━━━━━━━━━ Sort by Avg Rank ━━━━━━━━━━┓
    sorted_features = sorted(features, key=lambda f: (avg_ranks[f], ranks["mutual_info"].get(f, 999)))

    # ┏━━━━━━━━━━ Greedy Selection with Correlation Check ━━━━━━━━━━┓
    selected = []
    correlation_warnings = []

    # ┏━━━━━━━━━━ Iterate through sorted features ━━━━━━━━━━┓
    for feat in sorted_features:
        if len(selected) >= TOP_K:
            break

        # ┏━━━━━━━━━━ Check Correlation with Already-Selected Features ━━━━━━━━━━┓
        highly_correlated_with = None
        for sel in selected:
            if feat in corr_matrix.index and sel in corr_matrix.columns:
                r = abs(corr_matrix.loc[feat, sel])
                if r >= CORR_THRESHOLD:
                    highly_correlated_with = (sel, r)
                    break
        
        # ┏━━━━━━━━━━ Check if they bring different strengths ━━━━━━━━━━┓
        if highly_correlated_with:
            sel_feat, r_val = highly_correlated_with
            feat_best = _dominant_metric(feat, ranks)
            sel_best = _dominant_metric(sel_feat, ranks)

            # ┏━━━━━━━━━━ Different Strengths — Keep Both ━━━━━━━━━━┓
            if feat_best != sel_best:
                selected.append(feat)
                correlation_warnings.append({"pair": [sel_feat, feat],
                                             "pearson_r": round(float(r_val), 3),
                                             "action": "kept_both",
                                             "reason": f"{sel_feat} dominates {sel_best}, {feat} dominates {feat_best}"})
            else:
                # ┏━━━━━━━━━━ Same Dominant Metric — Skip, Try Next ━━━━━━━━━━┓
                correlation_warnings.append({"pair": [sel_feat, feat],
                                             "pearson_r": round(float(r_val), 3),
                                             "action": "skipped",
                                             "reason": f"both dominate {feat_best}, kept {sel_feat} (better avg rank)"})
        else:
            selected.append(feat)

    # ┏━━━━━━━━━━ Build Detailed Output ━━━━━━━━━━┓
    top_details = []
    for f in selected:
        top_details.append({"feature": f,
                            "avg_rank": round(avg_ranks[f], 2),
                            "pb_rank": ranks["point_biserial"].get(f),
                            "mi_rank": ranks["mutual_info"].get(f),
                            "rf_rank": ranks["rf_importance"].get(f),
                            "pb_score": round(pb_scores.get(f, 0), 6),
                            "mi_score": round(mi_scores.get(f, 0), 6),
                            "rf_score": round(rf_scores.get(f, 0), 6),
                            "dominant_metric": _dominant_metric(f, ranks)})

    result = {"top_5_features": selected,
              "top_5_details": top_details,
              "correlation_warnings": correlation_warnings}

    # ┏━━━━━━━━━━ Print to Terminal ━━━━━━━━━━┓
    print(f"  [6/9] Top-{TOP_K} features (rank aggregation + correlation check):")
    for d in top_details:
        print(f"    {d['feature']:20s}  avg_rank={d['avg_rank']:.1f}  "
              f"PB={d['pb_rank']}  MI={d['mi_rank']}  RF={d['rf_rank']}  "
              f"dominant={d['dominant_metric']}")
    if correlation_warnings:
        print(f"  Correlation warnings:")
        for w in correlation_warnings:
            print(f"    {w['pair'][0]} <-> {w['pair'][1]}  |r|={w['pearson_r']}  → {w['action']} ({w['reason']})")

    # ┏━━━━━━━━━━ Save standalone CSV of full ranking ━━━━━━━━━━┓
    full_rank_rows = []
    for f in sorted_features:
        full_rank_rows.append({"feature": f,
                               "avg_rank": round(avg_ranks[f], 2),
                               "pb_rank": ranks["point_biserial"].get(f),
                               "mi_rank": ranks["mutual_info"].get(f),
                               "rf_rank": ranks["rf_importance"].get(f),
                               "selected": f in selected})

    pd.DataFrame(full_rank_rows).to_csv(save_dir / "6_rank_aggregation.csv", index=False)

    return result


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
      
    import json
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

# Back-import plot functions so topic-level code can call them.
from .plots import *  # noqa: E402,F401,F403
