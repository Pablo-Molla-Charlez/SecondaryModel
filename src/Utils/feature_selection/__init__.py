"""Feature selection + feature-level plotting (package).

Split from ``_features_impl.py`` into:

- :mod:`feature_selection` — MDA / SHAP / LIME rankings, rank combination,
  top-feature selection, classification metrics, main ``run_feature_selection``
- :mod:`plots` — all ``plot_*`` helpers (tree importance, confusion matrices,
  risk-coverage curves, OCP threshold evolution, return histograms, etc.)
"""
from Utils.feature_selection.feature_selection import (
    run_feature_selection,
    compute_top_features,
    compute_classification_metrics,
    combine_rankings,
    mda_rank,
    shap_rank,
    lime_rank,
    extract_time_features,
    compute_asset_correlation,
)
from Utils.feature_selection.plots import (
    plot_tree_importance,
    plot_pointbiserial,
    plot_mutual_information,
    plot_correlation_heatmap,
    plot_class_distributions,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_temporal_risk_coverage_curve,
    plot_temporal_risk_coverage_curve_final,
    plot_ocp_threshold_evolution,
    plot_selective_return_distribution,
    plot_meta_label_returns_histogram,
    plot_prediction_returns_histogram,
    plot_m1_prediction_returns_histogram,
    plot_asset_correlation,
    _plot_prob_distribution,
)