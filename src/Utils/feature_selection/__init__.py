"""Feature selection + feature-level plotting (package).

Meta-label diagnostics and M2 result-aggregation plots live in
``Utils.analysis``; CPCV/edge plots live in ``Utils.edge.plots``.
"""

# ┏━━━━━━━━━━ Feature Selection ━━━━━━━━━━┓
from Utils.feature_selection.feature_selection import (run_feature_selection,
                                                       compute_classification_metrics,
                                                       combine_rankings,
                                                       mda_rank,
                                                       shap_rank,
                                                       lime_rank,
                                                       extract_time_features,
                                                       compute_asset_correlation)
# ┏━━━━━━━━━━ Feature Selection Plots ━━━━━━━━━━┓
from Utils.feature_selection.plots import (plot_pointbiserial,
                                           plot_mutual_information,
                                           plot_correlation_heatmap,
                                           plot_confusion_matrix,
                                           plot_temporal_risk_coverage_curve_final,
                                           plot_performance_over_n_features)
