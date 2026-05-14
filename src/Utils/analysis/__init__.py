"""Analysis-stage plots (meta-label diagnostics + M2 result aggregation).

Split from ``Utils/feature_selection/plots.py`` during 2026-05-14 cleanup.
"""
from Utils.analysis.analysis_meta_labels import (
    plot_class_distributions,
    plot_meta_label_returns_histogram,
    plot_m1_prediction_returns_histogram,
    plot_prediction_returns_histogram,
    plot_up_down_meta_label_distance_histograms,
    plot_asset_correlation,
    plot_dataset_size_distribution,
    plot_return_quality_distribution,
    plot_aggregate_return_quality_violins,
    plot_aggregate_all_models,
    plot_all_models_granularities,
)

from Utils.analysis.analysis_m2 import (
    plot_selective_return_distribution,
    plot_results_radar,
    plot_results_radar_focused,
    plot_kronos_down_combined,
    build_combined_metrics_dict,
    plot_results_matrices,
    plot_best_m2_per_gran,
    plot_return_heatmap,
    plot_selective_classification_vs_profitability,
)
