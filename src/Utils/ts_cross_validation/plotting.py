"""
This is mainly for debugging purposes
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from src.Utils.ts_cross_validation._ts_cross_validation import BaseTimeSeriesCV


def plot_cv_splits(X_analysis: pd.DataFrame, y_analysis: np.ndarray, cv: BaseTimeSeriesCV, show=True):
    matplotlib.use('QtAgg')
    
    time_index = X_analysis.index
    n_samples = len(time_index)
    indices = np.arange(n_samples)
    groups = np.array_split(indices, cv.n_splits)
    
    paths = cv.get_evaluation_path_ids()
    
    # distinct colors for paths
    path_colors = plt.cm.tab10.colors
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, (train_idx, test_idx) in enumerate(cv.split(X_analysis, y_analysis)):
        y_level = i
        
        # --- masks ---
        train_mask = np.zeros(n_samples, dtype=bool)
        train_mask[train_idx] = True
        
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[test_idx] = True
        
        purged_mask = ~(train_mask | test_mask)
        
        # --- plot ---
        ax.scatter(time_index[train_mask],
                   np.full(train_mask.sum(), y_level),
                   color="green", s=10, label="Train" if i == 0 else "")
        
        ax.scatter(time_index[test_mask],
                   np.full(test_mask.sum(), y_level),
                   color="red", s=10, label="Test" if i == 0 else "")
        
        ax.scatter(time_index[purged_mask],
                   np.full(purged_mask.sum(), y_level),
                   color="orange", s=10, label="Purged/Embargo" if i == 0 else "")
        
        # --- overlay paths ---
        n = len(paths)
        offsets = - np.linspace(0.05, 0.95, n)[::-1]
        for path_id, path in enumerate(paths):
            for xxx in path:
                split_id, group_id = xxx['split_idx'], xxx['block']
                if split_id != i:
                    continue
                
                group_idx = groups[group_id]
                
                # slight vertical offset so paths are visible
                offset = offsets[path_id]
                
                ax.scatter(time_index[group_idx],
                           np.full(len(group_idx), y_level + offset),
                           color=path_colors[path_id % len(path_colors)],
                           s=20,
                           marker="x",
                           label=f"Path {path_id}" if i == 0 else "")
    
    # --- formatting ---
    ax.set_title(f"{cv.name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("CV Split")
    ax.invert_yaxis()
    ax.legend()
    
    # important for time axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    
    plt.xticks(rotation=30)
    plt.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    if show:
        plt.show()