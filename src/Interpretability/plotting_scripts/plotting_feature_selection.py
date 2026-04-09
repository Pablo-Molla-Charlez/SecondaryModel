# load and split data into training, validation and test sets
import sys
import os
from pathlib import Path

# NOTE Required to use methods form Utils
sys.path.insert(0, str(Path.cwd()))  # try current dir
sys.path.insert(0, str(Path.cwd().parent))  # or parent

import pandas as pd
import numpy as np
import time
import torch
from Utils.utils import _load_multi_cache
from Utils.data_preprocessing import split_by_global_time, ENG_FEATURE_NAMES
from sklearn.feature_selection import SequentialFeatureSelector
from Utils.ts_cross_validation.combinatorial_purged_cv import CombinatorialPurgedCV
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate

# make learning curves for all models
import matplotlib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
import numpy as np


def plot_cv_splits(cv, X):
    n_samples = len(X)
    splits = list(cv.split(X))

    fig, ax = plt.subplots(figsize=(12, len(splits) * 0.4))

    for i, (train_idx, test_idx) in enumerate(splits):
        mask = np.zeros(n_samples)

        # mark test = 2, train = 1
        mask[train_idx] = 1
        mask[test_idx] = 2

        ax.scatter(
            range(n_samples),
            [i] * n_samples,
            c=mask,
            cmap=ListedColormap(["lightgrey", "blue", "orange"]),
            marker="s",
            s=10
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color="blue", label="Train"),
        mpatches.Patch(color="orange", label="Test"),
        mpatches.Patch(color="lightgrey", label="Purged / Unused"),
    ]
    ax.legend(handles=legend_patches, loc="upper right")

    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([f"Split {i}" for i in range(len(splits))])
    ax.set_xlabel("Sample index")
    ax.set_title("Combinatorial Purged CV Splits")

    plt.tight_layout()
    plt.show()


def plot_scoring_over_features(ret_dict, args=None):
    # df = pd.read_parquet(
    #     "/home/till/PycharmProjects/Secondary-Model/src/Output/Kronos/interpretability/feature_selection/direction=up/1d/strategy=SFS_scoring=accuracy_n_splits=3_min_max=1_5.parquet")
    #
    # ret_dict = df.to_dict(orient="list")

    # means
    mean_train = [
        np.mean(ret_dict["evaluation"][m]["train_score"])
        for m in range(len(ret_dict["evaluation"]))
    ]

    mean_val = [
        np.mean(ret_dict["evaluation"][m]["test_score"])
        for m in range(len(ret_dict["evaluation"]))
    ]

    mean_test = [
        ret_dict["test"][m]
        for m in range(len(ret_dict["evaluation"]))
    ]

    # stds
    std_train = [
        np.std(ret_dict["evaluation"][m]["train_score"])
        for m in range(len(ret_dict["evaluation"]))
    ]

    std_val = [
        np.std(ret_dict["evaluation"][m]["test_score"])
        for m in range(len(ret_dict["evaluation"]))
    ]

    x = np.arange(1, len(ret_dict["evaluation"]) + 1, 1)

    plt.figure()

    # train
    plt.plot(x, mean_train, marker="o", label="train")
    plt.fill_between(
        x,
        np.array(mean_train) - np.array(std_train),
        np.array(mean_train) + np.array(std_train),
        alpha=0.2
    )

    # test
    plt.plot(x, mean_val, marker="o", label="validation")
    plt.fill_between(
        x,
        np.array(mean_val) - np.array(std_val),
        np.array(mean_val) + np.array(std_val),
        alpha=0.2
    )

    plt.plot(x, mean_test, marker="o", label="test")

    plt.xlabel("# Features")
    if args is not None:
        plt.ylabel(f"{args.scoring.capitalize()}")
    else:
        plt.ylabel("Scoring")
    plt.ylim(0, 1)
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.show()

    if args is not None:
        os.makedirs(
            f"{args.output_root}/{args.m1}/randforest/{args.direction.upper()}/"
            f"interpretability/{args.gran}_{args.meta_label_mode}",
            exist_ok=True)
        plt.savefig(
            f"{args.output_root}/{args.m1}/randforest/{args.direction.upper()}/"
            f"interpretability/{args.gran}_{args.meta_label_mode}/"
            f"feature_selection_strategy={args.strategy}_scoring={args.scoring}_n_splits={args.n_splits}_min_max={args.min_features}_{args.max_features}.pdf")
        plt.close()
    else:
        plt.show()

