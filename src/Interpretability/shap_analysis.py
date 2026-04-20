# load and split data into training, validation and test sets
import sys
from pathlib import Path

# NOTE Required to use methods form Utils
sys.path.insert(0, str(Path.cwd()))  # try current dir
sys.path.insert(0, str(Path.cwd().parent))  # or parent
import Utils

import pandas as pd
import torch
from Utils.utils import _load_multi_cache
from Utils.data import split_by_global_time, ENG_FEATURE_NAMES
from Interpretability.plotting_scripts.plotting_learning_curves import plot_learning_curve

# make learning curves for all models
import os
import platform
import matplotlib
import matplotlib.pyplot as plt
import shap
import argparse
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score
from concurrent.futures import ProcessPoolExecutor, as_completed


if __name__ == "__main__":
    matplotlib.use('QtAgg')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default="/Volumes/Data/other/2026_NII/Output")
    parser.add_argument('--direction', type=str, default="up", choices=["down", "up"])
    parser.add_argument('--m1', type=str, default="Kronos", choices=["Kronos", "Fincast"])
    parser.add_argument('--gran', type=str, default="1d",
                        choices=["1d", "1h", "2h", "4h", "6h", "8h", "12h", "15m", "30m", "unified"])
    parser.add_argument('--meta_label_mode', type=str, default="tp", choices=["tp", "fp", "og"])

    parser.add_argument('--forecast_horizon', type=int, default=7)

    # parser.add_argument('--n_splits', type=int, default=100)

    args = parser.parse_args()

    if args.m1 == "Kronos" and args.direction == "down":
        hash_val = "c7ffb394d7"
    elif args.m1 == "Fincast" and args.direction == "down":
        hash_val = "46493cbe60"
    elif args.m1 == "Kronos" and args.direction == "up":
        hash_val = "7b548bc3e5"
    elif args.m1 == "Fincast" and args.direction == "up":
        hash_val = "dc96af59d5"
    else:
        raise ValueError(f"Unknown model {args.m1}")

    multi = _load_multi_cache(
        f'{args.output_root}/{args.m1}/cache/multi_{args.forecast_horizon}_fee_{args.direction}_{hash_val}.pt')

    sub = multi.sub[args.gran]
    idx_train, _, idx_val, idx_test = split_by_global_time(sub, train_end="2025-05-30", val_end="2025-10-01")
    eng_raw = sub["eng_features"].numpy() if isinstance(sub["eng_features"], torch.Tensor) else sub["eng_features"]
    labels_raw = sub["labels"].numpy() if isinstance(sub["labels"], torch.Tensor) else sub["labels"]

    X_train = pd.DataFrame(eng_raw[idx_train], columns=ENG_FEATURE_NAMES)
    y_train = labels_raw[idx_train].astype(int)

    X_val = pd.DataFrame(eng_raw[idx_val], columns=ENG_FEATURE_NAMES)
    y_val = labels_raw[idx_val].astype(int)

    X_test = pd.DataFrame(eng_raw[idx_test], columns=ENG_FEATURE_NAMES)
    y_test = labels_raw[idx_test].astype(int)


    # train model
    clf = RandomForestClassifier(n_estimators=500,
                                 max_depth=6,
                                 min_samples_leaf=20,
                                 random_state=42,
                                 n_jobs=-1,
                                 class_weight="balanced")

    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)

    # print accuracies
    print(f"train: {accuracy_score(y_train, y_pred_train):.4f} \n"
          f"val:   {accuracy_score(y_val, y_pred_val):.4f} \n"
          f"test:  {accuracy_score(y_test, y_pred_test):.4f}")

    # run shap over the train, val and test
    explainer = shap.TreeExplainer(clf)
    explanation_train = explainer(X_train)
    explanation_val = explainer(X_val)
    explanation_test = explainer(X_test)

    # plot the results
    explanations = [explanation_train, explanation_val, explanation_test]
    preds = [y_pred_train, y_pred_val, y_pred_test]
    gt = [y_train, y_val, y_test]
    titles = ["Train", "Validation", "Test"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for ax, explanation, title, pred, ground_truth in zip(axes, explanations, titles, preds, gt):
        shap.plots.beeswarm(explanation[:, :, 1], max_display=33, show=False, ax=ax, plot_size=None)
        ax.set_title(
            f"{title} | Acc = {accuracy_score(ground_truth, pred):.4f} | Pre = {precision_score(ground_truth, pred):.4f}")
    fig.suptitle(f"{args.m1} | {args.direction} | {args.gran} | {args.meta_label_mode} | {args.forecast_horizon}")
    plt.tight_layout()
    os.makedirs(
        f"{args.output_root}/{args.m1}/rf/{args.direction.upper()}/"
        f"interpretability/{args.gran}_{args.meta_label_mode}",
        exist_ok=True)
    plt.savefig(
        f"{args.output_root}/{args.m1}/rf/{args.direction.upper()}/"
        f"interpretability/{args.gran}_{args.meta_label_mode}/"
        f"overview_shap_beeswarm.pdf")
    plt.close()