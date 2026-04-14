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
from Utils.data import split_by_global_time, ENG_FEATURE_NAMES
from sklearn.feature_selection import SequentialFeatureSelector, RFECV
from Utils.ts_cross_validation.combinatorial_purged_cv import CombinatorialPurgedCV
from Utils.classifier.random_forest_classifier import RFClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate

# make learning curves for all models
import matplotlib
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import numpy as np
from Interpretability.plotting_scripts.plotting_feature_selection import plot_cv_splits, plot_scoring_over_features


def debugging(cv, X_analysis, y_analysis):
    for i, (train_idx, test_idx) in enumerate(cv.split(X_analysis)):
        X_train, X_test = X_analysis.iloc[train_idx], X_analysis.iloc[test_idx]
        y_train, y_test = y_analysis[train_idx], y_analysis[test_idx]

        print(f"Split {i}:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")
        print("-" * 40)
    plot_cv_splits(cv, X_analysis)  # NOTE this is for plotting the data distribution for the cv


def do_sfs(clf,
           X_analysis,
           y_analysis,
           X_test,
           y_test,
           scoring,
           n_splits=10,
           n_test_splits=2,
           direction='forward',
           n_jobs=20,
           min_features=1,
           max_features=33) -> dict:
    """
    Do sequential feature selection for a given classifier and dataset (X, y)
    Args:
        clf: any classifier that implements .fit() and .predict() methods
        X_analysis: Feature set as pandas dataframe (rows = samples, columns = features)
        y_analysis: labels of the feature set as numpy array
        scoring: scoring function
        n_splits: number of splits to use
        n_test_splits: (Optional) number of test splits to use (depends on the cross validation method)
        direction: 'forward' or 'backward'
        n_jobs: number of jobs to run in parallel
        min_features: minimal features to use
        max_features: maximal features to use

    Returns:
        {"feature_set": [feature names], "evaluation": [accuracy splits]}
    """
    ret_dict = {"feature_set": [], "evaluation": [], "test": []}

    scorer = get_scorer(scoring)

    for n_features in range(min_features, max_features + 1):
        print(f"Running SFS for {n_features} features", end="\t")
        start = time.time()
        t1 = pd.Series(X_analysis.index)
        cv = CombinatorialPurgedCV(n_splits=n_splits, n_test_splits=n_test_splits, mode="index",
                                   t1=t1, embargo_pct=0.05, random_state=42)
        # debugging(cv, X_analysis, y_analysis)
        if n_features == X_analysis.shape[1]:
            # recalculate scoring
            selected_features = [True for _ in range(X_analysis.shape[1])]
            scores = cross_validate(
                clone(clf),
                X_analysis.iloc[:, selected_features],
                y_analysis,
                cv=cv,
                scoring=scorer,
                return_train_score=True,
                n_jobs=n_jobs,
            )
            ret_dict["feature_set"].append(list(X_analysis.columns))
            ret_dict["evaluation"].append(scores)
        else:
            sfs = SequentialFeatureSelector(clone(clf), n_features_to_select=n_features, direction=direction,
                                            scoring=scorer, cv=cv, n_jobs=n_jobs)
            sfs.fit(X_analysis, y_analysis)

            # recalculate scoring
            selected_features = sfs.get_support(indices=True)
            scores = cross_validate(
                clone(clf),
                X_analysis.iloc[:, selected_features],
                y_analysis,
                cv=cv,
                scoring=scorer,
                return_train_score=True,
                n_jobs=n_jobs,
            )
            ret_dict["feature_set"].append(sfs.get_feature_names_out())
            ret_dict["evaluation"].append(scores)

        # get test score
        clf_fitted = clone(clf).fit(X_analysis.iloc[:, selected_features], y_analysis)
        ret_dict["test"].append(scorer(clf_fitted, X_test.iloc[:, selected_features], y_test))

        end = time.time()
        print(f"computed for {end - start:.2f} second(s)")

    return ret_dict


def do_rfecv(clf,
           X_analysis,
           y_analysis,
           X_test,
           y_test,
           scoring,
           n_splits=10,
           n_test_splits=2,
           n_jobs=20,
           min_features=1,
           max_features=33) -> dict:
    """
    Do recursive feature elimination with cross-validation for a given classifier and dataset (X, y)
    IMPORTANT: max_features is not considered here. You would need to pre-filter features that should not be considered
    Args:
        clf: any classifier that implements .fit() and .predict() methods
        X_analysis: Feature set as pandas dataframe (rows = samples, columns = features)
        y_analysis: labels of the feature set as numpy array
        scoring: scoring function
        n_splits: number of splits to use
        n_test_splits: (Optional) number of test splits to use (depends on the cross validation method)
        direction: 'forward' or 'backward'
        n_jobs: number of jobs to run in parallel
        min_features: minimal features to use
        max_features: maximal features to use

    Returns:
        {"feature_set": [feature names], "evaluation": [accuracy splits]}
    """
    ret_dict = {"feature_set": [], "evaluation": [], "test": []}

    scorer = get_scorer(scoring)

    for n_features in range(min_features, max_features + 1):
        print(f"Running SFS for {n_features} features", end="\t")
        start = time.time()
        t1 = pd.Series(X_analysis.index)
        cv = CombinatorialPurgedCV(n_splits=n_splits, n_test_splits=n_test_splits, mode="index",
                                   t1=t1, embargo_pct=0.05, random_state=42)
        # debugging(cv, X_analysis, y_analysis)
        if n_features == X_analysis.shape[1]:
            # recalculate scoring
            selected_features = [True for _ in range(X_analysis.shape[1])]
            scores = cross_validate(
                clone(clf),
                X_analysis.iloc[:, selected_features],
                y_analysis,
                cv=cv,
                scoring=scorer,
                return_train_score=True,
                n_jobs=n_jobs,
            )
            ret_dict["feature_set"].append(list(X_analysis.columns))
            ret_dict["evaluation"].append(scores)
        else:
            rfecv = RFECV(clone(clf), min_features_to_select=n_features,
                                            scoring=scorer, cv=cv, n_jobs=n_jobs)
            rfecv.fit(X_analysis, y_analysis)

            # recalculate scoring
            selected_features = rfecv.get_support(indices=True)
            scores = cross_validate(
                clone(clf),
                X_analysis.iloc[:, selected_features],
                y_analysis,
                cv=cv,
                scoring=scorer,
                return_train_score=True,
                n_jobs=n_jobs,
            )
            ret_dict["feature_set"].append(rfecv.get_feature_names_out())
            ret_dict["evaluation"].append(scores)

        # get test score
        clf_fitted = clone(clf).fit(X_analysis.iloc[:, selected_features], y_analysis)
        ret_dict["test"].append(scorer(clf_fitted, X_test.iloc[:, selected_features], y_test))

        end = time.time()
        print(f"computed for {end - start:.2f} second(s)")

    return ret_dict


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

    # Specific to feature selection
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--strategy', type=str, default="SFS", choices=["SFS", "SFS+", "RFECV"])
    parser.add_argument('--scoring', type=str, default="accuracy", choices=["accuracy", "precision", "roc_auc"])
    parser.add_argument('--min_features', type=int, default=1)
    parser.add_argument('--max_features', type=int, default=33)

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

    # merge X_train and X_val
    X_analysis = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_analysis = np.concatenate([y_train, y_val])

    # train model
    clf = RFClassifier()

    if args.strategy == "SFS":
        res = do_sfs(clf,
                     X_analysis,
                     y_analysis,
                     X_test,
                     y_test,
                     args.scoring,
                     n_splits=args.n_splits,
                     min_features=args.min_features,
                     max_features=args.max_features)
    elif args.strategy == "SFS+":
        raise ValueError("SFS+ not implemented yet")
    elif args.strategy == "RFECV":
        res = do_rfecv(clf,
                     X_analysis,
                     y_analysis,
                     X_test,
                     y_test,
                     args.scoring,
                     n_splits=args.n_splits,
                     min_features=args.min_features,
                     max_features=args.max_features)
    else:
        raise ValueError(f"Unknown strategy {args.strategy}")

    # for features, score in zip(res["feature_set"], res["evaluation"]):
    #     print(f"n_features: {len(features)} | score: {score} | features: {features}")

    # save results to csv?
    df = pd.DataFrame(res)

    save_dir_path = (f"{args.output_root}/{args.m1}/interpretability/feature_selection/"
                     f"direction={args.direction}/{args.gran}/")
    os.makedirs(save_dir_path, exist_ok=True)
    df.to_parquet(os.path.join(save_dir_path,
                               f"strategy={args.strategy}_scoring={args.scoring}_n_splits={args.n_splits}_min_max={args.min_features}_{args.max_features}.parquet"))

    plot_scoring_over_features(res, args)
