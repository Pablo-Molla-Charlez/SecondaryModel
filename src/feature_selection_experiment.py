# NOTE Required to use methods form Utils
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))  # try current dir
sys.path.insert(0, str(Path.cwd().parent))  # or parent
import Utils

import os
import json

# from Utils.classifier.autogluon_classifier import AutogluonClassifier
# from Utils.classifier.tabpfn_classifier import TabPFN
# from Utils.classifier.tabicl_classifier import TabICL
# from Utils.classifier.random_forest_classifier import RFClassifier
from Utils.feature_selection.sequential_feature_selection import SequentialFeatureSelection
from Utils.classifier import (MODEL_CHOICES,
                              MODELS_NO_SCALING,
                              _build_tree_model)

import pandas as pd
import time
from Utils.data_loaders.tabular_data_loader import load_tabular_dataset_from_cache_to_DataFrame
from sklearn.feature_selection import SequentialFeatureSelector, RFECV
from Utils.ts_cross_validation.combinatorial_purged_cv import CombinatorialPurgedCV
from Utils.ts_cross_validation.sklearn_ts_cv import SklearnTimeSeriesCV
from Utils.ts_cross_validation.purged_embargo_cv import PurgedEmbargoTimeSeriesCV
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate

# make learning curves for all models
import matplotlib
import argparse
from sklearn.base import clone

from Interpretability.plotting_scripts.plotting_feature_selection import plot_cv_splits, plot_scoring_over_features


# TODO remove
# def debugging(cv, X_analysis, y_analysis):
#     for i, (train_idx, test_idx) in enumerate(cv.split(X_analysis)):
#         X_train, X_test = X_analysis.iloc[train_idx], X_analysis.iloc[test_idx]
#         y_train, y_test = y_analysis[train_idx], y_analysis[test_idx]
#
#         print(f"Split {i}:")
#         print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
#         print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")
#         print("-" * 40)
#     plot_cv_splits(cv, X_analysis)  # NOTE this is for plotting the data distribution for the cv


def do_sfs(clf,
           X_analysis,
           y_analysis,
           X_test,
           y_test,
           scoring,
           cv,
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
    raise NotImplementedError
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


def do_sfs_plus(clf,
                X_analysis,
                y_analysis,
                X_test,
                y_test,
                scoring,
                cv,
                min_features=1,
                max_features=33,
                cache_feature_path=None,
                cache_feature_tag=None,
                take_n_best_combinations=3,
                can_be_parallelized=False):
    scorer = get_scorer(scoring)

    sfs = SequentialFeatureSelection(clf,
                                     scorer,
                                     cv,
                                     cache_feature_path=cache_feature_path,
                                     cache_feature_tag=cache_feature_tag,
                                     can_be_parallelized=can_be_parallelized,
                                     take_n_best_combinations=take_n_best_combinations)

    res = sfs.select_features(X_analysis, y_analysis, n_features=max_features, X_test=X_test, y_test=y_test)

    save_frame = []
    for k, val in res.items():
        best_row = val.loc[val['mean_val_scoring'].idxmax()]
        save_frame.append(best_row)
    res = pd.DataFrame(save_frame)

    return res


def do_rfecv(clf,
             X_analysis,
             y_analysis,
             X_test,
             y_test,
             scoring,
             cv,
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
    raise NotImplementedError
    ret_dict = {"feature_set": [], "evaluation": [], "test": []}

    scorer = get_scorer(scoring)

    for n_features in range(min_features, max_features + 1):
        print(f"Running SFS for {n_features} features", end="\t")
        start = time.time()
        # t1 = pd.Series(X_analysis.index)
        # cv = CombinatorialPurgedCV(n_splits=n_splits, n_test_splits=n_test_splits, mode="index",
        #                            t1=t1, embargo_pct=0.05, random_state=42)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, default=None, help="Explicit path to dataset cache .pt")
    parser.add_argument("--config", type=json.loads, help="Experiment config", required=True)
    parser.add_argument("--phase", type=str, help="Experimental Phase", required=True)
    parser.add_argument("--m2", type=str, help="M2 model to use", required=True)
    parser.add_argument("--direction", type=str, help="Direction to use", required=True)
    parser.add_argument("--granularity", type=str, help="Granularity to use", required=True)

    args = parser.parse_args()

    X_analysis, y_analysis, X_test, y_test, returns_analysis, asset_ids_analysis, returns_test, asset_ids_test, asset_map = load_tabular_dataset_from_cache_to_DataFrame(
        cache_path=args.cache_path,
        gran=args.granularity)

    # select model
    clf = _build_tree_model(args.m2, X_analysis.shape[0])  # TODO number of samples not used

    print(f"[Feature selection] Is parallelizable: {True if args.m2 == "randforest" else False}")
    # select cross validation strategy
    if args.config["runtime"][args.phase]["cv_strategy"] == "cpcv":
        cv = CombinatorialPurgedCV(n_splits=args.config["runtime"][args.phase]["n_blocks"],
                                   n_test_splits=args.config["runtime"][args.phase]["k_test"],
                                   embargo_pct=0.05,
                                   random_state=42)
    elif args.config["runtime"][args.phase]["cv_strategy"] == "tscv":
        cv = SklearnTimeSeriesCV(n_splits=args.config["runtime"][args.phase]["n_blocks"],
                                 random_state=42)
    elif args.config["runtime"][args.phase]["cv_strategy"] == "pecv":
        cv = PurgedEmbargoTimeSeriesCV(n_splits=args.config["runtime"][args.phase]["n_blocks"],
                                       embargo_pct=0.05,
                                       random_state=42)
    else:
        raise ValueError(f"Unknown CV strategy {args.cv_strategy}")

    if args.config["runtime"][args.phase]["method"] == "sfs":
        res = do_sfs(clf,
                     X_analysis,
                     y_analysis,
                     X_test,
                     y_test,
                     scoring=args.config["runtime"][args.phase]["scoring"],
                     cv=cv,
                     min_features=args.config["runtime"][args.phase]["min_features"],
                     max_features=args.config["runtime"][args.phase]["max_features"])
    elif args.config["runtime"][args.phase]["method"] == "sfs+":
        res = do_sfs_plus(clf,
                          X_analysis,
                          y_analysis,
                          X_test,
                          y_test,
                          scoring=args.config["runtime"][args.phase]["scoring"],
                          cv=cv,
                          min_features=args.config["runtime"][args.phase]["min_features"],
                          max_features=args.config["runtime"][args.phase]["max_features"],
                          cache_feature_path=f"{args.config["paths"]["output_root"]}/"
                                             f"{args.config["experiment"]["m1"].capitalize()}/"
                                             f"{args.m2}/"
                                             f"{args.direction.upper()}/"
                                             f"interpretability/"
                                             f"feature_selection/"
                                             f"{args.granularity}_{args.config["data"]["load"]["meta_label_mode"]}/",
                          cache_feature_tag=f"{args.config["runtime"][args.phase]["scoring"]}_{cv.name}_{args.config["runtime"][args.phase]["n_blocks"]}",
                          take_n_best_combinations=args.config["runtime"][args.phase]["take_n_best_combinations"],
                          can_be_parallelized=True if args.m2 == "randforest" else False)
    elif args.config["runtime"][args.phase]["method"] == "rfecv":
        res = do_rfecv(clf,
                       X_analysis,
                       y_analysis,
                       X_test,
                       y_test,
                       scoring=args.config["runtime"][args.phase]["scoring"],
                       cv=cv,
                       min_features=args.config["runtime"][args.phase]["min_features"],
                       max_features=args.config["runtime"][args.phase]["max_features"])
    else:
        raise ValueError(f"Unknown strategy {args.strategy}")

    save_dir_path = f"{args.config["paths"]["output_root"]}/" \
                    f"{args.config["experiment"]["m1"].capitalize()}/" \
                    f"{args.m2}/" \
                    f"{args.direction.upper()}/" \
                    f"interpretability/" \
                    f"feature_selection/" \
                    f"{args.granularity}_{args.config["data"]["load"]["meta_label_mode"]}/"

    os.makedirs(save_dir_path, exist_ok=True)
    res.to_csv(os.path.join(save_dir_path,
                            f"strategy={args.config["runtime"][args.phase]["method"]}_scoring={args.config["runtime"][args.phase]["scoring"]}_n_splits={args.config["runtime"][args.phase]["n_blocks"]}_min_max={args.config["runtime"][args.phase]["min_features"]}_{args.config["runtime"][args.phase]["max_features"]}.csv"),
               index=False)
