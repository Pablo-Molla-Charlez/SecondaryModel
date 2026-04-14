from typing import Optional, Iterator, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.base import clone
import warnings
from joblib import Parallel, delayed

from Utils.feature_selection._feature_selection import (FeatureSelection)


class SequentialFeatureSelection(FeatureSelection):

    def __init__(self, clf, scoring, cross_validation_strategy, **kwargs):
        super().__init__(clf, scoring, cross_validation_strategy, **kwargs)

        self.cache_path = kwargs.get("cache_path", None)
        self.cache_name = kwargs.get("cache_name", None)

        self.take_n_best_combinations = kwargs.get("take_n_best_combinations", 3)
        self.can_be_parallelized = kwargs.get("can_be_parallelized", False)

    def select_features(
            self,
            X: pd.DataFrame,
            y: np.ndarray,
            n_features: int,
            X_test: Optional[pd.DataFrame] = None,
            y_test: Optional[np.ndarray] = None,
            **kwargs
    ) -> pd.DataFrame:

        res_dict = {}
        feature_list = list(X.columns)

        done_combinations = set()

        for k_features in range(1, n_features + 1):  # features evaluation loops
            print(f"running {k_features} features")
            check_for_evaluations_done = False  # used for checking that we have any evaluations
            tmp_res = []
            if k_features == 1:  # initial state
                # evaluate each feature
                if self.can_be_parallelized:
                    tmp_res = Parallel(n_jobs=-1)(
                        delayed(self._evaluate_feature)(feat, X, y, X_test, y_test)
                        for feat in feature_list
                    )
                else:
                    for feat in feature_list:
                        print(f"{k_features} -> {feat}")
                        scores_val, scores_test = self._evaluate(pd.DataFrame(X[feat]), y, pd.DataFrame(X_test[feat]),
                                                                 y_test)
                        tmp_res.append(
                            {"features_selected": [feat],
                             "mean_val_scoring": np.mean(scores_val),
                             "std_val_scoring": np.std(scores_val),
                             "val_scoring": scores_val,
                             "mean_test_scoring": np.nanmean(scores_test) if scores_test is not None else np.nan,
                             "std_test_scoring": np.nanstd(scores_test) if scores_test is not None else np.nan,
                             "test_scoring": scores_test if scores_test is not None else np.nan}
                        )
                check_for_evaluations_done = True
            else:  # we have a previous state
                if self.can_be_parallelized:
                    tasks = []
                    for feature_set in res_dict[f"{k_features - 1}_features"]["features_selected"].iloc[
                        0:self.take_n_best_combinations]:
                        for feat in [f for f in feature_list if f not in feature_set]:
                            tmp_feature_list = [f for sublist in feature_set + [feat] for f in
                                                (sublist if isinstance(sublist, list) else [sublist])]
                            if tuple(sorted(tmp_feature_list)) in done_combinations:
                                continue
                            done_combinations.add(tuple(sorted(tmp_feature_list)))
                            check_for_evaluations_done = True
                            tasks.append((feature_set, feat))

                    if tasks:
                        tmp_res = Parallel(n_jobs=-1)(
                            delayed(self._evaluate_feature_set)(feature_set, feat, X, y, X_test, y_test)
                            for feature_set, feat in tasks
                        )
                else:
                    for feature_set in res_dict[f"{k_features - 1}_features"]["features_selected"].iloc[
                        0:self.take_n_best_combinations]:
                        # feature_set = feature_set
                        for feat in [f for f in feature_list if f not in feature_set]:
                            print(f"{k_features}: {feature_set} -> {feat}")
                            # ensure that we check each combination once only
                            tmp_feature_list = [f for sublist in feature_set + [feat] for f in
                                                (sublist if isinstance(sublist, list) else [sublist])]
                            if tuple(sorted(tmp_feature_list)) in done_combinations:
                                continue
                            done_combinations.add(tuple(sorted(tmp_feature_list)))
                            print(f"    tmp_feature_list: {tmp_feature_list}")
                            scores_val, scores_test = self._evaluate(X[tmp_feature_list], y, X_test[tmp_feature_list],
                                                                     y_test)
                            tmp_res.append(
                                {"features_selected": tmp_feature_list,
                                 "mean_val_scoring": np.mean(scores_val),
                                 "std_val_scoring": np.std(scores_val),
                                 "val_scoring": scores_val,
                                 "mean_test_scoring": np.nanmean(scores_test) if scores_test is not None else np.nan,
                                 "std_test_scoring": np.nanstd(scores_test) if scores_test is not None else np.nan,
                                 "test_scoring": scores_test if scores_test is not None else np.nan}
                            )
                            check_for_evaluations_done = True
            if check_for_evaluations_done:
                tmp_res = pd.DataFrame(tmp_res).sort_values("mean_val_scoring", ascending=False)
                # cache results
                if self.cache_path is not None:
                    os.makedirs(self.cache_path, exist_ok=True)
                    tmp_res.to_csv(f"{self.cache_path}/{k_features}_features_{self.cache_name}_cached.csv", index=False)
                res_dict[f"{k_features}_features"] = tmp_res
            else:
                warnings.warn("No more subsests to check ending feature selection")
                # n_features = k_features - 1
                break
        # make last entry to pd.Dataframe
        return res_dict  # [f"{n_features}_features"]

    def _evaluate(self, X, y, X_test, y_test) -> Tuple[list, list]:
        split_scoring = []
        if X_test is not None and y_test is not None:
            split_scoring_test = []
        else:
            split_scoring_test = None
        for i_train_split, i_val_split in self.cross_validation_strategy.split(X, y):
            fitted_classifier = clone(self.clf).fit(X.iloc[i_train_split], y[i_train_split])
            # y_pred = fitted_classifier.predict(X.iloc[i_val_split])
            # print(y[i_val_split].shape, y_pred.shape)
            split_scoring.append(self.scoring(fitted_classifier, X.iloc[i_val_split], y[i_val_split]))
            if X_test is not None and y_test is not None:
                split_scoring_test.append(self.scoring(fitted_classifier, X_test, y_test))

        return split_scoring, split_scoring_test

    def _evaluate_feature(self, feat, X, y, X_test, y_test):
        scores_val, scores_test = self._evaluate(pd.DataFrame(X[feat]), y, pd.DataFrame(X_test[feat]), y_test)
        return {"features_selected": [feat],
                "mean_val_scoring": np.mean(scores_val),
                "std_val_scoring": np.std(scores_val),
                "val_scoring": scores_val,
                "mean_test_scoring": np.nanmean(scores_test) if scores_test is not None else np.nan,
                "std_test_scoring": np.nanstd(scores_test) if scores_test is not None else np.nan,
                "test_scoring": scores_test if scores_test is not None else np.nan}

    def _evaluate_feature_set(self, feature_set, feat, X, y, X_test, y_test):
        tmp_feature_list = [f for sublist in feature_set + [feat] for f in
                            (sublist if isinstance(sublist, list) else [sublist])]
        scores_val, scores_test = self._evaluate(X[tmp_feature_list], y, X_test[tmp_feature_list], y_test)

        return {"features_selected": tmp_feature_list,
                "mean_val_scoring": np.mean(scores_val),
                "std_val_scoring": np.std(scores_val),
                "val_scoring": scores_val,
                "mean_test_scoring": np.nanmean(scores_test) if scores_test is not None else np.nan,
                "std_test_scoring": np.nanstd(scores_test) if scores_test is not None else np.nan,
                "test_scoring": scores_test if scores_test is not None else np.nan}
