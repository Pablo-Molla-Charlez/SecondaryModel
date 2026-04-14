import os
from typing import Union
import warnings

import numpy as np
import pandas as pd

from Utils.classifier._classifier import (BaseClassifier)
from autogluon.tabular import TabularPredictor

import hashlib


class AutogluonClassifier(BaseClassifier):

    def __init__(self, random_state=None, **kwargs) -> None:
        super().__init__(random_state)
        self.args = kwargs.pop("args", None)  # pop so it's not passed to TabularPredictor
        if self.args is None:
            raise ValueError("Argument 'args' is None -- Please pass it!")
        self.model_cache_path = (
            f"{self.args.output_root}/{self.args.m1}/interpretability/feature_selection/{self.args.m2}/"
            f"direction={self.args.direction}/{self.args.gran}")
        self.time_limit = 60

    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: np.ndarray) -> object:
        # TODO the idea is to only run the hyperparameter optimization once
        #  cache the optimized model and retrain it on the subsplit
        # get hased feature list
        feature_list_hash = self.hash_list(list(X_train.columns))
        self.model_cache_path = self.model_cache_path + f"/feature_hash={feature_list_hash}_timelimit={self.time_limit}/autogluon_cache"
        train_data = X_train.copy()
        train_data["target_column"] = y_train
        if os.path.exists(self.model_cache_path):
            print(f"{self.model_cache_path} exists, loading cached autogluon model...")
            self.load_model(self.model_cache_path)
            self.clf.refit_full(model="all", set_best_to_refit_full=True, train_data_extra=train_data)
            self.fitted_clf = self.clf
        else:
            print(f"{self.model_cache_path} does not exist, need to train autogluon model...")
            # train
            TabularPredictor(label="target_column", path=self.model_cache_path).fit(
                train_data,
                time_limit=self.time_limit,  # seconds to train
                presets="best_quality",  # or "medium_quality", "fast_training"
            )
            self.clf = TabularPredictor.load(self.model_cache_path)
            self.fitted_clf = self.clf
        return self.fitted_clf


    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fitted_clf.predict(X_test)

    def predict_proba(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fitted_clf.predict_proba(X_test)

    def get_params(self, deep: bool = True) -> dict:
        return {"random_state": self.random_state,
                "args": self.args}

    def save_model(self, model_path: str) -> None:
        warnings.warn("AutoGluon does not need to save, happens automatically in constructor.")

    def load_model(self, model_path: str) -> None:
        self.clf = TabularPredictor.load(model_path)
        self.fitted_clf = self.clf

    def hash_list(self, lst: list, length: int = 10) -> str:
        combined = ",".join(sorted(lst))  # sort for consistent ordering
        return hashlib.md5(combined.encode()).hexdigest()[:length]
