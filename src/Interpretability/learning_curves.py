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
from Utils.data_preprocessing import split_by_global_time, ENG_FEATURE_NAMES

# make learning curves for all models
import os
import argparse
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, precision_score
from concurrent.futures import ProcessPoolExecutor, as_completed


# utility functions
def exponential_schedule(max_samples, n_steps=20, start=10):
    schedule = np.logspace(
        np.log10(start),
        np.log10(max_samples),
        num=n_steps
    )
    return schedule.astype(int)


def fit_classifier_parallel(x_train, y_train, x_val, y_val, x_test, y_test, classifier, i_train):
    # clone model
    fitted_classifier = clone(classifier).fit(x_train.iloc[i_train], y_train[i_train])
    
    y_pred = fitted_classifier.predict(x_val)
    acc_val = accuracy_score(y_val, y_pred)
    prec_val = precision_score(y_val, y_pred)
    acc_test = accuracy_score(y_test, y_pred)
    prec_test = precision_score(y_test, y_pred)
    
    return acc_val, prec_val, acc_test, prec_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default="/Volumes/Data/other/2026_NII/Output")
    parser.add_argument('--direction', type=str, default="down", choices=["down", "up"])
    parser.add_argument('--m1', type=str, default="Kronos", choices=["Kronos", "Fincast"])
    parser.add_argument('--gran', type=str, default="1d",
                        choices=["1d", "1h", "2h", "4h", "6h", "8h", "12h", "15min", "30min", "unified"])
    
    parser.add_argument('--forecast_horizon', type=int, default=7)
    
    parser.add_argument('--n_splits', type=int, default=100)
    
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
    
    # print(f"train: {X_train.shape} | Features: {X_train.columns}")
    # print(f"val: {X_val.shape} | Features: {X_val.columns}")
    # print(f"test: {X_test.shape} | Features: {X_test.columns}")
    print(f"Done loading ....")
    
    max_samples = X_train.shape[0]  # get number of training samples
    
    clf = RandomForestClassifier(n_estimators=500,
                                 max_depth=6,
                                 min_samples_leaf=20,
                                 random_state=42,
                                 n_jobs=-1,
                                 class_weight="balanced")
    
    save_dict_acc_val = defaultdict(list)
    save_dict_pre_val = defaultdict(list)
    save_dict_acc_test = defaultdict(list)
    save_dict_pre_test = defaultdict(list)
    print("Starting training with exponential schedule...")
    schedule = exponential_schedule(max_samples=max_samples - 2,
                                    n_steps=10)  # NOTE 2 because of 2 classes and we need at least 1 sample per class
    for training_quota in schedule:
        print(f"\tTraining quota: {training_quota:.2f}")
        # stratified shuffle split
        cv = StratifiedShuffleSplit(n_splits=args.n_splits, train_size=training_quota, random_state=42)
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # now you can fit
            futures = [
                executor.submit(fit_classifier_parallel, X_train, y_train, X_val, y_val, X_test, y_test, clf,
                                i_train_split) for
                i_train_split, _ in cv.split(X_train, y_train)]
            
            # await results
            as_completed([f for f in futures])
        
        for i, future in enumerate(futures):
            # order: accuracy, precision
            acc_val, prec_val, acc_test, prec_test = future.result()
            save_dict_acc_val[f"{training_quota:.2f}"].append(acc_val)
            save_dict_pre_val[f"{training_quota:.2f}"].append(prec_val)
            save_dict_acc_test[f"{training_quota:.2f}"].append(acc_test)
            save_dict_pre_test[f"{training_quota:.2f}"].append(prec_test)
            
    # save to files
    if not os.path.exists(
        f"{args.output_root}/{args.m1}/interpretability/learning_curves/direction={args.direction}/{args.gran}/"):
        os.makedirs(f"{args.output_root}/{args.m1}/interpretability/learning_curves/")
    
    
    save_frame = pd.DataFrame(save_dict_acc_val)
    save_frame.to_csv(
        f"{args.output_root}/{args.m1}/interpretability/learning_curves/direction={args.direction}/{args.gran}/"
        f"acc_val_multi_{args.forecast_horizon}_fee_n_splits={args.n_splits}.csv",
        index=False)
    save_frame = pd.DataFrame(save_dict_acc_test)
    save_frame.to_csv(
        f"{args.output_root}/{args.m1}/interpretability/learning_curves/direction={args.direction}/{args.gran}/"
        f"acc_test_multi_{args.forecast_horizon}_fee_n_splits={args.n_splits}.csv",
        index=False)
    
    save_frame = pd.DataFrame(save_dict_pre_val)
    save_frame.to_csv(
        f"{args.output_root}/{args.m1}/interpretability/learning_curves/direction={args.direction}/{args.gran}/"
        f"precision_val_multi_{args.forecast_horizon}_fee_n_splits={args.n_splits}.csv",
        index=False)

    save_frame = pd.DataFrame(save_dict_pre_test)
    save_frame.to_csv(
        f"{args.output_root}/{args.m1}/interpretability/learning_curves/direction={args.direction}/{args.gran}/"
        f"precision_test_multi_{args.forecast_horizon}_fee_n_splits={args.n_splits}.csv",
        index=False)
