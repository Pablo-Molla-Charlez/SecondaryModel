import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from Utils.data_loaders.tabular_data_loader import load_tabular_dataset_from_cache_to_DataFrame
from Utils.edge.edge import run_cpcv_analysis

from Utils.classifier.autogluon_classifier import AutogluonClassifier
from Utils.classifier.tabpfn_classifier import TabPFN
from Utils.classifier.tabicl_classifier import TabICL
from Utils.classifier.random_forest_classifier import RFClassifier

from Utils.ts_cross_validation.combinatorial_purged_cv import CombinatorialPurgedCV
from Utils.ts_cross_validation.sklearn_ts_cv import SklearnTimeSeriesCV
from Utils.ts_cross_validation.purged_embargo_cv import PurgedEmbargoTimeSeriesCV

from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.base import clone
from sklearn.metrics import confusion_matrix


def fit_classifier(x_analysis, y_analysis, x_test, y_test, classifier, i_train, i_val):
    # clone model
    fitted_classifier = clone(classifier).fit(x_analysis.iloc[i_train], y_analysis[i_train])
    
    y_pred = fitted_classifier.predict(x_analysis.iloc[i_val])
    y_pred_test = fitted_classifier.predict(x_test)
    
    # Validation confusion matrix values
    val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_analysis[i_val], y_pred).ravel()
    
    # Test confusion matrix values
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, y_pred_test).ravel()
    
    return int(val_tp), int(val_fp), int(val_tn), int(val_fn), int(test_tp), int(test_fp), int(test_tn), int(test_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=str, default="/Volumes/Data/other/2026_NII/Output")
    parser.add_argument('--direction', type=str, default="up", choices=["down", "up"])
    parser.add_argument('--m1', type=str, default="Kronos", choices=["Kronos", "Fincast", "Tirex", "Chronos2"])
    parser.add_argument('--m2', type=str, default="randforest", choices=["randforest", "AutoGluon", "TabPFN", "TabICL"])
    parser.add_argument('--gran', type=str, default="1d",
                        choices=["1d", "1h", "2h", "4h", "6h", "8h", "12h", "30m"])  # ,"15m", "unified"
    parser.add_argument('--meta_label_mode', type=str, default="tp", choices=["tp", "fp", "og"])
    
    parser.add_argument('--forecast_horizon', type=int, default=7)
    
    # Specific to feature selection
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--n_test_splits', type=int, default=3)
    parser.add_argument('--cv_strategy', type=str, default="cpcv", choices=["cpcv", "tscv", "pecv"])
    
    args = parser.parse_args()
    
    # select model
    if args.m2 == "randforest":
        clf = RFClassifier()
    elif args.m2 == "AutoGluon":
        clf = AutogluonClassifier(args=args)
    elif args.m2 == "TabPFN":
        clf = TabPFN()
    elif args.m2 == "TabICL":
        clf = TabICL()
    else:
        raise ValueError(f"Unknown model {args.m2}")
    
    # select cross validation strategy
    if args.cv_strategy == "cpcv":
        cv = CombinatorialPurgedCV(n_splits=args.n_splits, n_test_splits=args.n_test_splits, embargo_pct=0.05,
                                   random_state=42)
    elif args.cv_strategy == "tscv":
        cv = SklearnTimeSeriesCV(n_splits=args.n_splits, random_state=42)
    elif args.cv_strategy == "pecv":
        cv = PurgedEmbargoTimeSeriesCV(n_splits=args.n_splits, embargo_pct=0.05, random_state=42)
    else:
        raise ValueError(f"Unknown CV strategy {args.cv_strategy}")
    
    # check whether results already exist and or create path
    save_path = (f"{args.output_root}/{args.m1}/{args.m2}/{args.direction.upper()}/classification_results/{args.gran}/"
                 f"results_meta_label_mode={args.meta_label_mode}_forecast_horizon={args.forecast_horizon}_cv={cv.name}_n_splits={cv.get_n_splits()}.csv")
    if os.path.exists(save_path):
        print(f"Results already exist at {save_path}. Skipping computation.")
        exit(0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # load dataset
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
    
    cache_path = f'{args.output_root}/{args.m1}/cache/multi_{args.forecast_horizon}_fee_{args.direction}_{hash_val}'
    
    # X_analysis, y_analysis, X_test, y_test = load_tabular_dataset_from_cache_to_DataFrame(cache_path=cache_path,
    #                                                                                       gran=args.gran)
    
    cache_path = Path(cache_path + f".pt")
    root = Path(args.output_root).parent
    print(f"root: {root}")
    cfg = {'paths': {'csv_dir': f'{root}/Data_MLA/{args.m1}/Crypto/TP/horizon_7', 'output_root': f'{args.output_root}'},
           'data': {'load': {'symbol': None, 'm1': f'{args.m1}', 'target_col': 'meta_label', 'meta_label_mode': 'tp',
                             'direction': f'{args.direction}', 'granularity': f'{args.gran}', 'forecast_horizon': 7},
                    'split': {'start_date': '2024-07-01', 'train_end': '2025-05-30', 'val_end': '2025-10-01',
                              'end_date': '2026-01-25'},
                    'features': {'input': ['open', 'high', 'low', 'close', 'volume'], 'engineered_features': {
                        'selected': ['bb_pctb_last', 'rsi_last', 'roc_5_last', 'roc_20_last', 'atr_norm_last']},
                                 'feature_selection': {'enabled': False, 'methods': ['mda', 'shap', 'lime'],
                                                       'top_k': None}}}, 'evaluation': {'fee_per_trade': 0.002}}
    exit(11)
    run_cpcv_analysis(cache_path, cfg, n_blocks=args.n_splits, k_test=args.n_test_splits, output_root=Path(args.output_root),
                      model_name=args.m2)
    # prepare dicts for saving results
    # save_dict = {"val_tp": [], "val_fp": [], "val_tn": [], "val_fn": [],
    #              "test_tp": [], "test_fp": [], "test_tn": [], "test_fn": []}
    # if args.m2 == "RF":  # NOTE we can parallelize RF as it doesnt need a GPU ;)
    #     with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    #         # now you can fit
    #         futures = [
    #             executor.submit(fit_classifier, X_analysis, y_analysis, X_test, y_test, clf, i_train_split, i_val_split)
    #             for
    #             i_train_split, i_val_split in cv.split(X_analysis, y_analysis)]
    #
    #         # await results
    #         as_completed([f for f in futures])
    #
    #     for i, future in enumerate(futures):
    #         # order: accuracy, precision
    #         val_tp, val_fp, val_tn, val_fn, test_tp, test_fp, test_tn, test_fn = future.result()
    #         save_dict["val_tp"].append(val_tp)
    #         save_dict["val_fp"].append(val_fp)
    #         save_dict["val_tn"].append(val_tn)
    #         save_dict["val_fn"].append(val_fn)
    #         save_dict["test_tp"].append(test_tp)
    #         save_dict["test_fp"].append(test_fp)
    #         save_dict["test_tn"].append(test_tn)
    #         save_dict["test_fn"].append(test_fn)
    # else:
    #     for i_train_split, i_val_split in cv.split(X_analysis, y_analysis):
    #         val_tp, val_fp, val_tn, val_fn, test_tp, test_fp, test_tn, test_fn = fit_classifier(X_analysis, y_analysis, X_test, y_test, clf, i_train_split, i_val_split)
    #         save_dict["val_tp"].append(val_tp)
    #         save_dict["val_fp"].append(val_fp)
    #         save_dict["val_tn"].append(val_tn)
    #         save_dict["val_fn"].append(val_fn)
    #         save_dict["test_tp"].append(test_tp)
    #         save_dict["test_fp"].append(test_fp)
    #         save_dict["test_tn"].append(test_tn)
    #         save_dict["test_fn"].append(test_fn)
    #
    # # per-fold accuracy and precision
    # val_tp_arr = np.array(save_dict["val_tp"])
    # val_fp_arr = np.array(save_dict["val_fp"])
    # val_tn_arr = np.array(save_dict["val_tn"])
    # val_fn_arr = np.array(save_dict["val_fn"])
    # test_tp_arr = np.array(save_dict["test_tp"])
    # test_fp_arr = np.array(save_dict["test_fp"])
    # test_tn_arr = np.array(save_dict["test_tn"])
    # test_fn_arr = np.array(save_dict["test_fn"])
    #
    # val_accuracy_per_fold = (val_tp_arr + val_tn_arr) / (val_tp_arr + val_fp_arr + val_tn_arr + val_fn_arr)
    # val_precision_per_fold = val_tp_arr / (val_tp_arr + val_fp_arr)
    # test_accuracy_per_fold = (test_tp_arr + test_tn_arr) / (test_tp_arr + test_fp_arr + test_tn_arr + test_fn_arr)
    # test_precision_per_fold = test_tp_arr / (test_tp_arr + test_fp_arr)
    #
    # save_dict["mean_val_accuracy"] = np.mean(val_accuracy_per_fold)
    # save_dict["mean_val_precision"] = np.mean(val_precision_per_fold)
    # save_dict["mean_test_accuracy"] = np.mean(test_accuracy_per_fold)
    # save_dict["mean_test_precision"] = np.mean(test_precision_per_fold)
    # save_dict["std_val_accuracy"] = np.std(val_accuracy_per_fold)
    # save_dict["std_val_precision"] = np.std(val_precision_per_fold)
    # save_dict["std_test_accuracy"] = np.std(test_accuracy_per_fold)
    # save_dict["std_test_precision"] = np.std(test_precision_per_fold)
    #
    # # store fold-level values as lists, summary stats as scalars
    # df = pd.DataFrame({
    #     "val_tp": [save_dict["val_tp"]],
    #     "val_fp": [save_dict["val_fp"]],
    #     "val_tn": [save_dict["val_tn"]],
    #     "val_fn": [save_dict["val_fn"]],
    #     "test_tp": [save_dict["test_tp"]],
    #     "test_fp": [save_dict["test_fp"]],
    #     "test_tn": [save_dict["test_tn"]],
    #     "test_fn": [save_dict["test_fn"]],
    #     "mean_val_accuracy": [save_dict["mean_val_accuracy"]],
    #     "mean_val_precision": [save_dict["mean_val_precision"]],
    #     "mean_test_accuracy": [save_dict["mean_test_accuracy"]],
    #     "mean_test_precision": [save_dict["mean_test_precision"]],
    #     "std_val_accuracy": [save_dict["std_val_accuracy"]],
    #     "std_val_precision": [save_dict["std_val_precision"]],
    #     "std_test_accuracy": [save_dict["std_test_accuracy"]],
    #     "std_test_precision": [save_dict["std_test_precision"]],
    # })
    #
    # df.to_csv(save_path, index=False)
