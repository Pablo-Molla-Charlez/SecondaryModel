import os
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(df, save_path, display=False, metric=None, gran=None, m1=None, n_splits=None,
        forecast_horizon=None, direction=None):
    # Ensure correct column order
    df = df.reindex(sorted(df.columns, key=float), axis=1)

    x = df.columns.astype(float).to_numpy()
    y_mean = df.mean(axis=0).to_numpy()
    y_std = df.std(axis=0).to_numpy()

    lower = np.clip(y_mean - y_std, 0, 1)
    upper = np.clip(y_mean + y_std, 0, 1)

    plt.figure(figsize=(6, 4))
    plt.plot(x, y_mean, marker='o', markerfacecolor='r', markeredgecolor='k', label='mean')
    plt.fill_between(x, lower, upper, alpha=0.3)

    plt.xlabel("Quota of training data used")
    plt.ylabel(f"{metric}")
    plt.ylim(0.1, 1)
    plt.xlim(0, np.max(x) + 0.5)
    plt.yscale("log")

    # Title
    title_parts = []
    if m1: title_parts.append(f"{m1}")
    if gran: title_parts.append(f"gran={gran}")
    if n_splits: title_parts.append(f"splits={n_splits}")
    if forecast_horizon: title_parts.append(f"h={forecast_horizon}")
    if direction: title_parts.append(f"dir={direction}")

    plt.title(" | ".join(title_parts))

    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if display:
        plt.show()


def plot_learning_curves_grid(
    dir_path,
    m1,
    direction,
    forecast_horizon,
    n_splits,
    gran_list,
    metric="Accuracy",
    save_path=None,
    display=False
):
    n_grans = len(gran_list)
    fig, axes = plt.subplots(2, n_grans, figsize=(4 * n_grans, 8), sharey=True, sharex='col')
    metric_to_load = "acc" if metric == "Accuracy" else "precision"
    for col_idx, gran in enumerate(gran_list):
        # --- VALIDATION DATA ---
        val_path = os.path.join(
            dir_path,
            f"{m1}/interpretability/learning_curves/direction={direction}/{gran}",
            f"{metric_to_load}_val_multi_{forecast_horizon}_fee_n_splits={n_splits}.csv"
        )
        try:
            df_val = pd.read_csv(val_path)
        except FileNotFoundError:
            continue
        df_val = df_val.reindex(sorted(df_val.columns, key=float), axis=1)
        print(f"{gran}: {df_val.columns}")

        x = df_val.columns.astype(float).to_numpy()
        y_mean = df_val.mean(axis=0).to_numpy()
        y_std = df_val.std(axis=0).to_numpy()
        lower = np.clip(y_mean - y_std, 0, 1)
        upper = np.clip(y_mean + y_std, 0, 1)

        ax = axes[0, col_idx] if n_grans > 1 else axes[0]
        ax.plot(x, y_mean, marker='o', markerfacecolor='r', markeredgecolor='k', label='mean')
        ax.fill_between(x, lower, upper, alpha=0.3)
        ax.set_title(f"{m1} | {gran} | val")
        ax.set_ylim(0.3, 1)
        ax.set_xlim(0, np.max(x) + 0.5)
        ax.set_yscale("log")
        ax.grid(axis='y', which='both', linestyle='--', alpha=0.6)

        # --- TEST DATA ---
        test_path = os.path.join(
            dir_path,
            f"{m1}/interpretability/learning_curves/direction={direction}/{gran}",
            f"{metric_to_load}_test_multi_{forecast_horizon}_fee_n_splits={n_splits}.csv"
        )
        try:
            df_test = pd.read_csv(test_path)
        except FileNotFoundError:
            continue
        df_test = df_test.reindex(sorted(df_test.columns, key=float), axis=1)

        x_test = df_test.columns.astype(float).to_numpy()
        y_mean_test = df_test.mean(axis=0).to_numpy()
        y_std_test = df_test.std(axis=0).to_numpy()
        lower_test = np.clip(y_mean_test - y_std_test, 0, 1)
        upper_test = np.clip(y_mean_test + y_std_test, 0, 1)

        ax = axes[1, col_idx] if n_grans > 1 else axes[1]
        ax.plot(x_test, y_mean_test, marker='o', markerfacecolor='b', markeredgecolor='k', label='mean')
        ax.fill_between(x_test, lower_test, upper_test, alpha=0.3)
        ax.set_title(f"{m1} | {gran} | test")
        ax.set_ylim(0.3, 1)
        ax.set_xlim(0, np.max(x) + 0.5)
        ax.set_yscale("log")
        ax.grid(axis='y', which='both', linestyle='--', alpha=0.6)

    axes[0, 0].set_ylabel(f"Validation {metric}")
    axes[1, 0].set_ylabel(f"Test {metric}")
    fig.supxlabel("Quota of training data used")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if display:
        plt.show()


if __name__ == "__main__":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    dir_path = "/home/till/PycharmProjects/Secondary-Model/src/Output"
    # /Kronos/interpretability/learning_curves/direction=up/1d/acc_test_multi_7_fee_n_splits=100.csv
    direction = "up"  # or up
    forecast_horizon = 7  # can only be 7 atm
    m1 = "Kronos"  # Kronos or Fincast
    n_splits = 100
    gran_list = ["15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

    # for gran in ["15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]:
    #     df = pd.read_csv(os.path.join(dir_path, f"{m1}/interpretability/learning_curves/direction={direction}",
    #                                   f"/{gran}/acc_val_multi_{forecast_horizon}_fee_n_splits={n_splits}.csv"))
    #
    #     # plot the val accuracy in the upper row as a subfigure
    #     plot_learning_curve(...)
    #
    #     df = pd.read_csv(os.path.join(dir_path, f"{m1}/interpretability/learning_curves/direction={direction}",
    #                                   f"/{gran}/acc_val_multi_{forecast_horizon}_fee_n_splits={n_splits}.csv"))
    #
    #     # plot the test accuracy in the lower row as a subfigure
    #     plot_learning_curve(...)

    # plot_learning_curves_grid(
    #     dir_path,
    #     m1,
    #     direction,
    #     forecast_horizon,
    #     n_splits,
    #     gran_list,
    #     metric="Accuracy",
    #     display=True,
    #     save_path=os.path.join(dir_path, f"{m1}/rf/{direction.upper()}/interpretability", "acc_learning_curves_overview.pdf")
    # )

    plot_learning_curves_grid(
        dir_path,
        m1,
        direction,
        forecast_horizon,
        n_splits,
        gran_list,
        metric="Precision",
        display=True,
        save_path=os.path.join(dir_path, f"{m1}/rf/{direction.upper()}/interpretability", "pre_learning_curves_overview.pdf")
    )