
#!/usr/bin/env python
"""Run an ablation study over context features using the train.py pipeline."""
from __future__ import annotations

import argparse
import copy
import yaml

from pathlib import Path
from train import run_training
from ablation_utils import (
    resolve_baseline_features,
    ensure_clean_dir,
    reorganize_outputs,
    slugify,
    feature_groups,
    write_metadata,
)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ BASELINE DEFINITIONS                                                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
BASELINE_FEATURES_BY_TASK = {
    "UP": ["m1_prediction", "m1_up", "m1_pred_proba_up"],
    "DN": ["m1_prediction", "m1_dn", "m1_pred_proba_dn"],
}

GROUP_LABELS = {
    0: "Baseline",
    1: "Singles",
    2: "Pairs",
    3: "Trios",
    4: "Quartets",
    5: "Quintets",
    6: "Sextets",
}


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ MAIN ENTRYPOINT                                                       ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
def main() -> None:
    parser = argparse.ArgumentParser(description="Run context-feature ablation study")
    parser.add_argument(
        "--config",
        type=str,
        default="config_431.yaml",
        help="Base configuration file to use as template",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find config file at {cfg_path}")

    base_cfg = yaml.safe_load(cfg_path.open())
    training_mode = base_cfg.get("training_mode", {})
    task = training_mode.get(
        "ablation_task",
        training_mode.get("normal_task", training_mode.get("optuna_task", "")),
    ).upper()
    if not task:
        raise ValueError("ablation_task (or a fallback) must be defined in training_mode")

    context_key = f"context_features_{task.lower()}"
    context_features = base_cfg.get(context_key) or base_cfg.get("context_features", [])
    baseline_features = resolve_baseline_features(task, context_features, BASELINE_FEATURES_BY_TASK)

    additional_features = [feat for feat in context_features if feat not in baseline_features]

    output_root = Path(base_cfg["paths"]["output_root"])
    if not output_root.is_absolute():
        output_root = (Path(__file__).parent / output_root).resolve()

    ablation_root = output_root / "Ablation"
    ablation_root.mkdir(parents=True, exist_ok=True)
    symbol = base_cfg["dataset"]["symbol"]

    for size, combo in feature_groups(additional_features):
        group_name = GROUP_LABELS.get(size, f"{size}_Combos")
        combo_features = list(baseline_features) + list(combo)
        combo_slug = slugify(combo)
        run_dir = ablation_root / group_name / combo_slug

        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Group: {group_name} | Features: {combo_features}")
        print(f"Output directory: {run_dir}")

        ensure_clean_dir(run_dir)

        cfg_variant = copy.deepcopy(base_cfg)
        cfg_variant[context_key] = combo_features
        if "context_features" in cfg_variant:
            cfg_variant["context_features"] = combo_features
        cfg_variant.setdefault("training_mode", {})
        cfg_variant["training_mode"]["normal_task"] = task
        cfg_variant["paths"]["output_root"] = str(run_dir)

        run_training(cfg_variant)
        reorganize_outputs(run_dir, symbol)

        column_key = f"column_features_{task.lower()}"
        column_features = cfg_variant.get(column_key) or cfg_variant.get("column_features", [])
        write_metadata(
            run_dir / "metadata.json",
            task=task,
            column_features=column_features,
            baseline=baseline_features,
            extras=combo,
            context=combo_features,
        )


if __name__ == "__main__":
    main()
