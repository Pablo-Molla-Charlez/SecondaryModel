"""Utility helpers for Optuna config generation with preserved formatting."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import optuna
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq


def to_builtin(obj: Any):
    """Recursively convert numpy / scalar types to builtin Python values."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    return obj


def build_candidate_config(base_cfg: Dict[str, Any], task: str, trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    """Create a config dictionary containing the trial-specific values."""
    task_lower = task.lower()
    model_key = f"model_{task_lower}"
    train_key = f"train_{task_lower}"

    candidate_cfg = copy.deepcopy(base_cfg)
    params = trial.params

    # CNN stack
    user_attrs = getattr(trial, "user_attrs", {}) or {}

    cnn_embed_dim = user_attrs.get("cnn_embed_dim_list")
    cnn_kernel = user_attrs.get("cnn_kernel_list")
    cnn_stride = user_attrs.get("cnn_stride_list")
    n_convs = user_attrs.get("n_convs")

    if cnn_embed_dim is None or cnn_kernel is None or cnn_stride is None:
        n_convs = int(params.get("n_convs", len(candidate_cfg.get(model_key, {}).get("cnn_embed_dim", []))))
        cnn_embed_dim = [params[f"cnn_embed_dim_{i}"] for i in range(n_convs)]
        cnn_kernel = [params[f"cnn_kernel_{i}"] for i in range(n_convs)]
        cnn_stride = [params[f"cnn_stride_{i}"] for i in range(n_convs)]

    model_section = candidate_cfg.get(model_key, {})
    model_section["cnn_blocks"] = len(cnn_embed_dim)
    model_section["cnn_embed_dim"] = cnn_embed_dim
    model_section["cnn_kernel"] = cnn_kernel
    model_section["cnn_stride"] = cnn_stride
    model_section["p_pos_drop"] = params["p_pos_drop"]

    transformer = model_section.get("transformer", {})
    transformer.update(
        {
            "heads": params["heads"],
            "layers": params["layers"],
            "ffn_dim": params["ffn_dim"],
            "dropout": params["dropout"],
            "activation": params["activation"],
        }
    )
    model_section["transformer"] = transformer

    classifier = model_section.get("classifier", {})
    classifier.update(
        {
            "mlp_hidden": params["mlp_hidden"],
            "mlp_dropout": params["mlp_dropout"],
            "mlp_activation": params["mlp_activation"],
            "mlp_pooling": params["mlp_pooling"],
        }
    )
    model_section["classifier"] = classifier
    candidate_cfg[model_key] = model_section

    train_section = copy.deepcopy(candidate_cfg.get(train_key, {}))
    train_section["optimizer"] = params["optimizer"]
    train_section["lr"] = params["lr"]
    train_section["weight_decay"] = params["weight_decay"]
    train_section["lr_decay"] = params["lr_decay"]
    train_section["batch_size"] = 2 ** int(params["batch_pow"])
    betas = train_section.get("betas", [0.9, 0.999])
    train_section["betas"] = [params.get("beta1", betas[0]), params.get("beta2", betas[1])]
    train_section["loss_function"] = params["loss_function"]

    focal_gamma = params.get("focal_gamma", train_section.get("focal_gamma"))
    if focal_gamma is not None:
        train_section["focal_gamma"] = focal_gamma
    train_section["focal_alpha"] = train_section.get("focal_alpha", 2.954 / (2.954 + 1.1514))

    scheduler_section = copy.deepcopy(train_section.get("scheduler", {}))
    scheduler_section["sch_name"] = params["sch_name"]
    scheduler_section["warmup_epochs"] = scheduler_section.get("warmup_epochs", 5)
    scheduler_section["plateau_patience"] = params["plateau_patience"]
    scheduler_section["plateau_factor"] = params["plateau_factor"]
    train_section["scheduler"] = scheduler_section
    candidate_cfg[train_key] = train_section

    training_mode = copy.deepcopy(candidate_cfg.get("training_mode", {}))
    training_mode["optuna_task"] = task.upper()
    training_mode["loss_function"] = params["loss_function"]

    if focal_gamma is not None:
        training_mode["focal_gamma"] = focal_gamma
    training_mode["focal_alpha"] = train_section.get("focal_alpha", training_mode.get("focal_alpha"))
    training_mode["num_classes"] = 1 if params["loss_function"] == "bce" else 2
    candidate_cfg["training_mode"] = training_mode

    return to_builtin(candidate_cfg)


def _update_structure(target, updates):
    """Recursively update ruamel structures preserving formatting."""
    if isinstance(target, CommentedMap) and isinstance(updates, dict):
        for key, value in updates.items():
            if key in target:
                current = target[key]
                if isinstance(current, CommentedMap) and isinstance(value, dict):
                    _update_structure(current, value)
                elif isinstance(current, CommentedSeq) and isinstance(value, list):
                    # Update sequence in-place to preserve formatting
                    for idx in range(min(len(current), len(value))):
                        current[idx] = value[idx]
                    if len(current) > len(value):
                        del current[len(value):]
                    elif len(value) > len(current):
                        current.extend(value[len(current):])
                else:
                    target[key] = value
            else:
                target[key] = value
    elif isinstance(target, CommentedSeq) and isinstance(updates, list):
        for idx in range(min(len(target), len(updates))):
            target[idx] = updates[idx]
        if len(target) > len(updates):
            del target[len(updates):]
        elif len(updates) > len(target):
            target.extend(updates[len(target):])


def export_pareto_configs(
    base_config_path: Path,
    base_cfg: Dict[str, Any],
    trials: Iterable[optuna.trial.FrozenTrial],
    run_root: Path,
    task: str,
    folder_name: str = "Optuna_Pareto_Candidates",
) -> int:
    """Write YAML configs for Pareto-optimal trials and return the count saved."""
    candidates_dir = Path(run_root) / folder_name
    candidates_dir.mkdir(parents=True, exist_ok=True)

    for prev_cfg in candidates_dir.glob("config_*.yaml"):
        prev_cfg.unlink()

    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    yaml_rt.indent(mapping=2, sequence=4, offset=2)

    base_text = Path(base_config_path).read_text()

    saved = 0
    for trial in trials:
        candidate = build_candidate_config(base_cfg, task, trial)
        doc = yaml_rt.load(base_text)

        _update_structure(doc["training_mode"], candidate["training_mode"])
        task_lower = task.lower()
        _update_structure(doc[f"model_{task_lower}"], candidate[f"model_{task_lower}"])
        _update_structure(doc[f"train_{task_lower}"], candidate[f"train_{task_lower}"])

        cfg_path = candidates_dir / f"config_{trial.number}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as handle:
            yaml_rt.dump(doc, handle)
        saved += 1

    return saved


__all__ = ["to_builtin", "build_candidate_config", "export_pareto_configs"]
