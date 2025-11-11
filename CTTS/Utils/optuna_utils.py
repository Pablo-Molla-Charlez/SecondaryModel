# ┏━━━━━━━━━━ Provide module-level summary for these helper utilities ━━━━━━━━━━┓
"""Utility helpers for Optuna config generation with preserved formatting."""
from __future__ import annotations

import copy
import optuna
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ┏━━━━━━━━━━ Declare default objective directions and metrics when none provided ━━━━━━━━━━┓
DEFAULT_OPTUNA_OBJECTIVES: List[Tuple[str, str]] = [("minimize", "mean_val_loss"),
                                                    ("maximize", "best_fbeta")]

def parse_optuna_objectives(raw: Optional[Any],
                            fallback: Optional[Sequence[Any]] = None) -> List[Tuple[str, str]]:
    """Normalize configurable Optuna objectives from YAML or CLI sources."""
    
    # ┏━━━━━━━━━━ When user provided explicit objectives, adopt them directly ━━━━━━━━━━┓
    if raw:
        objectives_source = raw

    # ┏━━━━━━━━━━ Prepare container for normalized direction/metric pairs ━━━━━━━━━━┓
    normalized: List[Tuple[str, str]] = []
    # ┏━━━━━━━━━━ Enumerate provided objectives to normalize each entry ━━━━━━━━━━┓
    for idx, entry in enumerate(objectives_source):
        direction_raw = None
        metric = None

        # ┏━━━━━━━━━━ Allow dictionary-based objectives with explicit keys ━━━━━━━━━━┓
        if isinstance(entry, dict):
            # ┏━━━━━━━━━━ Collect possible direction key aliases in priority order ━━━━━━━━━━┓
            direction_raw = (entry.get("direction")
                            or entry.get("dir")
                            or entry.get("objective"))
                            
            # ┏━━━━━━━━━━ Pull metric descriptor from multiple acceptable keys ━━━━━━━━━━┓
            metric = entry.get("metric") or entry.get("name") or entry.get("key")
        
        # ┏━━━━━━━━━━ Handle list/tuple style objectives ━━━━━━━━━━┓
        elif isinstance(entry, (list, tuple)):
            # ┏━━━━━━━━━━ Support single-element sequences that only contain a metric ━━━━━━━━━━┓
            if len(entry) == 1:
                metric = entry[0]
            # ┏━━━━━━━━━━ Handle two-or-more element sequences for direction + metric ━━━━━━━━━━┓
            elif len(entry) >= 2:
                # ┏━━━━━━━━━━ For two or more entries treat first as direction, second as metric ━━━━━━━━━━┓
                direction_raw, metric = entry[0], entry[1]
        
        # ┏━━━━━━━━━━ Handle bare string objectives ━━━━━━━━━━┓
        elif isinstance(entry, str):
            # ┏━━━━━━━━━━ Interpret bare strings as metric names ━━━━━━━━━━┓
            metric = entry
        # ┏━━━━━━━━━━ Reject everything else as unsupported configuration ━━━━━━━━━━┓
        else:
            raise ValueError(f"Unsupported optuna_objectives entry type: {type(entry)!r}")

        # ┏━━━━━━━━━━ Enforce that a metric name is always present ━━━━━━━━━━┓
        if metric is None:
            raise ValueError("Each optuna objective must specify a metric name.")

        # ┏━━━━━━━━━━ Default missing directions to minimize first metric, maximize subsequent ━━━━━━━━━━┓
        direction = direction_raw or ("minimize" if idx == 0 else "maximize")
        # ┏━━━━━━━━━━ Normalize direction text by lowercasing and stringifying ━━━━━━━━━━┓
        direction_str = str(direction).lower()
        if direction_str in {"min", "minimize"}:
            # ┏━━━━━━━━━━ Map all minimizing aliases to canonical string ━━━━━━━━━━┓
            direction_norm = "minimize"
        # ┏━━━━━━━━━━ Recognize maximizing aliases ━━━━━━━━━━┓
        elif direction_str in {"max", "maximize"}:
            # ┏━━━━━━━━━━ Map maximizing aliases likewise ━━━━━━━━━━┓
            direction_norm = "maximize"
        # ┏━━━━━━━━━━ Reject everything else as invalid ━━━━━━━━━━┓
        else:
            raise ValueError(f"Unknown Optuna objective direction '{direction}'. Use 'minimize' or 'maximize'.")

        # ┏━━━━━━━━━━ Accumulate the sanitized direction/metric pair ━━━━━━━━━━┓
        normalized.append((direction_norm, str(metric)))

    return normalized

def to_builtin(obj: Any):
    """Recursively convert numpy / scalar types to builtin Python values."""
    # ┏━━━━━━━━━━ Convert numpy scalar objects via .item() ━━━━━━━━━━┓
    if isinstance(obj, np.generic):
        return obj.item()
    # ┏━━━━━━━━━━ Convert dictionary values recursively ━━━━━━━━━━┓
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    # ┏━━━━━━━━━━ Convert list/tuple entries recursively ━━━━━━━━━━┓
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]

    return obj


# ┏━━━━━━━━━━ Define feature mapper that resolves per-task feature lists ━━━━━━━━━━┓
def feature_map(cfg: Dict[str, Any], prefix: str) -> Dict[str, list]:
    """Return per-task feature lists, falling back to shared definitions."""
    # ┏━━━━━━━━━━ Build expected UP & DN suffix key ━━━━━━━━━━┓
    up_key = f"{prefix}_up"
    dn_key = f"{prefix}_dn"

    # ┏━━━━━━━━━━ When both keys exist, return mapping for each task ━━━━━━━━━━┓
    if up_key in cfg and dn_key in cfg:
        # ┏━━━━━━━━━━ Return explicit mapping for UP and DN tasks ━━━━━━━━━━┓
        return {"UP": cfg[up_key], 
                "DN": cfg[dn_key]}

    # ┏━━━━━━━━━━ Otherwise raise helpful error referencing expected names ━━━━━━━━━━┓
    raise KeyError(f"Configuration must define '{prefix}_up'/'{prefix}_dn' or shared '{prefix}'.")

def build_candidate_config(base_cfg: Dict[str, Any], 
                           task: str, 
                           trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:

    """Create a config dictionary containing the trial-specific values."""
    # ┏━━━━━━━━━━ Normalize task name for key derivation ━━━━━━━━━━┓
    task_lower = task.lower()
    model_key = f"model_{task_lower}"
    train_key = f"train_{task_lower}"

    # ┏━━━━━━━━━━ Clone the base config so modifications stay isolated ━━━━━━━━━━┓
    candidate_cfg = copy.deepcopy(base_cfg)

    # ┏━━━━━━━━━━ Short-hand trial parameters dictionary for readability ━━━━━━━━━━┓
    params = trial.params

    # ┏━━━━━━━━━━ Capture any auxiliary architecture hints stored in user attrs ━━━━━━━━━━┓
    user_attrs = getattr(trial, "user_attrs", {}) or {}

    # ┏━━━━━━━━━━ Pull CNN embedding dims, kernel sizes, CNN strides and explicit convolution count ━━━━━━━━━━┓
    cnn_embed_dim = user_attrs.get("cnn_embed_dim_list")
    cnn_kernel = user_attrs.get("cnn_kernel_list")
    cnn_stride = user_attrs.get("cnn_stride_list")
    n_convs = user_attrs.get("n_convs")

    # ┏━━━━━━━━━━ Fall back to trial params for CNN definitions when attrs missing ━━━━━━━━━━┓
    if cnn_embed_dim is None or cnn_kernel is None or cnn_stride is None:
        n_convs = int(params.get("n_convs", len(candidate_cfg.get(model_key, {}).get("cnn_embed_dim", []))))
        cnn_embed_dim = [params[f"cnn_embed_dim_{i}"] for i in range(n_convs)]
        cnn_kernel = [params[f"cnn_kernel_{i}"] for i in range(n_convs)]
        cnn_stride = [params[f"cnn_stride_{i}"] for i in range(n_convs)]

    # ┏━━━━━━━━━━ Grab model section for mutation ━━━━━━━━━━┓
    model_section = candidate_cfg.get(model_key, {})

    # ┏━━━━━━━━━━ Set number of CNN blocks observed, per-layer embed dims, per-layer kernel sizes, per-layer stride sizes ━━━━━━━━━━┓
    model_section["cnn_blocks"] = len(cnn_embed_dim)
    model_section["cnn_embed_dim"] = cnn_embed_dim
    model_section["cnn_kernel"] = cnn_kernel
    model_section["cnn_stride"] = cnn_stride

    # ┏━━━━━━━━━━ Persist positional dropout probability ━━━━━━━━━━┓
    model_section["p_pos_drop"] = params["p_pos_drop"]

    # ┏━━━━━━━━━━ Retrieve nested transformer structure for updates ━━━━━━━━━━┓
    transformer = model_section.get("transformer", {})
    
    # ┏━━━━━━━━━━ Apply transformer hyper-parameters from trial values ━━━━━━━━━━┓
    transformer.update({"heads": params["heads"],
                        "layers": params["layers"],
                        "ffn_dim": params["ffn_dim"],
                        "dropout": params["dropout"],
                        "activation": params["activation"]})

    # ┏━━━━━━━━━━ Persist transformer block back into model section ━━━━━━━━━━┓
    model_section["transformer"] = transformer

    # ┏━━━━━━━━━━ Retrieve classifier head configuration ━━━━━━━━━━┓
    classifier = model_section.get("classifier", {})

    # ┏━━━━━━━━━━ Update classifier hyper-parameters using trial values ━━━━━━━━━━┓
    classifier.update(
        {"mlp_hidden": params["mlp_hidden"],
         "mlp_dropout": params["mlp_dropout"],
         "mlp_activation": params["mlp_activation"],
         "mlp_pooling": params["mlp_pooling"]})

    # ┏━━━━━━━━━━ Persist classifier block back into model section ━━━━━━━━━━┓
    model_section["classifier"] = classifier
    # ┏━━━━━━━━━━ Write fully-updated model section into candidate config ━━━━━━━━━━┓
    candidate_cfg[model_key] = model_section

    # ┏━━━━━━━━━━ Duplicate training section so edits are isolated ━━━━━━━━━━┓
    train_section = copy.deepcopy(candidate_cfg.get(train_key, {}))
    
    # ┏━━━━━━━━━━ Store Hyper-Parameters ━━━━━━━━━━┓
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

    # ┏━━━━━━━━━━ Copy scheduler subsection for isolated mutation ━━━━━━━━━━┓
    scheduler_section = copy.deepcopy(train_section.get("scheduler", {}))
    scheduler_section["sch_name"] = params["sch_name"]
    scheduler_section["warmup_epochs"] = scheduler_section.get("warmup_epochs", 5)
    scheduler_section["plateau_patience"] = params["plateau_patience"]
    scheduler_section["plateau_factor"] = params["plateau_factor"]
    train_section["scheduler"] = scheduler_section
    candidate_cfg[train_key] = train_section

    # ┏━━━━━━━━━━ Copy overarching training_mode block for updates ━━━━━━━━━━┓
    training_mode = copy.deepcopy(candidate_cfg.get("training_mode", {}))
    
    # ┏━━━━━━━━━━ Record which task Optuna optimized for this candidate ━━━━━━━━━━┓
    training_mode["optuna_task"] = task.upper()
    # ┏━━━━━━━━━━ Keep training mode loss function consistent ━━━━━━━━━━┓
    training_mode["loss_function"] = params["loss_function"]

    # ┏━━━━━━━━━━ Mirror resolved focal gamma into training_mode when present ━━━━━━━━━━┓
    if focal_gamma is not None:
        training_mode["focal_gamma"] = focal_gamma
    
    # ┏━━━━━━━━━━ Mirror focal alpha from train section back into training_mode ━━━━━━━━━━┓
    training_mode["focal_alpha"] = train_section.get("focal_alpha", training_mode.get("focal_alpha"))
    
    # ┏━━━━━━━━━━ Derive class count from chosen loss ━━━━━━━━━━┓
    training_mode["num_classes"] = 1 if params["loss_function"] == "bce" else 2
    
    # ┏━━━━━━━━━━ Persist training_mode block back in candidate config ━━━━━━━━━━┓
    candidate_cfg["training_mode"] = training_mode

    return to_builtin(candidate_cfg)

# ┏━━━━━━━━━━ Define ruamel-aware deep update helper ━━━━━━━━━━┓
def _update_structure(target, updates):
    """Recursively update ruamel structures preserving formatting."""
    # ┏━━━━━━━━━━ Handle mapping-to-mapping updates ━━━━━━━━━━┓
    if isinstance(target, CommentedMap) and isinstance(updates, dict):
        # ┏━━━━━━━━━━ Iterate over provided update items ━━━━━━━━━━┓
        for key, value in updates.items():
            if key in target:
                # ┏━━━━━━━━━━ Fetch existing node for nested merging ━━━━━━━━━━┓
                current = target[key]
                if isinstance(current, CommentedMap) and isinstance(value, dict):
                    # ┏━━━━━━━━━━ Recurse into nested mapping ━━━━━━━━━━┓
                    _update_structure(current, value)
                
                elif isinstance(current, CommentedSeq) and isinstance(value, list):
                    # ┏━━━━━━━━━━ Update sequence in-place to preserve formatting ━━━━━━━━━━┓
                    for idx in range(min(len(current), len(value))):
                        current[idx] = value[idx]
                    
                    # ┏━━━━━━━━━━ When current list longer, trim trailing entries ━━━━━━━━━━┓
                    if len(current) > len(value):
                        del current[len(value):]
                    
                    # ┏━━━━━━━━━━ When incoming list longer, append remaining entries ━━━━━━━━━━┓
                    elif len(value) > len(current):
                        current.extend(value[len(current):])
                else:
                    # ┏━━━━━━━━━━ Overwrite scalar entries directly ━━━━━━━━━━┓
                    target[key] = value
            else:
                # ┏━━━━━━━━━━ Add brand new key preserving insertion order ━━━━━━━━━━┓
                target[key] = value

    elif isinstance(target, CommentedSeq) and isinstance(updates, list):
        # ┏━━━━━━━━━━ Update list contents element-wise ━━━━━━━━━━┓
        for idx in range(min(len(target), len(updates))):
            target[idx] = updates[idx]
        
        # ┏━━━━━━━━━━ If target longer, remove extra tail items ━━━━━━━━━━┓
        if len(target) > len(updates):
            del target[len(updates):]
        
        # ┏━━━━━━━━━━ If updates longer, extend target with remainder ━━━━━━━━━━┓
        elif len(updates) > len(target):
            target.extend(updates[len(target):])

def _sync_usual_with_optuna(training_mode: CommentedMap):
    # ┏━━━━━━━━━━ Clarify synchronization purpose for training_mode fields ━━━━━━━━━━┓
    """Mirror Optuna selections into the usual training fields."""
    # ┏━━━━━━━━━━ Guard clause to ensure we can mutate the mapping ━━━━━━━━━━┓
    if not isinstance(training_mode, CommentedMap):
        return

    # ┏━━━━━━━━━━ Map each Optuna-only field to its usual counterpart ━━━━━━━━━━┓
    mappings = [("meta_label_optuna",  "meta_label_usual"),
                ("optuna_task",        "normal_task"),
                ("granularity_optuna", "granularity_usual")]

    # ┏━━━━━━━━━━ Copy values from Optuna keys into standard keys ━━━━━━━━━━┓
    for optuna_key, usual_key in mappings:
        if optuna_key in training_mode:
            training_mode[usual_key] = training_mode[optuna_key]

# ┏━━━━━━━━━━ Define exporter that persists Pareto candidate configs ━━━━━━━━━━┓
def export_pareto_configs(base_config_path: Path,
                          base_cfg: Dict[str, Any],
                          trials: Iterable[optuna.trial.FrozenTrial],
                          run_root: Path,
                          task: str,
                          folder_name: str = "Optuna_Pareto_Candidates") -> int:

    """Write YAML configs for Pareto-optimal trials and return the count saved."""
    # ┏━━━━━━━━━━ Resolve folder path where configs will be stored ━━━━━━━━━━┓
    candidates_dir = Path(run_root) / folder_name
    
    # ┏━━━━━━━━━━ Create folder tree if needed ━━━━━━━━━━┓
    candidates_dir.mkdir(parents = True, exist_ok = True)

    # ┏━━━━━━━━━━ Remove previous configs to avoid stale mixes ━━━━━━━━━━┓
    for prev_cfg in candidates_dir.glob("config_*.yaml"):
        prev_cfg.unlink()

    # ┏━━━━━━━━━━ Instantiate YAML runtime with formatting preservation ━━━━━━━━━━┓
    yaml_rt = YAML()

    # ┏━━━━━━━━━━ Preserve quotes exactly as in source YAML ━━━━━━━━━━┓
    yaml_rt.preserve_quotes = True
    
    # ┏━━━━━━━━━━ Configure indentation to match base config ━━━━━━━━━━┓
    yaml_rt.indent(mapping=2, sequence=4, offset=2)

    # ┏━━━━━━━━━━ Load base config file contents for templating ━━━━━━━━━━┓
    base_text = Path(base_config_path).read_text()
    saved = 0
    # ┏━━━━━━━━━━ Iterate over each Pareto-tracked trial ━━━━━━━━━━┓
    for trial in trials:
        # ┏━━━━━━━━━━ Build trial-specific candidate configuration ━━━━━━━━━━┓
        candidate = build_candidate_config(base_cfg, task, trial)
        
        # ┏━━━━━━━━━━ Reload pristine YAML document for patching ━━━━━━━━━━┓
        doc = yaml_rt.load(base_text)

        # ┏━━━━━━━━━━ Merge training_mode settings from candidate ━━━━━━━━━━┓
        _update_structure(doc["training_mode"], candidate["training_mode"])
        
        # ┏━━━━━━━━━━ Mirror Optuna choices into usual training slots ━━━━━━━━━━┓
        _sync_usual_with_optuna(doc.get("training_mode"))
        
        # ┏━━━━━━━━━━ Lowercase task for key suffix resolution ━━━━━━━━━━┓
        task_lower = task.lower()
        
        # ┏━━━━━━━━━━ Merge model section overrides ━━━━━━━━━━┓
        _update_structure(doc[f"model_{task_lower}"], candidate[f"model_{task_lower}"])
        
        # ┏━━━━━━━━━━ Merge train section overrides ━━━━━━━━━━┓
        _update_structure(doc[f"train_{task_lower}"], candidate[f"train_{task_lower}"])

        # ┏━━━━━━━━━━ Compute destination file path for this trial ━━━━━━━━━━┓
        cfg_path = candidates_dir / f"config_{trial.number}.yaml"
        
        # ┏━━━━━━━━━━ Write updated YAML document to disk ━━━━━━━━━━┓
        with open(cfg_path, "w", encoding="utf-8") as handle:
            yaml_rt.dump(doc, handle)

        saved += 1
    return saved

# ┏━━━━━━━━━━ Enumerate public symbols exposed by this module ━━━━━━━━━━┓
__all__ = ["to_builtin",
          "feature_map",
          "parse_optuna_objectives",
          "build_candidate_config",
          "export_pareto_configs"]
