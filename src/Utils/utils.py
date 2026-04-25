"""
Utility classes and functions for Bi-FAST training.
"""
import json
import torch
import yaml
import hashlib
import numpy as np
from pathlib import Path
from typing import Any
from Utils.data import (load_dataset_from_config,
                        prepare_multi_gran_dataset,
                        prepare_multi_asset_dataset,
                        GRAN_SEQ_LEN,
                        GRAN_TO_ID)

# ┏━━━━━━━━━━ Model Label ━━━━━━━━━━┓
def model_label(model_name: str) -> str:
    """Canonical short label for a model name."""
    return {"rf": "RF", "xgboost": "XGB", "autogluon": "AG", "tabicl": "TabICL"}.get(model_name, model_name.upper())

# ┏━━━━━━━━━━ M1 Model Name retrieval ━━━━━━━━━━┓
def m1_model_name(cfg: dict | None) -> str:
    """Configured M1 model name normalized to lowercase."""
    raw = (cfg or {}).get("data", {}).get("load", {}).get("m1", "kronos")
    return str(raw).strip().lower() or "kronos"

# ┏━━━━━━━━━━ M1 Output Bucket ━━━━━━━━━━┓
def m1_output_bucket(cfg: dict | None) -> str:
    """Folder name under Output/ for the configured M1 model."""
    # TODO missing m1 models
    name = m1_model_name(cfg)
    if name == "kronos":
        return "Kronos"
    if name == "fincast":
        return "Fincast"
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_") if part)

# ┏━━━━━━━━━━ M1 Display Label ━━━━━━━━━━┓
def m1_display_label(cfg: dict | None) -> str:
    """Human-readable M1 label used in plots and reports."""
    return f"M1 {m1_output_bucket(cfg)}"


# ┏━━━━━━━━━━ HPO best-params loader ━━━━━━━━━━┓
# Models whose hyperparameters are currently tunable via Utils.hpo.
HPO_SUPPORTED_M2 = {"rf", "tabpfn", "tabicl"}

def _load_best_params(cfg: dict,
                      m2: str,
                      direction: str,
                      granularity: str) -> dict | None:
    """Return best params dict from HPO, or None if absent.

    Reads ``Output/{M1}/HPO/{m2}/{DIR}/{gran}/best_params.json`` produced by
    ``Utils.hpo``. Only rf/tabpfn/tabicl are tunable; other m2 return None.
    """
    # ┏━━━━━━━━━━ Determine M2 output bucket ━━━━━━━━━━┓
    if m2 not in HPO_SUPPORTED_M2:
        return None
    
    # ┏━━━━━━━━━━ Build best-params path ━━━━━━━━━━┓
    best_path = (Path(cfg["paths"]["output_root"])
                 / m1_output_bucket(cfg) / "HPO"
                 / m2 / direction.upper() / granularity / "best_params.json")
    
    # ┏━━━━━━━━━━ Load best-params file ━━━━━━━━━━┓
    if not best_path.exists():
        return None
    with open(best_path) as f:
        payload = json.load(f)

    return payload.get("best_params")


def _load_ag_best_hyperparameters(cfg: dict,
                                   direction: str,
                                   granularity: str) -> dict | None:
    """Return AutoGluon's best-model hyperparameters saved during Phase 1 training.

    Reads ``Output/{M1}/autogluon/{DIR}/Utility_Score/{gran}_{meta_label_mode}/final_model/ag_best_hyperparameters.json``
    written by ``AutoGluon.save_best_hyperparameters()`` inside ``_save_final_model``.

    Returns a dict with keys ``best_model_name``, ``model_type``, ``hyperparameters``,
    ``eval_metric``, and ``feature_names``, or None if the file does not exist.
    """
    meta_mode = cfg.get("data", {}).get("load", {}).get("meta_label_mode", "tp")
    thres_mode = cfg.get("runtime", {}).get("training", {}).get("thres", "utility")
    thres_folder = "Utility_Score_NoCal" if thres_mode == "utility_nocal" else "Utility_Score"
    final_model_dir = (Path(cfg["paths"]["output_root"])
                       / m1_output_bucket(cfg)
                       / "autogluon"
                       / direction.upper()
                       / thres_folder
                       / f"{granularity}_{meta_mode}"
                       / "final_model")
    hp_path = final_model_dir / "ag_best_hyperparameters.json"
    if not hp_path.exists():
        return None
    with open(hp_path) as f:
        return json.load(f)


# ┏━━━━━━━━━━ JSON Encoder ━━━━━━━━━━┓
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles common numpy scalar and array types."""

    def default(self, obj: Any):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ┏━━━━━━━━━━ JSON Helper ━━━━━━━━━━┓
def _safe_json(v):
    """Convert a value to a JSON-serializable type."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_safe_json(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe_json(val) for k, val in v.items()}
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


# ┏━━━━━━━━━━ Reproducibility ━━━━━━━━━━┓
def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ┏━━━━━━━━━━ Device ━━━━━━━━━━┓
def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ┏━━━━━━━━━━ Config & Cache Helpers ━━━━━━━━━━┓
def _expand_env_vars(obj):
    """Recursively expand $VAR / ${VAR} in all string values of a config dict."""
    import os
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def _load_config(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = _expand_env_vars(cfg)
        
    # ┏━━━━━━━━━━ Validation: Ensure m1 matches csv_dir ━━━━━━━━━━┓
    m1_val = str(cfg.get("data", {}).get("load", {}).get("m1", "kronos")).strip().lower()
    csv_dir_raw = cfg.get("paths", {}).get("csv_dir", "")
    csv_dir = str(csv_dir_raw).lower()
    
    if m1_val not in csv_dir:
        # Special case: allow 'all' or if the path actually contains the model name
        if not ("data_mla" in csv_dir and m1_val in csv_dir):
            raise ValueError(f"\n\n[CONFIG ERROR] Mismatch between 'm1' model and 'csv_dir' in {cfg_path}!\n"
                             f"  m1 is set to:      '{m1_val}'\n"
                             f"  csv_dir points to: '{csv_dir_raw}'\n"
                             f"Please ensure the 'm1' value matches the dataset in 'csv_dir' to avoid mixing data.\n")
        
    return cfg

# ┏━━━━━━━━━━ Build cache from config ━━━━━━━━━━┓
def _build_cache_from_config(config: dict, direction: str) -> tuple[Path, object]:
    """Build dataset cache from config.yaml, mirroring kronos_clas.py cache logic.
    
    Inputs:
    config: full config dict (the one and only)
    direction: "up" or "down" (this needs to be passed explicitly because we allow iterating over both)
    
    Returns (cache_path, dataset).  If the cache already exists on disk it is
    loaded; otherwise the dataset is compiled from CSVs and saved.
    """
    # ┏━━━━━━━━━━ Extract Configs ━━━━━━━━━━┓
    forecast_horizon = config['data']['load']["forecast_horizon"]
    
    # ┏━━━━━━━━━━ Deterministic hash ━━━━━━━━━━┓
    data_signature = {"granularity":      "all",
                      "meta_label_mode":  config["data"]["load"]["meta_label_mode"],
                      "direction":        direction,
                      "start_date":       config['data']['split']["start_date"],
                      "end_date":         config['data']['split']["end_date"],
                      "forecast_horizon": forecast_horizon,
                      "features":         config["data"]["features"],
                      "data_root":        str(Path(config["paths"]["csv_dir"]).parent)}

    # ┏━━━━━━━━━━ Dump hash ━━━━━━━━━━┓
    cfg_str  = json.dumps(data_signature, sort_keys=True)
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:10]

    # ┏━━━━━━━━━━ Build cache path ━━━━━━━━━━┓
    cache_dir = Path(config["paths"]["output_root"]) / config["data"]["load"]["m1"].capitalize() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"multi_{config['data']['load']['m1']}_{forecast_horizon}_fee_{direction}_{cfg_hash}.pt"
    cache_path = cache_dir / cache_name

    # ┏━━━━━━━━━━ Try loading existing cache ━━━━━━━━━━┓
    if cache_path.exists():
        print(f"[utils] Loading dataset from cache: {cache_path}")
        try:
            dataset = torch.load(cache_path, weights_only=False)
            return cache_path, dataset
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}. Recomputing...")

    # ┏━━━━━━━━━━ Build from CSVs ━━━━━━━━━━┓
    print("[utils] Cache not found — compiling dataset from CSVs...\n")
    raw_df = load_dataset_from_config(config)
    column_features = config["data"]["features"]["input"]

    # ┏━━━━━━━━━━ Prepare dataset ━━━━━━━━━━┓
    dataset = prepare_multi_gran_dataset(raw_df,
                                         column_features  = column_features,
                                         target_col       = config['data']['load']['target_col'],
                                         forecast_horizon = forecast_horizon,
                                         cfg              = config)

    # ┏━━━━━━━━━━ Save cache ━━━━━━━━━━┓
    try:
        torch.save(dataset, cache_path)
        print(f"\n[utils] Dataset saved to cache: {cache_path}")
    except Exception as e:
        print(f"[WARN] Failed to save cache: {e}")

    return cache_path, dataset


# ┏━━━━━━━━━━ Resolve caches ━━━━━━━━━━┓
def _resolve_caches(cfg: dict, explicit: str | None) -> dict[str, Path]:
    """Return {direction: cache_path} for each direction that has a cache, building any that are missing."""
    # ┏━━━━━━━━━━ Extract Configs ━━━━━━━━━━┓
    gran = cfg["data"]["load"]["granularity"]
    cache_dir = Path(cfg["paths"]["output_root"]) / m1_output_bucket(cfg) / "cache"
    m1_name = m1_model_name(cfg)
    gran_prefix = "multi" if gran == "all" else gran

    # ┏━━━━━━━━━━ Explicit cache path ━━━━━━━━━━┓
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Cache not found: {p}")
        parts = p.stem.split("_")
        inferred = "up"
        for d in ("up", "down"):
            if d in parts:
                inferred = d
                break
        return {inferred: p}

    # ┏━━━━━━━━━━ Auto-detect: find caches for both directions ━━━━━━━━━━┓
    result = {}
    for d in ("up", "down"):
        if gran_prefix == "multi":
            pattern = f"multi_{m1_name}_*_fee_{d}_*.pt"
        else:
            pattern = f"{gran_prefix}_*_{m1_name}_*_fee_{d}_*.pt"
        candidates = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            result[d] = candidates[0]
            print(f"[utils] Auto-selected cache ({d}): {candidates[0].name}")

    # ┏━━━━━━━━━━ Build any missing direction caches from config ━━━━━━━━━━┓
    for d in ("up", "down"):
        if d not in result:
            print(f"[utils] No existing cache found for direction='{d}'. Building from config...")
            cache_path, _ = _build_cache_from_config(cfg, direction=d)
            result[d] = cache_path

    return result


# ┏━━━━━━━━━━ Filter Multi-Gran Cache by Granularity ━━━━━━━━━━┓
def _filter_dataset_by_granularity(dataset, granularity: str) -> dict:
    """Subset a multi-gran cache to rows matching the requested granularity.

    Returns a plain dict mirroring the MultiGranDataset attribute names that
    downstream helpers already read (eng_features, labels, dates, returns,
    asset_ids, m1_pred_*, gran_ids). All per-row tensors/lists are filtered
    to the same granularity so that split_by_global_time and every subsequent
    step operate only on that slice.
    """
    # ┏━━━━━━━━━━ Check granularity ━━━━━━━━━━┓
    if granularity not in GRAN_TO_ID:
        raise ValueError(f"Unknown granularity '{granularity}'. Valid: {sorted(GRAN_TO_ID.keys())}")

    # ┏━━━━━━━━━━ Check if dataset is a dict ━━━━━━━━━━┓
    _is_dict = isinstance(dataset, dict)
    gran_ids = dataset["gran_ids"] if _is_dict else dataset.gran_ids
    if isinstance(gran_ids, torch.Tensor):
        gran_ids_np = gran_ids.numpy()
    else:
        gran_ids_np = np.asarray(gran_ids)

    # ┏━━━━━━━━━━ Mask for the requested granularity ━━━━━━━━━━┓
    mask = gran_ids_np == GRAN_TO_ID[granularity]
    n_match = int(mask.sum())
    if n_match == 0:
        present = sorted({k for k, v in GRAN_TO_ID.items() if (gran_ids_np == v).any()})
        raise ValueError(f"Granularity '{granularity}' not present in cache. Present: {present}")

    # ┏━━━━━━━━━━ Get function ━━━━━━━━━━┓
    def _get(key):
        return dataset[key] if _is_dict else getattr(dataset, key, None)

    # ┏━━━━━━━━━━ Subset function ━━━━━━━━━━┓
    def _subset(arr):
        if arr is None:
            return None
        if isinstance(arr, torch.Tensor):
            return arr[torch.as_tensor(mask)]
        if isinstance(arr, list):
            return [arr[i] for i in range(len(arr)) if mask[i]]
        return np.asarray(arr)[mask]

    # ┏━━━━━━━━━━ Filtered dataset ━━━━━━━━━━┓
    filtered = {"eng_features":    _subset(_get("eng_features")),
                "labels":          _subset(_get("labels")),
                "dates":           _subset(_get("dates")),
                "returns":         _subset(_get("returns")),
                "asset_ids":       _subset(_get("asset_ids")),
                "gran_ids":        _subset(_get("gran_ids")),
                "m1_pred_labels":  _subset(_get("m1_pred_labels")),
                "m1_pred_returns": _subset(_get("m1_pred_returns")),
                "m1_true_labels":  _subset(_get("m1_true_labels"))}

    # ┏━━━━━━━━━━ Asset map ━━━━━━━━━━┓
    asset_map = _get("asset_map")
    if asset_map is not None:
        filtered["asset_map"] = asset_map

    print(f"[utils] Granularity filter '{granularity}': kept {n_match:,} / {len(gran_ids_np):,} windows")
    return filtered


# ┏━━━━━━━━━━ Class names ━━━━━━━━━━┓
def _class_names(direction: str, mode: str) -> list[str]:
    suffix = "UP" if direction == "up" else "DN"
    prefix = mode.upper()
    return [f"NOT_{prefix}_{suffix}", f"{prefix}_{suffix}"]

# ┏━━━━━━━━━━ Infer Direction ━━━━━━━━━━┓
def _infer_direction(cache_path: Path) -> str:
    """Infer direction ('up'/'down') from cache filename."""
    parts = cache_path.stem.split("_")
    for d in ("up", "down"):
        if d in parts:
            return d
    return "up"

# ┏━━━━━━━━━━ Load Multi Cache ━━━━━━━━━━┓
def _load_multi_cache(cache_path: Path):
    """Load and validate a multi-granularity cache file."""
    multi = torch.load(cache_path, weights_only=False)
    if not hasattr(multi, "sub") or not hasattr(multi, "grans"):
        raise ValueError(f"Cache {cache_path.name} is not a multi-granularity cache (missing 'sub'/'grans').")
    return multi
