"""
Utility classes and functions for Bi-FAST training.
"""
import json
import torch
import yaml
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional
from Utils.data_preprocessing import (load_dataset_from_config, 
                                      prepare_multi_gran_dataset, 
                                      prepare_multi_asset_dataset, 
                                      GRAN_SEQ_LEN)

def model_label(model_name: str) -> str:
    """Canonical short label for a model name."""
    return {"rf": "RF", "xgboost": "XGB", "autogluon": "AG"}.get(model_name, model_name.upper())


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
def _load_config(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

# ┏━━━━━━━━━━ Build cache from config ━━━━━━━━━━┓
def _build_cache_from_config(cfg: dict) -> tuple[Path, object]:
    """Build dataset cache from config.yaml, mirroring kronos_clas.py cache logic.

    Returns (cache_path, dataset).  If the cache already exists on disk it is
    loaded; otherwise the dataset is compiled from CSVs and saved.
    """
    # ┏━━━━━━━━━━ Extract Configs ━━━━━━━━━━┓
    data_cfg         = cfg.get("data", {})
    load_cfg         = data_cfg.get("load", {})
    split_cfg        = data_cfg.get("split", {})
    feat_cfg         = data_cfg.get("features", {})
    granularity      = load_cfg.get("granularity", "1h")
    direction        = load_cfg.get("direction", "up")
    forecast_horizon = cfg.get("main_model", {}).get("forecast_horizon", 7)
    is_multi_gran    = (granularity == "all")
    seq_len          = GRAN_SEQ_LEN.get(granularity, split_cfg.get("context_length", 48))

    # ┏━━━━━━━━━━ Deterministic hash ━━━━━━━━━━┓
    data_signature = {
        "granularity":      load_cfg.get("granularity"),
        "meta_label_mode":  load_cfg.get("meta_label_mode"),
        "direction":        direction,
        "start_date":       split_cfg.get("start_date"),
        "end_date":         split_cfg.get("end_date"),
        "context_length":   seq_len,
        "forecast_horizon": forecast_horizon,
        "features":         feat_cfg,
        "data_root":        str(Path(cfg.get("paths", {}).get("csv_dir", ".")).parent),
    }

    # ┏━━━━━━━━━━ Dump hash ━━━━━━━━━━┓
    cfg_str  = json.dumps(data_signature, sort_keys=True)
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:10]

    # ┏━━━━━━━━━━ Build cache path ━━━━━━━━━━┓
    cache_dir = Path(cfg["paths"]["output_root"]) / "Kronos" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if is_multi_gran:
        cache_name = f"multi_{forecast_horizon}_fee_{direction}_{cfg_hash}.pt"
    else:
        cache_name = f"{granularity}_{seq_len}_{forecast_horizon}_fee_{direction}_{cfg_hash}.pt"
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
    raw_df = load_dataset_from_config(cfg)
    column_features = feat_cfg.get("input", ["open", "high", "low", "close", "volume"])

    # ┏━━━━━━━━━━ Prepare dataset ━━━━━━━━━━┓
    if is_multi_gran:
        dataset = prepare_multi_gran_dataset(raw_df,
                                            column_features  = column_features,
                                            target_col       = load_cfg.get("target_col", "ground_truth"),
                                            forecast_horizon = forecast_horizon,
                                            cfg              = cfg)
    else:
        dataset = prepare_multi_asset_dataset(raw_df,
                                              seq_len          = seq_len,
                                              column_features  = column_features,
                                              target_col       = load_cfg.get("target_col", "ground_truth"),
                                              forecast_horizon = forecast_horizon,
                                              cfg              = cfg)

    # ┏━━━━━━━━━━ Save cache ━━━━━━━━━━┓
    try:
        torch.save(dataset, cache_path)
        print(f"\n[utils] Dataset saved to cache: {cache_path}")
    except Exception as e:
        print(f"[WARN] Failed to save cache: {e}")

    return cache_path, dataset


# ┏━━━━━━━━━━ Resolve caches ━━━━━━━━━━┓
def _resolve_caches(cfg: dict, explicit: str | None) -> dict[str, Path]:
    """Return {direction: cache_path} for each direction that has a cache."""
    # ┏━━━━━━━━━━ Extract Configs ━━━━━━━━━━┓
    gran = cfg["data"]["load"]["granularity"]
    cache_dir = Path(cfg["paths"]["output_root"]) / "Kronos" / "cache"

    # ┏━━━━━━━━━━ Explicit cache path ━━━━━━━━━━┓
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Cache not found: {p}")
        # ┏━━━━━━━━━━ Infer direction from filename ━━━━━━━━━━┓
        parts = p.stem.split("_")
        direction = "up"
        for d in ("up", "down"):
            if d in parts:
                direction = d
                break
        return {direction: p}

    # ┏━━━━━━━━━━ Auto-detect: find caches for both directions ━━━━━━━━━━┓
    result = {}
    for direction in ("up", "down"):
        candidates = sorted(cache_dir.glob(f"{gran}_*_fee_{direction}_*.pt"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            result[direction] = candidates[0]
            print(f"[utils] Auto-selected cache ({direction}): {candidates[0].name}")

    if not result:
        # ┏━━━━━━━━━━ Fallback: try old naming without direction ━━━━━━━━━━┓
        candidates = sorted(cache_dir.glob(f"{gran}_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            direction = cfg["data"]["load"].get("direction", "up").lower()
            result[direction] = candidates[0]
            print(f"[utils] Auto-selected cache (legacy): {candidates[0].name}")

    if not result:
        # ┏━━━━━━━━━━ No cache found — build one from config ━━━━━━━━━━┓
        print(f"[utils] No existing cache found for granularity={gran}. Building from config...")
        cache_path, _ = _build_cache_from_config(cfg)
        direction = cfg["data"]["load"].get("direction", "up").lower()
        result[direction] = cache_path
    return result


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
