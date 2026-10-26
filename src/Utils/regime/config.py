"""Config loading, path resolution and split windowing for the regime package.

These helpers are shared by the Stage 0 entrypoint (``train_jump_model``) and the
analysis script. They live here — not in the CLI entrypoint — so that consumers
can read config, resolve output/data roots, and slice train/val/test windows
without importing the argparse/fit machinery of the entrypoint.

Split-date semantics (matched to M2):
    train ≤ train_end < val ≤ val_end < test ≤ end_date
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd
import yaml

from Utils.regime.paths import DEFAULT_CONFIG_PATH, RAW_ROOT, SRC_ROOT

# Default split boundaries (mirror config.yaml). The JM itself is unsupervised;
# BIC selection fits on Train+Val (Test held out); Test labels are emitted for
# downstream diagnostics.
DEFAULT_START_DATE = "2024-07-01"
DEFAULT_TRAIN_END = "2025-09-30"
DEFAULT_VAL_END = "2025-12-31"
DEFAULT_END_DATE = "2026-05-31"
DEFAULT_REGIME_VARIANT = "sparse"
DEFAULT_REGIME_K = 2
DEFAULT_REGIME_JUMP_PENALTY = 20.0
REGIME_VARIANTS = ("standard", "continuous", "sparse", "sparse_continuous")
REGIME_SCOPES = ("asset", "global")


@dataclass(frozen=True)
class SplitConfig(Mapping[str, str | None]):
    """Typed train/validation/test split dates resolved from config + overrides."""

    start_date: str | None
    train_end: str | None
    val_end: str | None
    end_date: str | None
    config_path: str | None = None

    def __getitem__(self, key: str) -> str | None:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    def as_dict(self) -> dict[str, str | None]:
        return {
            "start_date": self.start_date,
            "train_end": self.train_end,
            "val_end": self.val_end,
            "end_date": self.end_date,
            "config_path": self.config_path,
        }


def _resolve_config_path(config_path: str | Path) -> Path:
    """Resolve config paths from repo root or src/ working directories."""
    path = Path(config_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([SRC_ROOT / path, SRC_ROOT.parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No config file found at {config_path}")


def _expand_env_vars(obj):
    """Recursively expand environment variables in YAML-loaded config values."""
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def load_regime_config(config_path: str | Path | None = DEFAULT_CONFIG_PATH) -> tuple[dict, Path | None]:
    """Load the regime config and return ``(config, resolved_config_path)``."""
    if config_path is None:
        return {}, None
    resolved = _resolve_config_path(config_path)
    with open(resolved, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return _expand_env_vars(cfg), resolved


def _resolve_configured_path(value: str | Path, config_file: Path | None) -> Path:
    """Resolve absolute paths as-is and relative paths from the config directory."""
    path = Path(value)
    if path.is_absolute():
        return path
    base = config_file.parent if config_file is not None else SRC_ROOT
    return (base / path).resolve()


def _required_config_path(cfg: dict,
                          config_file: Path | None,
                          key: str,
                          override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    value = cfg.get("paths", {}).get(key)
    if value in (None, ""):
        origin = str(config_file) if config_file is not None else "config"
        raise KeyError(f"Missing required config key paths.{key} in {origin}")
    return _resolve_configured_path(value, config_file)


def resolve_real_data_dir(config_path: str | Path | None = DEFAULT_CONFIG_PATH,
                          override: str | Path | None = None,
                          cfg: dict | None = None,
                          config_file: Path | None = None) -> Path:
    """Resolve required ``paths.real_data_dir`` from the current config schema."""
    if cfg is None:
        cfg, config_file = load_regime_config(config_path)
    return _required_config_path(cfg, config_file, "real_data_dir", override)


def resolve_raw_data_dir(config_path: str | Path | None = DEFAULT_CONFIG_PATH,
                         override: str | Path | None = None,
                         cfg: dict | None = None,
                         config_file: Path | None = None) -> Path:
    """Resolve ``paths.raw_data_dir`` for OHLCV source data."""
    if override is not None:
        return Path(override)
    if cfg is None:
        cfg, config_file = load_regime_config(config_path)
    if config_file is None and not cfg:
        return RAW_ROOT
    return _required_config_path(cfg, config_file, "raw_data_dir")


def resolve_output_root(config_path: str | Path | None = DEFAULT_CONFIG_PATH,
                        override: str | Path | None = None,
                        cfg: dict | None = None,
                        config_file: Path | None = None) -> Path:
    """Resolve required ``paths.output_root`` from the current config schema."""
    if cfg is None:
        cfg, config_file = load_regime_config(config_path)
    if override is None and config_file is None and not cfg:
        return SRC_ROOT / "Output"
    return _required_config_path(cfg, config_file, "output_root", override)


def load_regime_model_defaults(cfg: dict | None = None) -> dict[str, object]:
    """Return config-driven default JM parameters for regime consumers."""
    load_cfg = (cfg or {}).get("data", {}).get("load", {})
    variant = str(load_cfg.get("regime_variant", DEFAULT_REGIME_VARIANT))
    if variant not in REGIME_VARIANTS:
        raise ValueError(f"Unknown regime_variant '{variant}'. Choices: {REGIME_VARIANTS}")
    return {
        "variant": variant,
        "k": int(load_cfg.get("regime_k", DEFAULT_REGIME_K)),
        "jump_penalty": float(load_cfg.get("regime_jump_penalty", DEFAULT_REGIME_JUMP_PENALTY)),
    }


def load_regime_scope(cfg: dict | None = None) -> str:
    """Return whether downstream loaders use asset-scoped or pooled global JMs."""
    load_cfg = (cfg or {}).get("data", {}).get("load", {})
    scope = str(load_cfg.get("regime_scope", "asset")).strip().lower()
    if scope not in REGIME_SCOPES:
        raise ValueError(f"Unknown regime_scope '{scope}'. Choices: {REGIME_SCOPES}")
    return scope


def load_split_config(config_path: str | Path | None = DEFAULT_CONFIG_PATH,
                      start_date: str | None = None,
                      train_end: str | None = None,
                      val_end: str | None = None,
                      end_date: str | None = None) -> SplitConfig:
    """Resolve the train/val/test split dates, overrides applied last.

    Returns the four split dates (``start_date``/``train_end``/``val_end``/
    ``end_date``) plus ``config_path`` — the resolved config file path (or
    ``None``) — so a caller can re-open the same config later (e.g. the analysis
    script reloads it for M1/M2 paths) without re-resolving it.
    """
    split_cfg: dict[str, str | None] = {
        "start_date": DEFAULT_START_DATE,
        "train_end": DEFAULT_TRAIN_END,
        "val_end": DEFAULT_VAL_END,
        "end_date": DEFAULT_END_DATE,
    }
    if config_path is not None:
        cfg, resolved = load_regime_config(config_path)
        split_cfg.update({
            k: v for k, v in (cfg.get("data", {}).get("split", {}) or {}).items()
            if v is not None
        })
        split_cfg["config_path"] = str(resolved)
    else:
        split_cfg["config_path"] = None

    overrides = {
        "start_date": start_date,
        "train_end": train_end,
        "val_end": val_end,
        "end_date": end_date,
    }
    split_cfg.update({k: v for k, v in overrides.items() if v is not None})
    return SplitConfig(
        start_date=split_cfg["start_date"],
        train_end=split_cfg["train_end"],
        val_end=split_cfg["val_end"],
        end_date=split_cfg["end_date"],
        config_path=split_cfg["config_path"],
    )


def clip_ohlcv_to_split_window(ohlcv: pd.DataFrame, split_cfg: Mapping[str, str | None]) -> pd.DataFrame:
    """Apply config start/end bounds before feature construction."""
    out = ohlcv
    if split_cfg.get("start_date") is not None:
        out = out.loc[out.index >= pd.Timestamp(split_cfg["start_date"])]
    if split_cfg.get("end_date") is not None:
        out = out.loc[out.index <= pd.Timestamp(split_cfg["end_date"])]
    return out


def split_frame_by_config(frame: pd.DataFrame | pd.Series,
                          split_cfg: Mapping[str, str | None]) -> dict[str, pd.DataFrame | pd.Series]:
    """Split an indexed frame/series using the same date semantics as M2."""
    train_end = pd.Timestamp(split_cfg["train_end"])
    val_end = pd.Timestamp(split_cfg["val_end"])
    return {
        "train": frame.loc[frame.index <= train_end],
        "val": frame.loc[(frame.index > train_end) & (frame.index <= val_end)],
        "test": frame.loc[frame.index > val_end],
    }


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_START_DATE",
    "DEFAULT_TRAIN_END",
    "DEFAULT_VAL_END",
    "DEFAULT_END_DATE",
    "DEFAULT_REGIME_VARIANT",
    "DEFAULT_REGIME_K",
    "DEFAULT_REGIME_JUMP_PENALTY",
    "REGIME_SCOPES",
    "SplitConfig",
    "load_regime_config",
    "load_regime_model_defaults",
    "load_regime_scope",
    "resolve_real_data_dir",
    "resolve_raw_data_dir",
    "resolve_output_root",
    "load_split_config",
    "clip_ohlcv_to_split_window",
    "split_frame_by_config",
]
