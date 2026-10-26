"""Filesystem anchors shared across the regime package (Stage 0).

These roots were previously recomputed from ``__file__`` in three sibling
modules (features, storage, train_jump_model), which let them drift and made
the raw-data / output layout an implicit, duplicated contract. Owning them in
one module keeps every reader and writer pointing at the same tree.

``src/`` is ``parents[2]`` of this file (Utils/regime/paths.py), and the data
and output trees live directly under it.
"""

from __future__ import annotations

from pathlib import Path

# src/ — the import root that every other anchor is derived from.
SRC_ROOT = Path(__file__).resolve().parents[2]

# Raw OHLCV(A) CSVs, laid out as Data_MLA/Raw/<granularity>/<ASSET>.csv.
RAW_ROOT = SRC_ROOT / "Data_MLA" / "Raw"

# Default persisted-fit tree (Output/Jump_Models/<asset>/<gran>/<slug>/).
# Writers normally resolve the output root from config; this is the static
# default readers fall back to when no config-derived root is supplied.
JUMP_MODELS_ROOT = SRC_ROOT / "Output" / "Jump_Models"

# Repo config consumed by the Stage 0 entrypoint and the analysis script.
DEFAULT_CONFIG_PATH = SRC_ROOT / "config.yaml"


__all__ = ["SRC_ROOT", "RAW_ROOT", "JUMP_MODELS_ROOT", "DEFAULT_CONFIG_PATH"]
