"""Pre-builds dataset caches for every mode config before any training runs
(tb_pilot_plan.md Phase 3, item 8).

`_build_cache_from_config` (Utils/utils.py) keys its cache directory off
`paths.output_root`, which make_configs.py already set to `Output_tbpilot/{mode}`
per mode. Different modes can therefore never collide on the same cache file
(tb_pilot_plan.md F5 a/b/c). This script just builds them all up front, sequentially,
so a SLURM array job with many tasks per mode doesn't have several tasks racing to
build the same mode's cache the first time they run.

CLI:
    python -m tb_pilot.build_caches --configs-dir tb_pilot/configs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from Utils.utils import _build_cache_from_config

DEFAULT_CONFIGS_DIR = "tb_pilot/configs"


def build_caches_for_configs(config_paths: list[Path]) -> dict[str, dict[str, Path]]:
    """Build (or load) the up/down caches for each config in `config_paths`.

    Returns `{mode: {"up": cache_path, "down": cache_path}}`, in the same order
    as `config_paths`. Each config's two directions are built one after another
    — never in parallel — so this can run ahead of a SLURM array job without
    two tasks racing to write the same `.pt` file.
    """
    built: dict[str, dict[str, Path]] = {}
    for path in config_paths:
        cfg = yaml.safe_load(path.read_text())
        mode = path.stem
        print(f"[build_caches] {mode}: building caches from {path}")
        built[mode] = {}
        for direction in ("up", "down"):
            cache_path, _ = _build_cache_from_config(cfg, direction=direction)
            built[mode][direction] = cache_path
    return built


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-build dataset caches for every TB-pilot mode config.")
    parser.add_argument("--configs-dir", type=Path, default=Path(DEFAULT_CONFIGS_DIR))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_paths = sorted(args.configs_dir.glob("*.yaml"))
    if not config_paths:
        raise SystemExit(f"no *.yaml configs found under {args.configs_dir}")

    built = build_caches_for_configs(config_paths)
    for mode, dirs in built.items():
        for direction, path in dirs.items():
            print(f"[build_caches] {mode}/{direction}: {path}")


if __name__ == "__main__":
    main()
