"""Generates per-mode config.yaml overrides for the TB-target pilot (tb_pilot_plan.md Phase 2).

Each k-mode (plus the tb_ref reference arm) gets its own validated config under
src/tb_pilot/configs/{mode}.yaml. Only four keys ever change (item 5 of the plan):

- `paths.csv_dir` points at the TB tree instead of TP's — the same value for every mode,
  the way "tp"/"fp"/"og" already share one `csv_dir` today.
- `paths.output_root` is unique per mode (`Output_tbpilot/{mode}`), which isolates the
  cache/HPO/AutoGluon directories per mode with no change to Pablo's resolvers — they
  already key off `paths.output_root` (Utils/utils.py: `_build_cache_from_config`,
  `_resolve_caches`, `_load_best_params`). This is what neutralizes F5 a-c.
- `data.load.meta_label_mode` selects this mode's subtree within the shared `csv_dir`.
- `evaluation.val_backtest` turns on the validation-split backtest Phase 3 needs to pick k.

Splits, features, fee and `thres` are untouched copies of the base config — deliberately:
`data.load.granularity` is NOT overridden (it stays "all"), because `_build_cache_from_config`
always writes `multi_*` cache filenames while `_resolve_caches` only looks for `multi_*` when
granularity=="all" — pinning it to "1d" here would desync the two and force a cache rebuild
on every SLURM task.

CLI:
    python -m tb_pilot.make_configs --tb-root Data_MLA/Kronos/Crypto/TB/horizon_7 \
        --ref --ks 0.5 1.0 1.5 2.0
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from Data_MLA.meta_label_tb import mode_name
from tb_pilot.schemas import SecondaryModelConfig

DEFAULT_BASE_CONFIG = "config.yaml"
DEFAULT_TB_ROOT = "Data_MLA/Kronos/Crypto/TB/horizon_7"
DEFAULT_OUT_DIR = "tb_pilot/configs"
DEFAULT_GRAN = "1d"


def requested_modes(ref: bool, ks: list[float]) -> list[str]:
    ks_as_modes: list[float | None] = list(ks)
    arms: list[float | None] = ([None] if ref else []) + ks_as_modes
    return [mode_name(k) for k in arms]


def discover_available_modes(tb_root: Path, gran: str) -> set[str]:
    """Mode names that actually have a `{gran}_{mode}` directory under `tb_root`."""
    if not tb_root.exists():
        return set()
    prefix = f"{gran}_"
    return {d.name.removeprefix(prefix) for d in tb_root.iterdir() if d.is_dir() and d.name.startswith(prefix)}


def _symbol_files(tb_root: Path, mode: str, gran: str) -> set[str]:
    return {p.name for p in (tb_root / f"{gran}_{mode}").glob("*.csv")}


def validate_tb_root(tb_root: Path, modes: list[str], gran: str) -> None:
    """Hard-fail unless `tb_root` is a real generated TB tree with every requested mode,
    and every requested mode covers the exact same symbol set.

    Catches a typo'd or stale --tb-root here, at config-generation time, rather than as a
    FileNotFoundError deep inside a SLURM array task hours later. The symbol-parity check
    catches a different, quieter failure: a partial or crashed regenerate for one k-arm
    would otherwise still pass (its directory exists, has some files) while silently
    comparing that arm against the others on a smaller universe — exactly the kind of
    apples-to-oranges comparison this pilot exists to avoid (tb_pilot_plan.md F5).
    """
    if not tb_root.exists():
        raise SystemExit(f"--tb-root does not exist: {tb_root}")
    if not (tb_root / "manifest.json").exists():
        raise SystemExit(f"--tb-root has no manifest.json (not a meta_label_tb.py output?): {tb_root}")
    available = discover_available_modes(tb_root, gran)
    missing = [m for m in modes if m not in available]
    if missing:
        raise SystemExit(
            f"mode(s) {missing} have no {gran}_<mode> directory under {tb_root} "
            f"(available: {sorted(available)})"
        )

    reference_mode = modes[0]
    reference_files = _symbol_files(tb_root, reference_mode, gran)
    for mode in modes[1:]:
        files = _symbol_files(tb_root, mode, gran)
        if files != reference_files:
            raise SystemExit(
                f"symbol mismatch between TB modes under {tb_root}: "
                f"{reference_mode} has {len(reference_files)} files, {mode} has {len(files)} "
                f"(only in {reference_mode}: {sorted(reference_files - files)}; "
                f"only in {mode}: {sorted(files - reference_files)})"
            )


def override_for_mode(base: dict, mode: str, tb_root: str) -> dict:
    """Return a new config dict with the TB pilot's four overrides applied and validated."""
    out = copy.deepcopy(base)
    out["paths"]["csv_dir"] = tb_root
    out["paths"]["output_root"] = f"Output_tbpilot/{mode}"
    out["data"]["load"]["meta_label_mode"] = mode
    out["evaluation"]["val_backtest"] = True
    SecondaryModelConfig.model_validate(out)
    return out


def write_mode_configs(
    base: dict,
    modes: list[str],
    tb_root_str: str,
    out_dir: Path,
    gran: str = DEFAULT_GRAN,
) -> dict[str, Path]:
    """Validate `tb_root_str` against the requested modes, then write one YAML per mode.

    `out_dir` becomes exactly this invocation's mode set: any `*.yaml` already there is
    removed first, so a leftover config from an earlier run with a different --ks/--ref
    (or a different --tb-root) can't sit alongside the new ones and get picked up by
    something that globs the directory instead of using this function's return value.
    """
    validate_tb_root(Path(tb_root_str), modes, gran)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.yaml"):
        stale.unlink()
    written = {}
    for mode in modes:
        cfg = override_for_mode(base, mode, tb_root_str)
        path = out_dir / f"{mode}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        written[mode] = path
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate per-mode config.yaml overrides for the TB-target pilot.")
    parser.add_argument("--base", type=Path, default=Path(DEFAULT_BASE_CONFIG))
    parser.add_argument("--tb-root", type=str, default=DEFAULT_TB_ROOT)
    parser.add_argument("--gran", type=str, default=DEFAULT_GRAN)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--ref", action="store_true", help="also write the tb_ref arm (k=None)")
    parser.add_argument("--ks", type=float, nargs="*", default=[], help="k multiples of sigma*sqrt(h)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    modes = requested_modes(args.ref, args.ks)
    if not modes:
        raise SystemExit("pass --ref and/or --ks")

    base = yaml.safe_load(args.base.read_text())
    SecondaryModelConfig.model_validate(base)

    written = write_mode_configs(base, modes, args.tb_root, args.out_dir, args.gran)
    for mode, path in written.items():
        print(f"[make_configs] wrote {path} (mode={mode})")


if __name__ == "__main__":
    main()
