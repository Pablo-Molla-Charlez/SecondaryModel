"""Per-mode config generation for the TB-target pilot (tb_pilot_plan.md Phase 2).

Fixtures build a minimal fake TB label tree (empty CSVs, just enough for the existence
checks) rather than depending on a real generator run, so these stay fast and isolated.
"""

from __future__ import annotations

import copy

import pytest
import yaml
from pydantic import ValidationError

from tb_pilot.make_configs import (
    discover_available_modes,
    override_for_mode,
    validate_tb_root,
    write_mode_configs,
)

GRAN = "1d"
SYMBOLS = ("BTCUSDT", "ETHUSDT")


@pytest.fixture
def base_config() -> dict:
    from pathlib import Path

    here = Path(__file__).resolve().parents[1]
    return yaml.safe_load((here / "src" / "config.yaml").read_text())


@pytest.fixture
def tb_root(tmp_path):
    root = tmp_path / "TB" / "horizon_7"
    for mode in ("tb_ref", "tb_k100"):
        mode_dir = root / f"{GRAN}_{mode}"
        mode_dir.mkdir(parents=True)
        for symbol in SYMBOLS:
            (mode_dir / f"{symbol}_up.csv").write_text("date,close\n")
            (mode_dir / f"{symbol}_down.csv").write_text("date,close\n")
    (root / "manifest.json").write_text("{}")
    return root


def test_override_for_mode_sets_exactly_the_four_keys(base_config):
    out = override_for_mode(base_config, mode="tb_k100", tb_root="Data_MLA/Kronos/Crypto/TB/horizon_7")

    assert out["paths"]["csv_dir"] == "Data_MLA/Kronos/Crypto/TB/horizon_7"
    assert out["paths"]["output_root"] == "Output_tbpilot/tb_k100"
    assert out["data"]["load"]["meta_label_mode"] == "tb_k100"
    assert out["evaluation"]["val_backtest"] is True

    # Everything else is untouched.
    untouched = copy.deepcopy(out)
    for key, value in untouched["evaluation"].items():
        if key != "val_backtest":
            assert value == base_config["evaluation"][key]
    assert out["data"]["split"] == base_config["data"]["split"]
    assert out["data"]["features"] == base_config["data"]["features"]
    assert out["runtime"] == base_config["runtime"]
    assert out["experiment"] == base_config["experiment"]


def test_override_for_mode_does_not_mutate_the_base_dict(base_config):
    before = copy.deepcopy(base_config)
    override_for_mode(base_config, mode="tb_ref", tb_root="Data_MLA/Kronos/Crypto/TB/horizon_7")
    assert base_config == before


def test_override_for_mode_rejects_an_invalid_mode_name(base_config):
    with pytest.raises(ValidationError):
        override_for_mode(base_config, mode="not_a_real_mode", tb_root="Data_MLA/Kronos/Crypto/TB/horizon_7")


def test_discover_available_modes_lists_what_the_tree_actually_has(tb_root):
    assert discover_available_modes(tb_root, gran=GRAN) == {"tb_ref", "tb_k100"}


def test_validate_tb_root_passes_for_modes_present_in_the_tree(tb_root):
    validate_tb_root(tb_root, modes=["tb_ref", "tb_k100"], gran=GRAN)


def test_validate_tb_root_rejects_a_mode_the_tree_does_not_have(tb_root):
    with pytest.raises(SystemExit):
        validate_tb_root(tb_root, modes=["tb_ref", "tb_k999"], gran=GRAN)


def test_validate_tb_root_rejects_a_missing_root(tmp_path):
    with pytest.raises(SystemExit):
        validate_tb_root(tmp_path / "does_not_exist", modes=["tb_ref"], gran=GRAN)


def test_validate_tb_root_rejects_a_root_with_no_manifest(tmp_path):
    root = tmp_path / "TB" / "horizon_7"
    (root / f"{GRAN}_tb_ref").mkdir(parents=True)
    with pytest.raises(SystemExit):
        validate_tb_root(root, modes=["tb_ref"], gran=GRAN)


def test_validate_tb_root_rejects_a_symbol_mismatch_between_modes(tb_root):
    # Simulate a partial/crashed regenerate: tb_k100 is missing one symbol's files.
    (tb_root / f"{GRAN}_tb_k100" / "ETHUSDT_up.csv").unlink()
    (tb_root / f"{GRAN}_tb_k100" / "ETHUSDT_down.csv").unlink()
    with pytest.raises(SystemExit, match="symbol mismatch"):
        validate_tb_root(tb_root, modes=["tb_ref", "tb_k100"], gran=GRAN)


def test_write_mode_configs_writes_one_valid_yaml_per_mode(base_config, tb_root, tmp_path):
    out_dir = tmp_path / "configs"
    written = write_mode_configs(
        base_config,
        modes=["tb_ref", "tb_k100"],
        tb_root_str=str(tb_root),
        out_dir=out_dir,
        gran=GRAN,
    )

    assert set(written) == {"tb_ref", "tb_k100"}
    for mode, path in written.items():
        assert path.exists()
        loaded = yaml.safe_load(path.read_text())
        assert loaded["data"]["load"]["meta_label_mode"] == mode
        assert loaded["paths"]["output_root"] == f"Output_tbpilot/{mode}"
        # Round-trips through the schema cleanly too (not just a dict dump).
        from tb_pilot.schemas import SecondaryModelConfig

        SecondaryModelConfig.model_validate(loaded)


def test_write_mode_configs_clears_stale_yaml_from_a_previous_invocation(base_config, tb_root, tmp_path):
    out_dir = tmp_path / "configs"
    write_mode_configs(base_config, modes=["tb_ref", "tb_k100"], tb_root_str=str(tb_root), out_dir=out_dir, gran=GRAN)
    assert {p.name for p in out_dir.glob("*.yaml")} == {"tb_ref.yaml", "tb_k100.yaml"}

    write_mode_configs(base_config, modes=["tb_ref"], tb_root_str=str(tb_root), out_dir=out_dir, gran=GRAN)
    assert {p.name for p in out_dir.glob("*.yaml")} == {"tb_ref.yaml"}
