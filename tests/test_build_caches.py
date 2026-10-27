"""Pre-building dataset caches for every TB-pilot mode config (tb_pilot_plan.md Phase 3, item 8).

Monkeypatches `_build_cache_from_config` so these stay fast — the real function loads
CSVs and builds a full MultiGranDataset, which belongs to an integration run, not a
unit test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import tb_pilot.build_caches as build_caches_mod
from tb_pilot.build_caches import build_caches_for_configs


@pytest.fixture
def mode_configs(tmp_path):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    paths = []
    for mode in ("tb_ref", "tb_k100"):
        cfg = {
            "data": {"load": {"meta_label_mode": mode}},
            "paths": {"output_root": f"Output_tbpilot/{mode}"},
        }
        path = configs_dir / f"{mode}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        paths.append(path)
    return paths


def test_build_caches_for_configs_builds_both_directions_for_every_mode(mode_configs, monkeypatch):
    calls = []

    def fake_build_cache(cfg, direction):
        calls.append((cfg["data"]["load"]["meta_label_mode"], direction))
        return Path(f"/fake/{cfg['data']['load']['meta_label_mode']}_{direction}.pt"), object()

    monkeypatch.setattr(build_caches_mod, "_build_cache_from_config", fake_build_cache)

    built = build_caches_for_configs(mode_configs)

    assert set(built) == {"tb_ref", "tb_k100"}
    assert {c[0] for c in calls} == {"tb_ref", "tb_k100"}
    assert {c[1] for c in calls} == {"up", "down"}
    assert len(calls) == 4
    for mode, dirs in built.items():
        assert set(dirs) == {"up", "down"}


def test_build_caches_for_configs_builds_sequentially_in_the_given_order(mode_configs, monkeypatch):
    order = []

    def fake_build_cache(cfg, direction):
        order.append((cfg["data"]["load"]["meta_label_mode"], direction))
        return Path("x"), object()

    monkeypatch.setattr(build_caches_mod, "_build_cache_from_config", fake_build_cache)

    build_caches_for_configs(mode_configs)

    assert order == [
        ("tb_ref", "up"),
        ("tb_ref", "down"),
        ("tb_k100", "up"),
        ("tb_k100", "down"),
    ]


def test_build_caches_for_configs_returns_the_paths_from_build_cache_from_config(mode_configs, monkeypatch):
    monkeypatch.setattr(
        build_caches_mod,
        "_build_cache_from_config",
        lambda cfg, direction: (Path(f"/cache/{cfg['data']['load']['meta_label_mode']}_{direction}.pt"), object()),
    )

    built = build_caches_for_configs(mode_configs)

    assert built["tb_ref"]["up"] == Path("/cache/tb_ref_up.pt")
    assert built["tb_k100"]["down"] == Path("/cache/tb_k100_down.pt")
