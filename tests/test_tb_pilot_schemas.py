"""Validation behavior of the config.yaml schema (tb_pilot_plan.md Phase 2).

The schema's job is to catch a malformed or mistyped config before it reaches a SLURM
array job, not to change how the pipeline consumes it — m2_pipeline.py and data.py keep
reading the plain dict exactly as before.
"""

from __future__ import annotations

import copy

import pytest
import yaml
from pydantic import ValidationError

from pathlib import Path

from tb_pilot.schemas import SecondaryModelConfig


@pytest.fixture(scope="module")
def raw_config() -> dict:
    here = Path(__file__).resolve().parents[1]
    return yaml.safe_load((here / "src" / "config.yaml").read_text())


def test_the_real_config_yaml_validates_as_is(raw_config):
    SecondaryModelConfig.model_validate(raw_config)


def test_an_unknown_top_level_key_is_rejected(raw_config):
    tampered = copy.deepcopy(raw_config)
    tampered["totally_bogus_section"] = {"x": 1}
    with pytest.raises(ValidationError):
        SecondaryModelConfig.model_validate(tampered)


def test_an_unknown_nested_key_is_rejected(raw_config):
    tampered = copy.deepcopy(raw_config)
    tampered["data"]["load"]["metaa_label_mode"] = "tp"  # typo, not a real key
    with pytest.raises(ValidationError):
        SecondaryModelConfig.model_validate(tampered)


@pytest.mark.parametrize("mode", ["tp", "fp", "og", "tb_ref", "tb_k050", "tb_k100", "tb_k200"])
def test_known_meta_label_modes_are_accepted(raw_config, mode):
    tampered = copy.deepcopy(raw_config)
    tampered["data"]["load"]["meta_label_mode"] = mode
    SecondaryModelConfig.model_validate(tampered)


@pytest.mark.parametrize("mode", ["TP", "tb", "tb_k1", "tb_kabc", "bogus"])
def test_unknown_meta_label_modes_are_rejected(raw_config, mode):
    tampered = copy.deepcopy(raw_config)
    tampered["data"]["load"]["meta_label_mode"] = mode
    with pytest.raises(ValidationError):
        SecondaryModelConfig.model_validate(tampered)


def test_misordered_split_dates_are_rejected(raw_config):
    tampered = copy.deepcopy(raw_config)
    tampered["data"]["split"]["train_end"] = tampered["data"]["split"]["start_date"]
    with pytest.raises(ValidationError):
        SecondaryModelConfig.model_validate(tampered)


def test_a_negative_forecast_horizon_is_rejected(raw_config):
    tampered = copy.deepcopy(raw_config)
    tampered["data"]["load"]["forecast_horizon"] = -1
    with pytest.raises(ValidationError):
        SecondaryModelConfig.model_validate(tampered)


def test_validation_does_not_mutate_the_input_dict(raw_config):
    before = copy.deepcopy(raw_config)
    SecondaryModelConfig.model_validate(raw_config)
    assert raw_config == before
