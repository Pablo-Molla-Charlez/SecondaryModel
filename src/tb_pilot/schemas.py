"""Pydantic schema for config.yaml (tb_pilot_plan.md Phase 2).

This validates the config dict at the boundary — on load, and again after the TB pilot's
overrides are applied — so a typo or a bad path fails loudly here instead of surfacing
hours into a SLURM array job. It does not change how the pipeline *consumes* a config:
m2_pipeline.py and Utils/data/data.py keep reading the plain dict exactly as before;
`SecondaryModelConfig.model_validate(cfg)` is a pass/fail gate, not a new call site.

Every section uses `extra="forbid"`, so a renamed or mistyped key is a validation error,
not a silently-ignored no-op.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Meta-labeling modes: "tp" (true-positive — NOT take-profit, there is no take-profit exit
# anywhere in this pipeline), "fp" (false-positive), "og" (original/baseline), or the
# triple-barrier pilot's "tb_ref"/"tb_kNNN" (src/Data_MLA/meta_label_tb.py).
META_LABEL_MODE_PATTERN = re.compile(r"^(tp|fp|og|tb_ref|tb_k\d{3})$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    csv_dir: str
    output_root: str


class DataLoadConfig(_Strict):
    symbol: str | list[str] | None = None
    m1: str
    target_col: Literal["meta_label", "close", "ground_truth"]
    meta_label_mode: str
    direction: Literal["up", "down"]
    granularity: str
    forecast_horizon: int = Field(gt=0)

    @field_validator("meta_label_mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        if not META_LABEL_MODE_PATTERN.match(v):
            raise ValueError(f"meta_label_mode {v!r} must be tp/fp/og/tb_ref/tb_kNNN")
        return v


class DataSplitConfig(_Strict):
    start_date: date
    train_end: date
    val_end: date
    end_date: date

    @model_validator(mode="after")
    def _chronological(self) -> "DataSplitConfig":
        if not (self.start_date < self.train_end < self.val_end <= self.end_date):
            raise ValueError(
                "split dates must satisfy start_date < train_end < val_end <= end_date, "
                f"got {self.start_date}, {self.train_end}, {self.val_end}, {self.end_date}"
            )
        return self


class FeatureSelectionConfig(_Strict):
    enabled: bool
    methods: list[Literal["mda", "shap", "lime"]]
    top_k: int | None = None


class FeaturesConfig(_Strict):
    input: list[str]
    feature_selection: FeatureSelectionConfig


class DataConfig(_Strict):
    load: DataLoadConfig
    split: DataSplitConfig
    features: FeaturesConfig


class EvaluationConfig(_Strict):
    fee_per_trade: float = Field(ge=0)
    val_backtest: bool = False


class ExperimentConfig(_Strict):
    m1: str
    m2: list[str]
    direction: list[Literal["up", "down"]]
    granularity: list[str]


class SkipConfig(_Strict):
    hpo: bool
    training: bool
    edge: bool
    combined: bool
    feature_selection: bool


class HpoRuntimeConfig(_Strict):
    n_trials: int = Field(gt=0)
    seed: int


class TrainingRuntimeConfig(_Strict):
    paradigm_comparison: list[str] | None = None
    combined_backtest: list[str] | None = None
    comparison: list[str] | None = None
    all_grans: bool
    thres: str


class EdgeRuntimeConfig(_Strict):
    n_trials: int
    n_blocks: int
    k_test: int


class CombinedRuntimeConfig(_Strict):
    combined_backtest: list[str]


class FeatureSelectionRuntimeConfig(_Strict):
    cv_strategy: Literal["cpcv", "tscv", "pecv"]
    n_blocks: int
    k_test: int
    method: str
    scoring: Literal["accuracy", "precision", "roc_auc"]
    min_features: int = Field(ge=1)
    max_features: int
    take_n_best_combinations: int

    @model_validator(mode="after")
    def _feature_bounds(self) -> "FeatureSelectionRuntimeConfig":
        if self.max_features < self.min_features:
            raise ValueError("max_features must be >= min_features")
        return self


class RuntimeConfig(_Strict):
    skip: SkipConfig
    hpo: HpoRuntimeConfig
    training: TrainingRuntimeConfig
    edge: EdgeRuntimeConfig
    combined: CombinedRuntimeConfig
    feature_selection: FeatureSelectionRuntimeConfig


class SecondaryModelConfig(_Strict):
    """The full config.yaml schema. `model_validate(raw_dict)` is the only call site."""

    paths: PathsConfig
    data: DataConfig
    evaluation: EvaluationConfig
    experiment: ExperimentConfig
    runtime: RuntimeConfig
