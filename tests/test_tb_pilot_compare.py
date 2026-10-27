"""Comparison and reporting for the TB-target pilot (tb_pilot_plan.md Phase 5).

Fixtures build minimal fake `analysis_summary.json` + `*_trades.csv` pairs and a tiny
manifest, rather than depending on a real sweep run, so these stay fast and isolated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tb_pilot.compare import (
    ModeRun,
    backtest_key,
    check_cache_under_own_output_root,
    check_features_used_match,
    check_m1_trades_match,
    daily_returns,
    economics_table,
    freeze_choice,
    load_frozen_choice,
    mode_run_dir,
    moving_block_bootstrap_delta_sharpe,
    per_asset_table,
    require_unseal,
    target_characterization,
    write_confirmation_report,
    write_selection_report,
)

BASE_CFG = {
    "runtime": {"training": {"thres": "utility_nocal"}},
    "data": {"load": {"m1": "kronos"}},
}


def _backtest_dict(n_total=100, n_m2=20, sharpe=1.0, execution_rate=20.0, fee=0.002):
    return {
        "threshold": 0.5,
        "constraint_satisfied": True,
        "execution_rate": execution_rate,
        "n_total_trades": n_total,
        "n_m2_trades": n_m2,
        "m1_win_rate": 50.0,
        "m2_win_rate": 60.0,
        "m2_total_return": 5.0,
        "m1_total_return": 2.0,
        "bh_total_return": -1.0,
        "m2_sharpe": sharpe,
        "m1_sharpe": 0.5,
        "bh_sharpe": -0.2,
        "m2_max_drawdown": -10.0,
        "m1_max_drawdown": -12.0,
        "bh_max_drawdown": -20.0,
        "fee": fee,
    }


def _trades_df(n=4, m2_approved=(True, False, True, False), returns=(0.01, -0.02, 0.03, 0.0)):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"][:n]),
            "asset": ["BTC", "ETH", "BTC", "ETH"][:n],
            "return": list(returns)[:n],
            "m2_approved": list(m2_approved)[:n],
        }
    )


def _run(mode="tb_ref", model="xgboost", direction="up", cache_root="Output_tbpilot", n_total=100, sharpe=1.0,
          features=("f1", "f2"), trades=None):
    summary = {
        "cache": f"{cache_root}/{mode}/Kronos/cache/multi_kronos_7_fee_{direction}_abc.pt",
        "features_used": list(features),
        f"{model}_backtest_all_features": _backtest_dict(n_total=n_total, sharpe=sharpe),
        f"{model}_backtest_val_all_features": _backtest_dict(n_total=n_total, sharpe=sharpe),
    }
    return ModeRun(mode=mode, model=model, direction=direction, summary=summary, trades=trades if trades is not None else _trades_df())


# ---------------------------------------------------------------------------
# mode_run_dir
# ---------------------------------------------------------------------------


def test_mode_run_dir_mirrors_run_analysis_save_dir_convention():
    path = mode_run_dir(BASE_CFG, mode="tb_k100", model="xgboost", direction="up", granularity="1d")
    assert path == Path("Output_tbpilot/tb_k100/Kronos/xgboost/UP/Utility_Score_NoCal/1d_tb_k100")


def test_mode_run_dir_uses_utility_score_without_nocal_suffix_for_other_thres_modes():
    cfg = {"runtime": {"training": {"thres": "utility"}}, "data": {"load": {"m1": "kronos"}}}
    path = mode_run_dir(cfg, mode="tb_ref", model="rf", direction="down", granularity="1d")
    assert "Utility_Score" in path.parts
    assert "Utility_Score_NoCal" not in path.parts


# ---------------------------------------------------------------------------
# Hard-fail parity checks
# ---------------------------------------------------------------------------


def test_check_cache_under_own_output_root_passes_for_a_correctly_isolated_run():
    run = _run(mode="tb_k100", cache_root="Output_tbpilot")
    check_cache_under_own_output_root(run)  # does not raise


def test_check_cache_under_own_output_root_rejects_a_cache_from_a_different_mode():
    run = _run(mode="tb_k100")
    run.summary["cache"] = "Output_tbpilot/tb_ref/Kronos/cache/multi_kronos_7_fee_up_abc.pt"
    with pytest.raises(SystemExit, match="not under its own output root"):
        check_cache_under_own_output_root(run)


def test_check_cache_under_own_output_root_rejects_a_cache_outside_the_pilot_tree():
    run = _run(mode="tb_ref")
    run.summary["cache"] = "Output/Kronos/cache/multi_kronos_7_fee_up_abc.pt"
    with pytest.raises(SystemExit, match="not under"):
        check_cache_under_own_output_root(run)


def test_check_features_used_match_passes_when_identical():
    runs = [_run(mode="tb_ref", features=("f1", "f2")), _run(mode="tb_k100", features=("f1", "f2"))]
    check_features_used_match(runs)  # does not raise


def test_check_features_used_match_fires_on_a_mismatched_fixture():
    runs = [_run(mode="tb_ref", features=("f1", "f2")), _run(mode="tb_k100", features=("f1", "f3"))]
    with pytest.raises(SystemExit, match="features_used differs"):
        check_features_used_match(runs)


def test_check_m1_trades_match_passes_when_identical():
    runs = [_run(mode="tb_ref"), _run(mode="tb_k100")]
    check_m1_trades_match(runs, split="test")  # does not raise


def test_check_m1_trades_match_fires_when_n_total_trades_differs():
    runs = [_run(mode="tb_ref", n_total=100), _run(mode="tb_k100", n_total=99)]
    with pytest.raises(SystemExit, match="n_total_trades differs"):
        check_m1_trades_match(runs, split="test")


def test_check_m1_trades_match_fires_when_a_trade_return_differs():
    other_trades = _trades_df(returns=(0.01, -0.02, 0.03, 0.05))  # last return changed
    runs = [_run(mode="tb_ref"), _run(mode="tb_k100", trades=other_trades)]
    with pytest.raises(SystemExit, match="trade set"):
        check_m1_trades_match(runs, split="test")


def test_check_m1_trades_match_uses_the_validation_backtest_key_for_split_val():
    runs = [_run(mode="tb_ref", n_total=100), _run(mode="tb_k100", n_total=100)]
    check_m1_trades_match(runs, split="val")  # does not raise; also exercises backtest_key("val")


def test_backtest_key_selects_val_or_test_suffix():
    assert backtest_key("xgboost", "val") == "xgboost_backtest_val_all_features"
    assert backtest_key("xgboost", "test") == "xgboost_backtest_all_features"


# ---------------------------------------------------------------------------
# Section (c): paired moving-block bootstrap for delta-Sharpe
# ---------------------------------------------------------------------------


def test_moving_block_bootstrap_matches_direct_calc_for_the_point_estimate():
    rng = np.random.default_rng(0)
    returns_k = rng.normal(0.002, 0.01, size=200)
    returns_ref = rng.normal(0.0, 0.01, size=200)

    result = moving_block_bootstrap_delta_sharpe(returns_k, returns_ref, ann=16.0, block_len=10, n_boot=500, rng=rng)

    from Utils.backtest.engine import _calc_sharpe

    expected = _calc_sharpe(returns_k, 16.0) - _calc_sharpe(returns_ref, 16.0)
    assert result["delta_sharpe"] == pytest.approx(expected)
    assert result["ci_low"] <= result["delta_sharpe"] <= result["ci_high"]
    assert result["n_boot"] == 500
    assert result["block_len"] == 10


def test_moving_block_bootstrap_ci_straddles_zero_for_identical_series():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.01, size=150)

    result = moving_block_bootstrap_delta_sharpe(returns, returns, ann=16.0, block_len=7, n_boot=500, rng=rng)

    assert result["delta_sharpe"] == pytest.approx(0.0)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_moving_block_bootstrap_rejects_unequal_length_series():
    with pytest.raises(ValueError, match="equal-length"):
        moving_block_bootstrap_delta_sharpe(np.zeros(10), np.zeros(9), ann=16.0, block_len=7)


def test_moving_block_bootstrap_rejects_empty_series():
    with pytest.raises(ValueError, match="no returns"):
        moving_block_bootstrap_delta_sharpe(np.array([]), np.array([]), ann=16.0, block_len=7)


def test_daily_returns_is_simple_pct_change_dropping_the_first_nan():
    equity = pd.Series([1.0, 1.1, 1.21])
    out = daily_returns(equity)
    np.testing.assert_allclose(out, [0.1, 0.1])


# ---------------------------------------------------------------------------
# Section (a): target characterization from the manifest
# ---------------------------------------------------------------------------


def _manifest(mode_entries):
    return {"summary": {"tb_k100": mode_entries}}


def test_target_characterization_averages_equal_per_asset_direction():
    manifest = _manifest(
        [
            {
                "symbol": "BTC", "direction": "up", "n_eligible": 500,
                "tp_positive_rate": 0.4, "tb_positive_rate": 0.3, "flip_rate": 0.1,
                "exit_mix": {"profit": 0.3, "stop": 0.2, "expiry": 0.5}, "ambiguous_share": 0.05,
            },
            {
                "symbol": "BTC", "direction": "down", "n_eligible": 400,
                "tp_positive_rate": 0.2, "tb_positive_rate": 0.5, "flip_rate": 0.3,
                "exit_mix": {"profit": 0.1, "stop": 0.4, "expiry": 0.5}, "ambiguous_share": 0.15,
            },
        ]
    )
    out = target_characterization(manifest, "tb_k100")
    assert out["n_assets_x_directions"] == 2
    assert out["n_eligible_total"] == 900
    assert out["tp_positive_rate"] == pytest.approx(0.3)
    assert out["tb_positive_rate"] == pytest.approx(0.4)
    assert out["flip_rate"] == pytest.approx(0.2)
    assert out["exit_mix"]["profit"] == pytest.approx(0.2)
    assert out["ambiguous_share"] == pytest.approx(0.1)


def test_target_characterization_skips_entries_with_zero_eligible_rows():
    manifest = _manifest(
        [
            {"symbol": "TON", "direction": "up", "n_eligible": 0},
            {
                "symbol": "BTC", "direction": "up", "n_eligible": 500,
                "tp_positive_rate": 0.4, "tb_positive_rate": 0.3, "flip_rate": 0.1,
            },
        ]
    )
    out = target_characterization(manifest, "tb_k100")
    assert out["n_assets_x_directions"] == 1


def test_target_characterization_rejects_a_mode_with_no_eligible_entries():
    manifest = _manifest([{"symbol": "TON", "direction": "up", "n_eligible": 0}])
    with pytest.raises(SystemExit, match="no eligible entries"):
        target_characterization(manifest, "tb_k100")


# ---------------------------------------------------------------------------
# Section (b)/(d): economics and per-asset tables
# ---------------------------------------------------------------------------


def test_economics_table_has_one_row_per_run_with_the_expected_columns():
    runs = [_run(mode="tb_ref", model="xgboost"), _run(mode="tb_k100", model="xgboost")]
    df = economics_table(runs, split="test")
    assert len(df) == 2
    assert set(df["mode"]) == {"tb_ref", "tb_k100"}
    for col in ("m2_sharpe", "m1_sharpe", "coverage_pct", "n_m2_trades", "trades_per_day", "fee"):
        assert col in df.columns


def test_per_asset_table_is_equal_weight_and_m2_approved_only():
    run = _run()
    out = per_asset_table(run)
    # _trades_df: BTC rows approved=(True, True) returns (0.01, 0.03); ETH rows approved=(False, False)
    assert set(out["asset"]) == {"BTC"}
    row = out[out["asset"] == "BTC"].iloc[0]
    assert row["avg_return"] == pytest.approx(0.02)
    assert row["n_trades"] == 2


# ---------------------------------------------------------------------------
# Report writers -- exercises to_markdown() for real (needs the `tabulate` dep)
# ---------------------------------------------------------------------------


def test_write_selection_report_writes_csv_and_markdown_with_no_test_columns(tmp_path):
    runs = [_run(mode="tb_ref", model="xgboost"), _run(mode="tb_k100", model="xgboost")]
    economics_val = economics_table(runs, split="val")
    characterization = {"tb_k100": {"n_eligible_total": 900, "flip_rate": 0.2}}

    csv_path, md_path = write_selection_report(tmp_path, economics_val, characterization)

    assert csv_path.exists() and md_path.exists()
    csv_text = csv_path.read_text()
    assert "m2_sharpe" in csv_text
    assert "test" not in csv_text.lower()  # no test-split column ever entered this table
    md_text = md_path.read_text()
    assert "selection" in md_text.lower()
    assert "tb_k100" in md_text


def test_write_confirmation_report_writes_csv_and_markdown_with_the_frozen_choice(tmp_path):
    runs = [_run(mode="tb_ref", model="xgboost"), _run(mode="tb_k100", model="xgboost")]
    economics_test = economics_table(runs, split="test")
    frozen = {"chosen_mode": "tb_k100", "rule": "median val Sharpe"}
    delta_sharpe = {"tb_k100": {"xgboost": {"delta_sharpe": 0.1, "ci_low": -0.05, "ci_high": 0.25}}}

    csv_path, md_path = write_confirmation_report(tmp_path, frozen, economics_test, delta_sharpe)

    assert csv_path.exists() and md_path.exists()
    md_text = md_path.read_text()
    assert "tb_k100" in md_text
    assert "+0.100" in md_text
    assert "confirmation" in md_text.lower()


# ---------------------------------------------------------------------------
# Sealing: confirmation mode requires --unseal + a pre-existing frozen_choice.json
# ---------------------------------------------------------------------------


def test_require_unseal_refuses_without_the_flag():
    with pytest.raises(SystemExit, match="requires --unseal"):
        require_unseal(unseal=False, frozen_choice_path=None)


def test_require_unseal_refuses_without_a_frozen_choice_path():
    with pytest.raises(SystemExit, match="--frozen-choice"):
        require_unseal(unseal=True, frozen_choice_path=None)


def test_require_unseal_refuses_when_the_frozen_choice_file_does_not_exist(tmp_path):
    with pytest.raises(SystemExit, match="frozen_choice.json"):
        require_unseal(unseal=True, frozen_choice_path=tmp_path / "does_not_exist.json")


def test_require_unseal_returns_the_payload_once_both_are_given(tmp_path):
    path = tmp_path / "frozen_choice.json"
    freeze_choice(path, chosen_mode="tb_k100", rule_description="median val Sharpe")
    payload = require_unseal(unseal=True, frozen_choice_path=path)
    assert payload["chosen_mode"] == "tb_k100"


def test_freeze_choice_then_load_frozen_choice_round_trips(tmp_path):
    path = tmp_path / "frozen_choice.json"
    result = freeze_choice(path, chosen_mode="tb_ref", rule_description="placeholder")
    assert result["chosen_mode"] == "tb_ref"
    assert len(result["sha256"]) == 64

    loaded = load_frozen_choice(path)
    assert loaded["chosen_mode"] == "tb_ref"
    assert loaded["rule"] == "placeholder"


def test_load_frozen_choice_rejects_a_payload_missing_chosen_mode(tmp_path):
    path = tmp_path / "frozen_choice.json"
    path.write_text(json.dumps({"rule": "x"}))
    with pytest.raises(SystemExit, match="chosen_mode"):
        load_frozen_choice(path)
