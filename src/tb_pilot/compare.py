"""Comparison and reporting for the TB-target pilot (tb_pilot_plan.md Phase 5).

Reads each mode's `analysis_summary.json` (written by src/m2_pipeline.py's
run_analysis, one per {mode, model, direction}) and its sibling trades CSV, and
writes Output_tbpilot/comparison/{selection,confirmation}.{csv,md}.

Hard-fail parity checks protect the one invariant the whole pilot depends on
(tb_pilot_plan.md F9): only the *label* changes between modes. M1 ("all trades")
execution is label-independent, so its trade set, its summary stats, and the
feature set used to train M2 must be identical across every mode for a given
(model, direction, split) — any difference means something leaked (a stale
cache, a wrong output_root, ...) and every downstream number is suspect.

Selection mode (tb_pilot_plan.md item 11) reads validation only, so k is chosen
before test is ever looked at. Confirmation mode requires `--unseal` plus a
`frozen_choice.json` that already exists — a procedural guard against motivated
re-selection, enforced by `require_unseal`, not something a normal invocation
can bypass by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.backtest.engine import _annualization_factor, _build_spread_equity, _calc_sharpe
from Utils.utils import m1_output_bucket

DEFAULT_OUTPUT_ROOT_BASE = "Output_tbpilot"  # must match make_configs.py's override_for_mode


# ---------------------------------------------------------------------------
# Loading one (mode, model, direction) run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeRun:
    """One (mode, model, direction)'s analysis_summary.json + M1 ("all trades") rows."""

    mode: str
    model: str
    direction: str
    summary: dict
    trades: pd.DataFrame  # date, asset, return, m2_approved, ... (file_prefix + "_trades.csv")


def mode_run_dir(
    cfg: dict,
    mode: str,
    model: str,
    direction: str,
    granularity: str,
    output_root_base: str = DEFAULT_OUTPUT_ROOT_BASE,
) -> Path:
    """Mirrors run_analysis()'s save_dir (m2_pipeline.py) for a TB-pilot mode's own output root."""
    thres_mode = cfg.get("runtime", {}).get("training", {}).get("thres", "utility")
    thres_folder = "Utility_Score_NoCal" if thres_mode == "utility_nocal" else "Utility_Score"
    return (
        Path(output_root_base)
        / mode
        / m1_output_bucket(cfg)
        / model
        / direction.upper()
        / thres_folder
        / f"{granularity}_{mode}"
    )


def backtest_key(model: str, split: str) -> str:
    """The analysis_summary.json key run_analysis() stores this split's backtest under."""
    suffix = "val_all_features" if split == "val" else "all_features"
    return f"{model}_backtest_{suffix}"


def load_mode_run(
    cfg: dict,
    mode: str,
    model: str,
    direction: str,
    granularity: str,
    file_prefix: str,
    output_root_base: str = DEFAULT_OUTPUT_ROOT_BASE,
) -> ModeRun:
    run_dir = mode_run_dir(cfg, mode, model, direction, granularity, output_root_base)
    summary = json.loads((run_dir / "analysis_summary.json").read_text())
    trades = pd.read_csv(run_dir / f"{file_prefix}_trades.csv", parse_dates=["date"])
    return ModeRun(mode=mode, model=model, direction=direction, summary=summary, trades=trades)


# ---------------------------------------------------------------------------
# Hard-fail parity checks (tb_pilot_plan.md item 11)
# ---------------------------------------------------------------------------


def check_cache_under_own_output_root(run: ModeRun, output_root_base: str = DEFAULT_OUTPUT_ROOT_BASE) -> None:
    """Catches cross-mode cache contamination (tb_pilot_plan.md F5): the cache path a
    mode's summary records must sit under that mode's own output root, never another's."""
    cache_path = str(run.summary.get("cache", ""))
    parts = Path(cache_path).parts
    ok = output_root_base in parts and parts[parts.index(output_root_base) + 1 : parts.index(output_root_base) + 2] == (
        run.mode,
    )
    if not ok:
        raise SystemExit(
            f"{run.mode}/{run.model}/{run.direction}: cache path {cache_path!r} is not under "
            f"its own output root {output_root_base}/{run.mode}/ (tb_pilot_plan.md F5)"
        )


def check_features_used_match(runs: list[ModeRun]) -> None:
    """features_used must be identical across modes for the same (model, direction) —
    the TB target changes labels, never the engineered feature set."""
    if len(runs) < 2:
        return
    reference = runs[0]
    for run in runs[1:]:
        if run.summary.get("features_used") != reference.summary.get("features_used"):
            raise SystemExit(
                f"features_used differs between {reference.mode} and {run.mode} "
                f"(model={run.model}, direction={run.direction}) — expected identical feature sets"
            )


def _trade_identity(trades: pd.DataFrame) -> frozenset[tuple[str, str, float]]:
    return frozenset((str(d), str(a), round(float(r), 6)) for d, a, r in zip(trades["date"], trades["asset"], trades["return"]))


def check_m1_trades_match(runs: list[ModeRun], split: str) -> None:
    """The M1 ("all trades") (date, asset, return) set must be identical across modes for
    the same (model, direction, split) — execution is label-independent (tb_pilot_plan.md
    F9), so any difference means something leaked between modes."""
    if len(runs) < 2:
        return
    reference = runs[0]
    ref_identity = _trade_identity(reference.trades)
    ref_n = reference.summary[backtest_key(reference.model, split)]["n_total_trades"]
    for run in runs[1:]:
        n = run.summary[backtest_key(run.model, split)]["n_total_trades"]
        if n != ref_n:
            raise SystemExit(
                f"n_total_trades differs: {reference.mode}={ref_n} vs {run.mode}={n} "
                f"(model={run.model}, direction={run.direction}, split={split})"
            )
        if _trade_identity(run.trades) != ref_identity:
            raise SystemExit(
                f"M1 trade set (date, asset, return) differs between {reference.mode} and {run.mode} "
                f"(model={run.model}, direction={run.direction}, split={split})"
            )


def check_parity(runs: list[ModeRun], split: str, output_root_base: str = DEFAULT_OUTPUT_ROOT_BASE) -> None:
    """Every hard-fail parity check this pilot depends on, run together."""
    for run in runs:
        check_cache_under_own_output_root(run, output_root_base)
    check_features_used_match(runs)
    check_m1_trades_match(runs, split)


# ---------------------------------------------------------------------------
# Section (c): paired moving-block bootstrap for delta-Sharpe
# ---------------------------------------------------------------------------


def daily_returns(equity: pd.Series) -> np.ndarray:
    """Simple pct-change returns, dropping the leading NaN."""
    return equity.pct_change().dropna().to_numpy()


def build_combined_equity(up_trades: pd.DataFrame, down_trades: pd.DataFrame, horizon: int, m2_only: bool = True) -> pd.Series:
    """Combine UP+DOWN trades into one equity curve over their shared timeline
    (tb_pilot_plan.md item 11: "rebuilds the combined UP+DOWN portfolio with
    engine._build_spread_equity")."""
    trades = pd.concat([up_trades, down_trades], ignore_index=True)
    if m2_only:
        trades = trades[trades["m2_approved"]]
    timeline = pd.DatetimeIndex(sorted(pd.concat([up_trades["date"], down_trades["date"]]).unique()))
    equity, _ = _build_spread_equity(trades, timeline, horizon)
    return equity


def moving_block_bootstrap_delta_sharpe(
    returns_k: np.ndarray,
    returns_ref: np.ndarray,
    *,
    ann: float,
    block_len: int,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Paired moving-block (circular) bootstrap CI for Sharpe(k) - Sharpe(ref).

    `returns_k` and `returns_ref` must be the same length and date-aligned (the same
    bar index means the same date in both arms), so each resampled block preserves
    whatever correlation the two arms' returns actually have on a given day.
    `block_len` is in [7, 14] per tb_pilot_plan.md item 11c.
    """
    returns_k = np.asarray(returns_k, dtype=float)
    returns_ref = np.asarray(returns_ref, dtype=float)
    if len(returns_k) != len(returns_ref):
        raise ValueError(f"paired bootstrap needs equal-length, date-aligned series, got {len(returns_k)} vs {len(returns_ref)}")
    n = len(returns_k)
    if n == 0:
        raise ValueError("no returns to bootstrap")

    rng = rng if rng is not None else np.random.default_rng()
    point_delta = _calc_sharpe(returns_k, ann) - _calc_sharpe(returns_ref, ann)

    n_blocks = -(-n // block_len)  # ceil(n / block_len)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) % n for s in starts])[:n]
        deltas[b] = _calc_sharpe(returns_k[idx], ann) - _calc_sharpe(returns_ref[idx], ann)

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_sharpe": float(point_delta),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_boot": n_boot,
        "block_len": block_len,
    }


# ---------------------------------------------------------------------------
# Section (a): target characterization from the manifest (train split only, per
# the headline convention Data_MLA/meta_label_tb.py's generator CLI documents —
# see the docstring note on summarize() for what "train split only" does and
# does not restrict)
# ---------------------------------------------------------------------------


def target_characterization(manifest: dict, mode: str) -> dict:
    """Equal-(asset, direction)-weighted summary for one mode, from the manifest.

    `summarize()` (Data_MLA/meta_label_tb.py) computes positive rates, flip rate and
    exit mix over every eligible row of a (symbol, direction) file — the manifest does
    not separately expose a train-only positive rate, only `n_eligible_train` as a
    count. These are therefore all-eligible-split rates, not train-restricted; say so
    in any write-up rather than assuming otherwise.
    """
    entries = [e for e in manifest["summary"][mode] if e.get("n_eligible", 0) > 0]
    if not entries:
        raise SystemExit(f"mode {mode!r} has no eligible entries in this manifest")

    out: dict[str, object] = {
        "n_assets_x_directions": len(entries),
        "n_eligible_total": sum(e["n_eligible"] for e in entries),
        "tp_positive_rate": float(np.mean([e["tp_positive_rate"] for e in entries])),
        "tb_positive_rate": float(np.mean([e["tb_positive_rate"] for e in entries])),
        "flip_rate": float(np.mean([e["flip_rate"] for e in entries])),
    }
    exit_entries = [e["exit_mix"] for e in entries if "exit_mix" in e]
    if exit_entries:
        out["exit_mix"] = {key: float(np.mean([e[key] for e in exit_entries])) for key in ("profit", "stop", "expiry")}
        ambiguous = [e["ambiguous_share"] for e in entries if "ambiguous_share" in e]
        if ambiguous:
            out["ambiguous_share"] = float(np.mean(ambiguous))
    return out


# ---------------------------------------------------------------------------
# Section (b): economics per arm
# ---------------------------------------------------------------------------


def economics_table(runs: list[ModeRun], split: str) -> pd.DataFrame:
    """Sharpe/MDD/total-return/coverage/trade-count/fees per (mode, model, direction),
    for M2 (selective) and M1 (all trades)."""
    rows = []
    for run in runs:
        bt = run.summary[backtest_key(run.model, split)]
        n_days = run.trades["date"].nunique()
        rows.append(
            {
                "mode": run.mode,
                "model": run.model,
                "direction": run.direction,
                "n_total_trades": bt["n_total_trades"],
                "n_m2_trades": bt["n_m2_trades"],
                "coverage_pct": bt["execution_rate"],
                "trades_per_day": bt["n_m2_trades"] / n_days if n_days else 0.0,
                "fee": bt["fee"],
                "m2_sharpe": bt["m2_sharpe"],
                "m1_sharpe": bt["m1_sharpe"],
                "m2_total_return_pct": bt["m2_total_return"],
                "m1_total_return_pct": bt["m1_total_return"],
                "m2_max_drawdown_pct": bt["m2_max_drawdown"],
                "m1_max_drawdown_pct": bt["m1_max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section (d): equal-weight per-asset table — never a portfolio Sharpe
# ---------------------------------------------------------------------------


def per_asset_table(run: ModeRun) -> pd.DataFrame:
    """Equal-weight per-asset average M2-approved return (tb_pilot_plan.md item 11d:
    "never presented as a portfolio Sharpe")."""
    m2 = run.trades[run.trades["m2_approved"]]
    return (
        m2.groupby("asset")["return"]
        .agg(avg_return="mean", n_trades="count")
        .reset_index()
        .sort_values("asset")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Sealing: confirmation mode needs --unseal + a frozen_choice.json written beforehand
# ---------------------------------------------------------------------------


def freeze_choice(path: Path, chosen_mode: str, rule_description: str) -> dict:
    """Writes frozen_choice.json and returns its hash, to be recorded before any test
    number is read (tb_pilot_plan.md "Run order", step 5)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"chosen_mode": chosen_mode, "rule": rule_description}, indent=2, sort_keys=True))
    return {"chosen_mode": chosen_mode, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_frozen_choice(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"--unseal requires a frozen_choice.json at {path}")
    payload = json.loads(path.read_text())
    if "chosen_mode" not in payload:
        raise SystemExit(f"{path} has no 'chosen_mode'")
    return payload


def require_unseal(unseal: bool, frozen_choice_path: Path | None) -> dict:
    """Confirmation mode's seal — refuses to show test numbers without both --unseal
    and a frozen_choice.json that already exists (tb_pilot_plan.md item 11)."""
    if not unseal:
        raise SystemExit("confirmation mode requires --unseal (test is sealed by default)")
    if frozen_choice_path is None:
        raise SystemExit("--unseal requires --frozen-choice <path to frozen_choice.json>")
    return load_frozen_choice(frozen_choice_path)


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def write_selection_report(output_dir: Path, economics_val: pd.DataFrame, characterization: dict[str, dict]) -> tuple[Path, Path]:
    """Validation-only report. Test columns never enter `economics_val`, so there is
    nothing to hide here beyond calling this with the val split in the first place."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "selection.csv"
    economics_val.to_csv(csv_path, index=False)

    lines = ["# TB-target pilot — selection (validation only; test is sealed)", ""]
    lines.append("## Target characterization (manifest, all-eligible rates)")
    for mode, stats in characterization.items():
        lines.append(f"- **{mode}**: {json.dumps(stats)}")
    lines.append("")
    lines.append("## Validation economics")
    lines.append(economics_val.to_markdown(index=False) or "")
    md_path = output_dir / "selection.md"
    md_path.write_text("\n".join(lines))
    return csv_path, md_path


def write_confirmation_report(
    output_dir: Path, frozen: dict, economics_test: pd.DataFrame, delta_sharpe: dict[str, dict[str, dict]]
) -> tuple[Path, Path]:
    """`delta_sharpe` is `{mode: {model: bootstrap_result}}` — one combined UP+DOWN
    portfolio per (mode, model), vs. that same model's tb_ref portfolio."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "confirmation.csv"
    economics_test.to_csv(csv_path, index=False)

    lines = [
        "# TB-target pilot — confirmation (test, unsealed)",
        "",
        f"Frozen choice: **{frozen['chosen_mode']}** ({frozen.get('rule', '')})",
        "",
        "## Test economics",
        economics_test.to_markdown(index=False) or "",
        "",
        "## Delta-Sharpe vs tb_ref (paired moving-block bootstrap, combined UP+DOWN per model)",
    ]
    for mode, by_model in delta_sharpe.items():
        for model, stats in by_model.items():
            lines.append(
                f"- **{mode} / {model}**: {stats['delta_sharpe']:+.3f} [{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}]"
            )
    md_path = output_dir / "confirmation.md"
    md_path.write_text("\n".join(lines))
    return csv_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare TB-pilot modes (tb_pilot_plan.md Phase 5).")
    parser.add_argument("--phase", choices=["selection", "confirmation"], required=True)
    parser.add_argument("--config", type=Path, required=True, help="base config.yaml (for thres_mode/m1)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--directions", nargs="+", default=["up", "down"])
    parser.add_argument("--granularity", default="1d")
    parser.add_argument("--file-prefix", default="10_backtest")
    parser.add_argument("--output-root-base", default=DEFAULT_OUTPUT_ROOT_BASE)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_ROOT_BASE) / "comparison")
    parser.add_argument("--unseal", action="store_true")
    parser.add_argument("--frozen-choice", type=Path, default=None)
    parser.add_argument("--block-len", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> None:
    import yaml

    args = _build_parser().parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    split = "val" if args.phase == "selection" else "test"
    file_prefix = f"{args.file_prefix}_val" if split == "val" else f"{args.file_prefix}_all"

    if args.phase == "confirmation":
        frozen = require_unseal(args.unseal, args.frozen_choice)

    characterization = {mode: target_characterization(manifest, mode) for mode in args.modes}

    all_runs: list[ModeRun] = []
    runs_by_mode_model_direction: dict[tuple[str, str, str], ModeRun] = {}
    for model in args.models:
        for direction in args.directions:
            runs = [
                load_mode_run(cfg, mode, model, direction, args.granularity, file_prefix, args.output_root_base)
                for mode in args.modes
            ]
            check_parity(runs, split, args.output_root_base)
            all_runs.extend(runs)
            for run in runs:
                runs_by_mode_model_direction[(run.mode, run.model, run.direction)] = run

    economics = economics_table(all_runs, split)

    if args.phase == "selection":
        write_selection_report(args.output_dir, economics, characterization)
    else:
        horizon = int(cfg.get("data", {}).get("load", {}).get("forecast_horizon", 7))
        ann = _annualization_factor(args.granularity)

        ref_returns_by_model = {
            model: daily_returns(
                build_combined_equity(
                    runs_by_mode_model_direction[("tb_ref", model, "up")].trades,
                    runs_by_mode_model_direction[("tb_ref", model, "down")].trades,
                    horizon,
                )
            )
            for model in args.models
        }

        delta_sharpe: dict[str, dict[str, dict]] = {}
        for mode in args.modes:
            if mode == "tb_ref":
                continue
            delta_sharpe[mode] = {}
            for model in args.models:
                k_returns = daily_returns(
                    build_combined_equity(
                        runs_by_mode_model_direction[(mode, model, "up")].trades,
                        runs_by_mode_model_direction[(mode, model, "down")].trades,
                        horizon,
                    )
                )
                ref_returns = ref_returns_by_model[model]
                n = min(len(k_returns), len(ref_returns))
                delta_sharpe[mode][model] = moving_block_bootstrap_delta_sharpe(
                    k_returns[:n], ref_returns[:n], ann=ann, block_len=args.block_len, n_boot=args.n_boot
                )

        write_confirmation_report(args.output_dir, frozen, economics, delta_sharpe)


if __name__ == "__main__":
    main()
