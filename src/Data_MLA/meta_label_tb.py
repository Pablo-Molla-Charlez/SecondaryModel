"""TB-target label generator for the SecondaryModel M2 pilot (report/triple_barrier/tb_pilot_plan.md).

Reads the real TP forecast CSVs (Data_MLA/Kronos/Crypto/TP/horizon_7/1d_tp), validates the
contract they are built under (tb_pilot_plan.md F2), and writes parallel per-k trees under
Data_MLA/Kronos/Crypto/TB/horizon_7 whose `lab`/`meta_label` columns come from the
triple-barrier rule in `triple_barrier.py` instead of the fixed J+7 horizon. Mirrors
`meta_label_conversion.py`'s location and CLI style; unlike it, every transform here is a
pure function returning a new DataFrame.

CLI:
    python -m Data_MLA.meta_label_tb --src .../TP/horizon_7/1d_tp --dst .../TB/horizon_7 \
        --ks 0.5 1.0 1.5 2.0 --ref --h 7 --fee 0.002 --span 100 --warmup 120 --touch hl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from Data_MLA.triple_barrier import EWMA_SPAN, FEE, LOWER, UPPER, VERTICAL, ewma_vol, triple_barrier

REQUIRED_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "eos_price",
    "prediction",
    "ground_truth",
    "pred",
    "lab",
    "meta_label",
)
PAIR_SHARED_COLUMNS = ("date", "open", "high", "low", "close", "volume", "prediction")


def validate_tp_contract(df: pd.DataFrame, side: int, h: int, fee: float) -> None:
    """Hard-fail unless `df` matches the TP CSV contract (tb_pilot_plan.md F2).

    `side` is the file's direction: +1 for an "up" file, -1 for "down".
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}")

    dates = pd.to_datetime(df["date"])
    gaps = dates.diff().iloc[1:]
    if not (gaps == pd.Timedelta(days=1)).all():
        bad_rows = gaps.index[gaps != pd.Timedelta(days=1)].tolist()
        raise ValueError(f"date gap(s) at row(s) {bad_rows}")

    forecast = df["pred"].notna().to_numpy()
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    eos = df["eos_price"].to_numpy(dtype=float)[forecast]
    if not np.array_equal(eos, close[forecast]):
        raise ValueError("eos_price != close on a forecast row")

    pos = np.flatnonzero(forecast)
    checkable = pos[pos + h < n]
    gt_checkable = df["ground_truth"].to_numpy(dtype=float)[checkable]
    if not np.array_equal(gt_checkable, close[checkable + h]):
        raise ValueError("ground_truth != close[pos + h] on a row with a future bar")

    gt = df["ground_truth"].to_numpy(dtype=float)[forecast]
    ret = side * (gt / close[forecast] - 1.0)
    expected_lab = (ret > fee).astype(float)
    actual_lab = df["lab"].to_numpy(dtype=float)[forecast]
    if not np.array_equal(expected_lab, actual_lab):
        raise ValueError("lab != 1[s * ret_h > fee] on a forecast row")

    pred = df["pred"].to_numpy(dtype=float)[forecast]
    meta = df["meta_label"][forecast].reset_index(drop=True)
    is_pred1 = pred == 1
    meta_pred1 = meta[is_pred1].astype(float).to_numpy()
    if not np.array_equal(meta_pred1, actual_lab[is_pred1]):
        raise ValueError("meta_label != lab where pred == 1")
    if not (meta[~is_pred1].astype(str) == "False").all():
        raise ValueError("meta_label != 'False' where pred == 0")

    if df["meta_label"][~forecast].notna().any():
        raise ValueError("meta_label is set on a non-forecast row")


def validate_tp_pair(up: pd.DataFrame, down: pd.DataFrame) -> None:
    """Hard-fail unless the UP and DOWN files share OHLCV/prediction and never both fire."""
    if not up[list(PAIR_SHARED_COLUMNS)].equals(down[list(PAIR_SHARED_COLUMNS)]):
        raise ValueError("UP and DOWN files disagree on OHLCV or prediction")
    both_fire = (up["pred"].to_numpy() == 1) & (down["pred"].to_numpy() == 1)
    if bool(np.any(both_fire)):
        raise ValueError("UP and DOWN both have pred == 1 on the same row")


def eligibility_mask(df: pd.DataFrame, h: int, warmup: int) -> np.ndarray:
    """Forecast row, far enough past `warmup`, with `h` candles still ahead of it."""
    n = len(df)
    pos = np.arange(n)
    forecast = df["pred"].notna().to_numpy()
    return forecast & (pos >= warmup) & (pos + h < n)


@dataclass(frozen=True)
class TbResult:
    """Per-position triple-barrier outcome, aligned to the `pos` array passed in."""

    lab: np.ndarray  # bool
    outcome: np.ndarray  # UPPER / VERTICAL / LOWER
    bars: np.ndarray  # int, 1..h
    ret: np.ndarray  # float, side-adjusted
    ambiguous: np.ndarray  # bool
    sigma: np.ndarray  # float, EWMA vol at each position


def tb_outcomes(
    df: pd.DataFrame,
    pos: np.ndarray,
    side: int,
    k: float,
    h: int,
    fee: float,
    span: int = EWMA_SPAN,
    touch: str = "hl",
) -> TbResult:
    """Triple-barrier outcome at each position in `pos` (mirrors brief_diagnostics.measure)."""
    sigma_all = ewma_vol(df["close"], span=span).to_numpy()
    sigma = sigma_all[pos]
    width = k * sigma * np.sqrt(h)
    side_arr = np.full(len(pos), float(side))
    touches = triple_barrier(df[["close", "high", "low"]], pos, width, h, side=side_arr, mode=touch)
    lab = touches.ret > fee
    return TbResult(
        lab=lab, outcome=touches.outcome, bars=touches.bars, ret=touches.ret, ambiguous=touches.ambiguous, sigma=sigma
    )


def relabel(
    df: pd.DataFrame,
    side: int,
    k: float | None,
    h: int = 7,
    fee: float = FEE,
    span: int = EWMA_SPAN,
    warmup: int = 120,
    touch: str = "hl",
    keep_lab: bool = False,
) -> pd.DataFrame:
    """Return a copy of `df` with `lab`/`meta_label` driven by the triple-barrier rule.

    `k=None` is the reference arm: it keeps the source TP labels, just restricted to the
    same eligibility mask as the TB arms (forecast row, `warmup` candles of history, `h`
    candles ahead), so every arm scores the exact same event set.

    `keep_lab=True` leaves the `lab` column at its source J+7 value — an ablation on the
    m1 precision floor (tb_pilot_plan.md F3), not on the training target: `meta_label` is
    always derived from the TB outcome for a TB arm (k is not None), independent of this flag.
    """
    mask = eligibility_mask(df, h, warmup)
    pred = df["pred"].to_numpy(dtype=float)

    if k is None:
        training_lab = df["lab"].to_numpy(dtype=float)
        floor_lab = training_lab
        tb_meta: dict[str, object] = {}
    else:
        pos = np.flatnonzero(mask)
        tb = tb_outcomes(df, pos, side, k, h, fee, span=span, touch=touch)
        training_lab = np.full(len(df), np.nan)
        training_lab[pos] = tb.lab.astype(float)
        floor_lab = df["lab"].to_numpy(dtype=float) if keep_lab else training_lab
        tb_meta = {"outcome": tb.outcome, "bars": tb.bars, "ambiguous": tb.ambiguous, "sigma": tb.sigma, "pos": pos}

    out = df.copy()
    out["lab"] = np.where(mask, floor_lab, np.nan)

    # Every row outside the mask reads "False", not NaN — a non-forecast row and a forecast
    # row the mask excluded (warmup/lookahead) must look identical, so the two arms score
    # the exact same event set (tb_pilot_plan.md Phase 1, item 2).
    meta = np.full(len(df), False, dtype=object)
    pred1 = mask & (pred == 1)
    meta[pred1] = training_lab[pred1].astype(int)
    out["meta_label"] = meta

    out.attrs["tb"] = tb_meta
    out.attrs["mask"] = mask
    return out


def summarize(
    source: pd.DataFrame,
    relabeled: pd.DataFrame,
    mask: np.ndarray,
    train_end: pd.Timestamp | None = None,
) -> dict:
    """Headline counts for one (asset, direction) file (tb_pilot_plan.md Phase 1, item 2).

    `train_end`, when given, additionally counts eligible events up to that date — the run
    order (tb_pilot_plan.md) looks at train-only counts before any validation/test number.
    """
    n_eligible = int(mask.sum())
    if n_eligible == 0:
        return {"n_eligible": 0}

    tp_lab = source["lab"].to_numpy(dtype=float)[mask]
    tb_lab = relabeled["lab"].to_numpy(dtype=float)[mask]
    out: dict[str, object] = {
        "n_eligible": n_eligible,
        "tp_positive_rate": float(np.mean(tp_lab)),
        "tb_positive_rate": float(np.mean(tb_lab)),
        "flip_rate": float(np.mean(tp_lab != tb_lab)),
    }
    tb_meta = relabeled.attrs.get("tb")
    if tb_meta:
        outcome = tb_meta["outcome"]
        out["exit_mix"] = {
            "profit": float(np.mean(outcome == UPPER)),
            "stop": float(np.mean(outcome == LOWER)),
            "expiry": float(np.mean(outcome == VERTICAL)),
        }
        out["ambiguous_share"] = float(np.mean(tb_meta["ambiguous"]))
    if train_end is not None:
        in_train = (pd.to_datetime(source["date"]) <= train_end).to_numpy()
        out["n_eligible_train"] = int(np.sum(mask & in_train))
    return out


def mode_name(k: float | None) -> str:
    """Directory/meta_label_mode name for a k-arm: "tb_ref" (k=None) or "tb_kNNN"."""
    return "tb_ref" if k is None else f"tb_k{round(k * 100):03d}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TB-target label trees from the TP source CSVs.")
    parser.add_argument("--src", type=Path, required=True, help="e.g. .../TP/horizon_7/1d_tp")
    parser.add_argument("--dst", type=Path, required=True, help="e.g. .../TB/horizon_7")
    parser.add_argument("--ks", type=float, nargs="*", default=[], help="k multiples of sigma*sqrt(h)")
    parser.add_argument("--ref", action="store_true", help="also write the tb_ref arm (k=None)")
    parser.add_argument("--h", type=int, default=7)
    parser.add_argument("--fee", type=float, default=FEE)
    parser.add_argument("--span", type=int, default=EWMA_SPAN)
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument("--touch", choices=["close", "hl"], default="hl")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--keep-lab", action="store_true")
    return parser


def _train_end_from_config(config_path: Path | None) -> pd.Timestamp | None:
    if config_path is None:
        return None
    config = yaml.safe_load(config_path.read_text())
    train_end = config.get("data", {}).get("split", {}).get("train_end")
    if not train_end:
        return None
    ts = pd.Timestamp(train_end)
    return ts if isinstance(ts, pd.Timestamp) else None


SIDES = {"up": 1, "down": -1}


def _discover_symbols(src: Path, requested: list[str] | None) -> list[str]:
    """All symbols with an `*_up.csv` in `src`, hard-failing on an empty or mistyped request."""
    available = sorted(p.stem.removesuffix("_up") for p in src.glob("*_up.csv"))
    if not available:
        raise SystemExit(f"no *_up.csv files found in {src}")
    if not requested:
        return available
    missing = [s for s in requested if s not in available]
    if missing:
        raise SystemExit(f"--symbols not found under {src}: {missing}")
    return [s for s in available if s in requested]


def _load_and_validate_symbol(src: Path, symbol: str, h: int, fee: float) -> dict[str, pd.DataFrame]:
    """Read and contract-check both directions of one symbol (tb_pilot_plan.md F2)."""
    source_by_dir = {}
    for direction, side in SIDES.items():
        df = pd.read_csv(src / f"{symbol}_{direction}.csv", parse_dates=["date"])
        validate_tp_contract(df, side=side, h=h, fee=fee)
        source_by_dir[direction] = df
    validate_tp_pair(source_by_dir["up"], source_by_dir["down"])
    return source_by_dir


def _write_mode_for_symbol(
    symbol: str,
    source_by_dir: dict[str, pd.DataFrame],
    k: float | None,
    out_dir: Path,
    args: argparse.Namespace,
    train_end: pd.Timestamp | None,
    files: dict[str, str],
) -> list[dict]:
    """Relabel, write and summarize one symbol's two directions for one k-mode."""
    summaries = []
    for direction, side in SIDES.items():
        df = source_by_dir[direction]
        relabeled = relabel(
            df,
            side=side,
            k=k,
            h=args.h,
            fee=args.fee,
            span=args.span,
            warmup=args.warmup,
            touch=args.touch,
            keep_lab=args.keep_lab,
        )
        mask = relabeled.attrs["mask"]

        dst_path = out_dir / f"{symbol}_{direction}.csv"
        relabeled.to_csv(dst_path, index=False)
        files[str(dst_path)] = _sha256(dst_path)

        summaries.append(
            {
                "symbol": symbol,
                "direction": direction,
                **summarize(df, relabeled, mask, train_end=train_end),
            }
        )
    return summaries


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    gran = args.src.name.split("_")[0]
    train_end = _train_end_from_config(args.config)

    modes: list[float | None] = list(args.ks)
    if args.ref:
        modes = [None, *modes]
    if not modes:
        raise SystemExit("pass --ref and/or --ks")

    symbols = _discover_symbols(args.src, args.symbols)

    manifest: dict[str, object] = {
        "parameters": {
            "h": args.h,
            "fee": args.fee,
            "span": args.span,
            "warmup": args.warmup,
            "touch": args.touch,
            "keep_lab": args.keep_lab,
            "src": str(args.src),
        },
        "files": {},
        "summary": {name: [] for name in (mode_name(k) for k in modes)},
    }
    files = manifest["files"]
    summary = manifest["summary"]
    assert isinstance(files, dict) and isinstance(summary, dict)

    out_dirs = {k: args.dst / f"{gran}_{mode_name(k)}" for k in modes}
    for out_dir in out_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        source_by_dir = _load_and_validate_symbol(args.src, symbol, args.h, args.fee)
        for k in modes:
            summary[mode_name(k)].extend(_write_mode_for_symbol(symbol, source_by_dir, k, out_dirs[k], args, train_end, files))

    manifest_path = args.dst / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
