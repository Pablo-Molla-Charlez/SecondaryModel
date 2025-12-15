"""Real-time inference pipeline combining Kronos (M1) and CTTS (M2)."""
from __future__ import annotations

import os
import warnings

# ┏━━━━━━━━━━ Ignoring Warning Message ━━━━━━━━━━┓
os.environ.setdefault("TORCH_CUDA_NVML_DISABLE_WARNING", "1")
warnings.filterwarnings("ignore", message="Can't initialize NVML")


import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
import torch

# ┏━━━━━━━━━━ Location of Current Repository Root ━━━━━━━━━━┓
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

# ┏━━━━━━━━━━ System Paths ━━━━━━━━━━┓
from paths import add_sys_path
add_sys_path(BASE_DIR / "chronos_meta_labeling_v2" / "src" / "m1" / "kronos", prepend=True)
add_sys_path(BASE_DIR / "chronos_meta_labeling_v2" / "src", prepend=True)
add_sys_path(SCRIPT_DIR)
add_sys_path(BASE_DIR / "Meta-Labeling-CTTS" / "CTTS")
add_sys_path(BASE_DIR)

# ┏━━━━━━━━━━ Configuration Settings from M1 & M2 ━━━━━━━━━━┓
from config import CTTSSettings, KronosSettings, load_ctts_settings, load_kronos_settings
from data_preprocessing import prepare_dataset

# ┏━━━━━━━━━━ Location of M2 model.py file ━━━━━━━━━━┓
_ctts_model_spec_path = BASE_DIR / "Meta-Labeling-CTTS" / "CTTS" / "model.py"
if not _ctts_model_spec_path.exists():
    raise FileNotFoundError(f"Expected CTTS model file at {_ctts_model_spec_path}")

# ┏━━━━━━━━━━ Dynamic Loading of M2 (CTTS) Model ━━━━━━━━━━┓
import importlib.util
_ctts_spec = importlib.util.spec_from_file_location("ctts_model_module", _ctts_model_spec_path)
if _ctts_spec is None or _ctts_spec.loader is None:
    raise ImportError(f"Unable to load CTTS model from {_ctts_model_spec_path}")
_ctts_module = importlib.util.module_from_spec(_ctts_spec)
_ctts_spec.loader.exec_module(_ctts_module)
CTTSModel = _ctts_module.CTTSModel

# ┏━━━━━━━━━━ Imports for M1 (Kronos) Model ━━━━━━━━━━┓
from chronos_meta_labeling_v2.src.m1.kronos.model import (Kronos, KronosPredictor, KronosTokenizer)
from chronos_meta_labeling_v2.src.m1.kronos.utils import (_ensure_kronos_features, _extract_timestamps, resolve_model_paths, select_device)

# ┏━━━━━━━━━━ Predictor Parameters for M1 (Kronos) Model ━━━━━━━━━━┓
KRONOS_TEMPERATURE = 0.9
KRONOS_TOP_P = 0.9
KRONOS_SAMPLE_COUNT = 30
DEFAULT_KRONOS_MAX_CONTEXT = 512


@dataclass
class CTTSModelBundle:
    model: CTTSModel
    column_features: Sequence[str]
    context_features: Sequence[str]
    threshold: float


class MetaDeployment:
    """Orchestrates Kronos predictions followed by CTTS validation."""
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        # ┏━━━━━━━━━━ 1. Root Paths for M1 & M2 ━━━━━━━━━━┓
        self.base_dir = Path(base_dir) if base_dir else BASE_DIR
        self.chronos_src = self.base_dir / "chronos_meta_labeling_v2" / "src"
        self.ctts_root = self.base_dir / "Meta-Labeling-CTTS" / "CTTS"

        # ┏━━━━━━━━━━ 2. Extracting Configuration for M1 & M2 ━━━━━━━━━━┓
        self.kronos_settings: KronosSettings = load_kronos_settings(self.base_dir)
        self.ctts_settings: CTTSSettings = load_ctts_settings(self.base_dir)

        # ┏━━━━━━━━━━ 3. Checking seq_len match between M1 & M2 ━━━━━━━━━━┓
        self.sequence_length = self.ctts_settings.sequence_length
        if self.kronos_settings.seq_len != self.sequence_length:
            raise ValueError("Sequence length mismatch between Kronos and CTTS "
                             f"({self.kronos_settings.seq_len} vs {self.sequence_length}).")

        # ┏━━━━━━━━━━ 4. Checking symbol match between M1 & M2 ━━━━━━━━━━┓
        kronos_asset = self.kronos_settings.predict_for.upper()
        ctts_asset = self.ctts_settings.dataset_symbol.upper()
        self.provider = self.ctts_settings.dataset_source
        self.asset_symbol = self.ctts_settings.dataset_symbol

        # ┏━━━━━━━━━━ Helper Function ━━━━━━━━━━┓
        def _base_symbol(symbol: str) -> str:
            for suffix in ("USDT", "USDC", "USD"):
                if symbol.endswith(suffix):
                    return symbol[: -len(suffix)]
            return symbol

        if not (kronos_asset == ctts_asset or _base_symbol(kronos_asset) == _base_symbol(ctts_asset)):
            raise ValueError(
                "Asset mismatch between M1 and M2 configurations: "
                f"M1 predict_for = {kronos_asset}, M2 dataset.symbol = {ctts_asset}. "
                "Please align both models on the same asset before running the deployment pipeline.")

        # ┏━━━━━━━━━━ 5. Checking symbol match between M1 & M2 ━━━━━━━━━━┓
        self.kronos_granularity_slug = self.kronos_settings.granularity_slug
        self.ctts_granularity_value = self.ctts_settings.granularity_usual
        self.ctts_granularity_slug = self.ctts_settings.granularity_usual_slug
        if self.kronos_granularity_slug != self.ctts_granularity_slug:
            raise ValueError("Granularity mismatch between Kronos and CTTS configurations: "
                             f"Kronos granularity={self.kronos_settings.granularity} "
                             f"vs CTTS granularity={self.ctts_granularity_value}. "
                             "Please ensure both models target the same timeframe.")

        # ┏━━━━━━━━━━ 6. M1 (Kronos) Predictor Instantiation & Parameters ━━━━━━━━━━┓
        tokenizer_id, model_id = resolve_model_paths()
        kronos_device = select_device()
        max_context = max(self.kronos_settings.seq_len, DEFAULT_KRONOS_MAX_CONTEXT)
        self._kronos_predictor = KronosPredictor(Kronos.from_pretrained(model_id),  
                                                 KronosTokenizer.from_pretrained(tokenizer_id),
                                                 device      = kronos_device,
                                                 max_context = max_context,)


        # ┏━━━━━━━━━━ 7. Cached Variable to store M2's Configurations ━━━━━━━━━━┓
        self.ctts_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._ctts_models: Dict[str, CTTSModelBundle] = {}
        self._ctts_checkpoints: Dict[str, Path] = {}
        self._context_features = {"UP": ["m1_up"], "DN": ["m1_dn"]}
        self._freq_offset = None
        """
        Example Content of _ctts_models:
        {"UP": CTTSModelBundle(model=<CTTSModel …>,                
                               column_features  = ["open", "high", "low", "close", "volume", "amount"],
                               context_features = ["m1_prediction", "m1_up"],
                               threshold        = 0.5),
        {"DN": CTTSModelBundle(model=<CTTSModel …>,                
                               column_features  = ["open", "high", "low", "close", "volume", "amount"],
                               context_features = ["m1_prediction", "m1_dn"],
                               threshold        = 0.5)}
        """

    def forecast_window(self, window_df: pd.DataFrame) -> Dict[str, Optional[dict]]:
        """Run M1 (Kronos) on the window, then M2 (CTTS) [If Kronos emits UP or DN]."""
        # ┏━━━━━━━━━━ Checking Window ━━━━━━━━━━┓
        window = self._window_check(window_df)
        
        # ┏━━━━━━━━━━ Extracting Frequency Helper ━━━━━━━━━━┓
        self._ensure_freq_offset(window["date"])

        # ┏━━━━━━━━━━ M1 (Kronos) Forecasting Call ━━━━━━━━━━┓
        kronos_result = self._run_kronos(window)
        
        # ┏━━━━━━━━━━ M1 (Kronos) Result Printing ━━━━━━━━━━┓
        self._print_kronos_decision(kronos_result)

        # ┏━━━━━━━━━━ Indecisiveness - Avoiding Trading ━━━━━━━━━━┓
        if kronos_result["SAME_pred"] == 1:
            print("[Kronos Conclusion] Skipping M2 evaluation.")
            print("\nSummary:")
            return {"kronos": kronos_result, "ctts": None}
        
        # ┏━━━━━━━━━━ M2 (CTTS) Forecasting Call ━━━━━━━━━━┓
        task = "UP" if kronos_result["UP_pred"] == 1 else "DN"
        ctts_result = self._run_ctts(task, window, kronos_result)
        
        # ┏━━━━━━━━━━ M2 (CTTS) Result Printing ━━━━━━━━━━┓
        self._print_ctts_decision(task, ctts_result)
        
        return {"kronos": kronos_result, "ctts": ctts_result}

    def _run_kronos(self, window: pd.DataFrame) -> dict:
        # ┏━━━━━━━━━━ Checking Columns ━━━━━━━━━━┓
        context = window.set_index("date")
        kronos_ready = _ensure_kronos_features(context).copy()
        kronos_ready.index = context.index

        # ┏━━━━━━━━━━ Creating Future Index ━━━━━━━━━━┓
        future_index = pd.date_range(start   = context.index[-1] + self._freq_offset,
                                     periods = self.kronos_settings.pred_len,
                                     freq    = self._freq_offset)

        # ┏━━━━━━━━━━ Creating Future Index ━━━━━━━━━━┓
        ctx_timestamps = _extract_timestamps(context.index)
        fut_timestamps = _extract_timestamps(future_index)

        # ┏━━━━━━━━━━ M1 (Kronos) Forecasting Call & Parameters ━━━━━━━━━━┓
        preds = self._kronos_predictor.predict(df           = kronos_ready,
                                               x_timestamp  = ctx_timestamps,
                                               y_timestamp  = fut_timestamps,
                                               pred_len     = self.kronos_settings.pred_len,
                                               T            = KRONOS_TEMPERATURE,
                                               top_p        = KRONOS_TOP_P,
                                               sample_count = KRONOS_SAMPLE_COUNT,
                                               verbose      = False)
        
        # ┏━━━━━━━━━━ Prediction, Last Closing Price & Ratio ━━━━━━━━━━┓
        horizon_close = float(preds["close"].iloc[-1])
        last_close = float(context["close"].iloc[-1])
        ratio = horizon_close / last_close if last_close else float("nan")

        # ┏━━━━━━━━━━ Label Predicted by M1 (Kronos) ━━━━━━━━━━┓
        up = int(ratio >= self.kronos_settings.threshold_up)
        dn = int(ratio <= self.kronos_settings.threshold_down)
        same = int(not up and not dn)

        return {"Prediction":        horizon_close,
                "UP_pred":           up,
                "DN_pred":           dn,
                "SAME_pred":         same,
                "Last_Close":        last_close,
                "Ratio":             ratio,
                "Predictions_frame": preds}

    def _print_kronos_decision(self, result: dict) -> None:
        direction = "UP" if result["UP_pred"] else "DN" if result["DN_pred"] else "SAME"
        print("\n[Kronos Prediction] "
              f"Predicted Close = {result['Prediction']:.4f} "
              f"(Last Seen Close= {result['Last_Close']:.4f}, Ratio = {result['Ratio']:.5f}) "
              f"Movement Direction → {direction}")

    def _run_ctts(self, task: str, window: pd.DataFrame, kronos_result: dict) -> dict:
        # ┏━━━━━━━━━━ M2 (CTTS) Model Loading ━━━━━━━━━━┓
        bundle = self._load_ctts_model(task)

        # ┏━━━━━━━━━━ Merge of M1's Prediction & Context ━━━━━━━━━━┓
        df_ctts = self._prepare_ctts_frame(window, kronos_result)

        # ┏━━━━━━━━━━ Dataset Slice Preparation ━━━━━━━━━━┓
        dataset = prepare_dataset(df_ctts,
                                  seq_len          = self.sequence_length,
                                  column_features  = bundle.column_features,
                                  context_features = bundle.context_features)

        # ┏━━━━━━━━━━ Data Slice Extraction ━━━━━━━━━━┓
        xb, _, _ = dataset[0]
        xb = xb.unsqueeze(0).to(self.ctts_device)

        # ┏━━━━━━━━━━ M2 (CTTS) Evaluation ━━━━━━━━━━┓
        model = bundle.model
        model.eval()
        with torch.no_grad():
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1].item()
            pred = logits.argmax(dim=1).item()
        # ┏━━━━━━━━━━ Label Predicted by M1 (Kronos) ━━━━━━━━━━┓
        
        return {"Probability": prob, "Prediction": pred, "Threshold": bundle.threshold}

    def _print_ctts_decision(self, task: str, result: dict) -> None:
        print(f"[M2 (CTTS-{task})] Probability Predicted = {result['Probability']:.4f} "
              f"(Threshold = {result['Threshold']:.2f}) → Prediction = {result['Prediction']}")

    def _prepare_ctts_frame(self, window: pd.DataFrame, kronos_result: dict) -> pd.DataFrame:
        # ┏━━━━━━━━━━ Window Copy ━━━━━━━━━━┓
        df = window.copy()

        # ┏━━━━━━━━━━ Zero-Instantiation ━━━━━━━━━━┓
        df["m1_prediction"] = 0.0
        df["m1_up"] = 0
        df["m1_dn"] = 0

        # ┏━━━━━━━━━━ Actual Values Replacement (M1 Prediction) ━━━━━━━━━━┓
        df.loc[df.index[-1], "m1_prediction"] = kronos_result["Prediction"]
        df.loc[df.index[-1], "m1_up"] = kronos_result["UP_pred"]
        df.loc[df.index[-1], "m1_dn"] = kronos_result["DN_pred"]

        # ┏━━━━━━━━━━ Zero-Instantiation to satisfy M2 (CTTS) format ━━━━━━━━━━┓
        df["isTP_UP"] = 0
        df["isTP_DN"] = 0

        return df.set_index("date")

    def _load_ctts_model(self, task: str) -> CTTSModelBundle:
        if task in self._ctts_models:
            return self._ctts_models[task]

        # ┏━━━━━━━━━━ Task Configuration ━━━━━━━━━━┓
        task_lower = task.lower()
        model_cfg = self.ctts_settings.model_block(task)
        column_features = self.ctts_settings.column_features(task)
        context_features = self._context_features[task]

        # ┏━━━━━━━━━━ M2 (CTTS) Parameters ━━━━━━━━━━┓
        model_kwargs = dict(cnn_embed_dim = model_cfg["cnn_embed_dim"],
                            cnn_kernel    = model_cfg["cnn_kernel"],
                            cnn_stride    = model_cfg["cnn_stride"],
                            p_pos_drop    = model_cfg["p_pos_drop"],
                            nb_features   = len(column_features),
                            trans_heads   = model_cfg["transformer"]["heads"],
                            trans_ff      = model_cfg["transformer"]["ffn_dim"] * model_cfg["cnn_embed_dim"][-1],
                            trans_layers  = model_cfg["transformer"]["layers"],
                            trans_dropout = model_cfg["transformer"]["dropout"],
                            trans_activ   = model_cfg["transformer"]["activation"],
                            mlp_hidden    = model_cfg["classifier"]["mlp_hidden"],
                            mlp_dropout   = model_cfg["classifier"]["mlp_dropout"],
                            mlp_activ     = model_cfg["classifier"]["mlp_activation"],
                            mlp_pooling   = model_cfg["classifier"]["mlp_pooling"],
                            num_classes   = 2,
                            padding       = self.ctts_settings.padding,
                            context_len   = self.sequence_length + len(context_features))

        # ┏━━━━━━━━━━ M2 (CTTS) Instantiation ━━━━━━━━━━┓
        model = CTTSModel(**model_kwargs).to(self.ctts_device)

        # ┏━━━━━━━━━━ M2 (CTTS) Checkpoint (where trained model is) ━━━━━━━━━━┓
        checkpoint = self._resolve_ctts_checkpoint(task)

        # ┏━━━━━━━━━━ Loading M2 (CTTS) Checkpoint ━━━━━━━━━━┓
        state = torch.load(checkpoint, map_location = self.ctts_device)
        model.load_state_dict(state)
        model.eval()

        # ┏━━━━━━━━━━ Extracting Threshold ━━━━━━━━━━┓
        threshold = self.ctts_settings.training_threshold(task)
        
        # ┏━━━━━━━━━━ Bundle Instantiation (fast use in cache) ━━━━━━━━━━┓
        bundle = CTTSModelBundle(model            = model,
                                 column_features  = column_features,
                                 context_features = context_features,
                                 threshold        = threshold)

        self._ctts_models[task] = bundle
        return bundle

    def _resolve_ctts_checkpoint(self, task: str) -> Path:
        """Locate the latest CTTS checkpoint matching the new run directory layout."""
        # ┏━━━━━━━━━━ Checkpoint based on Task ━━━━━━━━━━┓
        if task in self._ctts_checkpoints:
            return self._ctts_checkpoints[task]

        # ┏━━━━━━━━━━ Base Root Path ━━━━━━━━━━┓
        base = (self.ctts_root
                / self.ctts_settings.paths_root
                / "Usual"
                / self.provider
                / self.asset_symbol
                / task
                / self.ctts_granularity_slug)
        
        # ┏━━━━━━━━━━ Most Recent Trained Model ━━━━━━━━━━┓
        run_dirs = sorted(
            (path for path in base.glob("Run_*") if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )

        # ┏━━━━━━━━━━ Selecting Most Recent & Printing Choice ━━━━━━━━━━┓
        for run_dir in run_dirs:
            candidate = run_dir / f"{task}_best.pt"
            if candidate.exists():
                print(f"[CTTS] Using checkpoint for {task} at {candidate}")
                self._ctts_checkpoints[task] = candidate
                return candidate

        raise FileNotFoundError(
            "Could not locate CTTS checkpoint for deployment. "
            f"Searched under {base} for Run__*/{task}_best.pt and legacy Run/{task}_best.pt layouts."
        )
    
    # ┏━━━━━━━━━━ 1st Helper ━━━━━━━━━━┓
    def _window_check(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        data = df.copy()

        # ✅ Ensure there is a 'date' column
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
        elif isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index().rename(columns={"index": "date"})
        else:
            raise ValueError("Window must have a 'date' column or DateTimeIndex.")

        data = data.sort_index()
        if len(data) < self.sequence_length:
            raise ValueError(f"Window contains {len(data)} rows; require {self.sequence_length}.")

        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col not in data.columns:
                raise ValueError(f"Missing required price column: '{col}'.")

        return data

    # ┏━━━━━━━━━━ 2nd Helper ━━━━━━━━━━┓
    def _ensure_freq_offset(self, date_series: pd.Series) -> None:
        # ┏━━━━━━━━━━ Frequency Extraction ━━━━━━━━━━┓
        if self._freq_offset is not None:
            return

        dates = pd.to_datetime(date_series)
        freq = getattr(dates, "freq", None)
        if freq is not None:
            self._freq_offset = freq
            return

        deltas = dates.diff().dropna()
        if deltas.empty:
            raise ValueError("Unable to infer frequency from a single timestamp.")
        delta = deltas.iloc[-1]
        if delta <= pd.Timedelta(0):
            raise ValueError("Non-positive time delta detected in window timestamps.")
        self._freq_offset = pd.tseries.frequencies.to_offset(delta)


if __name__ == "__main__":
    raise SystemExit(
        "MetaDeployment is intended to be imported. Instantiate MetaDeployment() "
        "and call forecast_window(window_df) with a prepared context DataFrame."
    )