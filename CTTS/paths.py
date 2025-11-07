# CTTS/paths.py
from pathlib import Path
from typing import Optional

# ┏━━━━━━━━━━ Base Directories ━━━━━━━━━━┓
base   = Path(__file__).resolve().parent                   # …/CTTS
DATA_DIR  = base / "Data"                                  # …/CTTS/Data
OUTPUT_DIR = base / "Output"                               # keep for later

# ┏━━━━━━━━━━ Public Helper ━━━━━━━━━━┓
def dataset_path(provider: str,
                 market: str,
                 symbol: str,
                 split: str = "merge",
                 granularity: Optional[str] = None,
                 meta_label_mode: Optional[str] = None) -> Path:
    """
    Build absolute path to a CSV file.

    Parameters
    ----------
    provider : "Bolt", "Chronos", "Tirex", …   (folder under CTTS/Data/)
    market   : "Equities", "Crypto", …
    symbol   : e.g. "SPY", "BTC"
    split    : "merge" (default) or "up" / "down"
    granularity : e.g. "1d", "4h", "1min" (optional). If provided, files are resolved
                  under .../<symbol>/<granularity>/.

    Returns
    -------
    pathlib.Path
        .../CTTS/Data/<provider>/<market>/<symbol>/<granularity>/<symbol>_<split>.csv
    """
    root = DATA_DIR / provider / market / symbol
    if granularity:
        root = root / granularity
    if meta_label_mode:
        meta_label_mode = meta_label_mode.lower()
        suffix = "og" if meta_label_mode == "original" else meta_label_mode
        root = root.with_name(f"{root.name}_{suffix}")
    path = root / f"{symbol}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"[dataset_path] {path} does not exist")
    return path

# ┏━━━━━━━━━━ Convenience wrappers (optional) ━━━━━━━━━━┓
def bolt_path   (*args, **kw): return dataset_path("Bolt",   *args, **kw)
def chronos_path(*args, **kw): return dataset_path("Chronos",*args, **kw)
def tirex_path  (*args, **kw): return dataset_path("Tirex",  *args, **kw)
