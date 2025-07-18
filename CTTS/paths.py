# CTTS/paths.py
from pathlib import Path

# ─── base directories ────────────────────────────────────────────────────────
base   = Path(__file__).resolve().parent                   # …/CTTS
DATA_DIR  = base / "Data"                                  # …/CTTS/Data
OUTPUT_DIR = base / "Output"                               # keep for later

# ─── public helper ───────────────────────────────────────────────────────────
def dataset_path(provider: str,
                 market: str,
                 symbol: str,
                 split: str = "merge") -> Path:
    """
    Build absolute path to a CSV file.

    Parameters
    ----------
    provider : "Bolt", "Chronos", "Tirex", …   (folder under CTTS/Data/)
    market   : "Equities", "Crypto", …
    symbol   : e.g. "SPY", "BTC"
    split    : "merge" (default) or "up" / "down"

    Returns
    -------
    pathlib.Path
        .../CTTS/Data/<provider>/<market>/<symbol>/<symbol>_<split>.csv
    """
    path = DATA_DIR / provider / market / symbol / f"{symbol}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"[dataset_path] {path} does not exist")
    return path

# ─── convenience wrappers (optional) ─────────────────────────────────────────
def bolt_path   (*args, **kw): return dataset_path("Bolt",   *args, **kw)
def chronos_path(*args, **kw): return dataset_path("Chronos",*args, **kw)
def tirex_path  (*args, **kw): return dataset_path("Tirex",  *args, **kw)
