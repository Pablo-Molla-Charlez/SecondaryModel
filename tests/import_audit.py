"""Import every module under src/ in a fresh interpreter; report the ones that fail.

Usage: uv run python tests/import_audit.py [--json out.json]
Each module is imported in its own subprocess (scripts may run code at import time,
and one module's import side effects must not hide another's failure).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
SKIP_DIRS = {"Output", "__pycache__", "Images"}


def modules() -> list[str]:
    out = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        if SKIP_DIRS & set(rel.parts):
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            out.append(".".join(parts))
    return out


def try_import(mod: str, timeout: float = 180.0) -> tuple[str, str | None]:
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    code = f"import importlib, sys; sys.argv = ['x']; importlib.import_module({mod!r})"
    try:
        r = subprocess.run([sys.executable, "-c", code], cwd=SRC, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return mod, "TIMEOUT"
    if r.returncode == 0:
        return mod, None
    lines = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
    return mod, lines[-1] if lines else f"exit {r.returncode}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    mods = modules()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = dict(ex.map(try_import, mods))
    failed = {m: e for m, e in results.items() if e}
    for m, e in sorted(failed.items()):
        print(f"FAIL {m}: {e}")
    print(f"{len(mods) - len(failed)}/{len(mods)} modules import")
    if args.json:
        Path(args.json).write_text(json.dumps(failed, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
