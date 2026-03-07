"""
Shared utility helpers used across the pipeline.
"""

import io
import os
import sys
import pickle
import time
from pathlib import Path

import pandas as pd


def save_pickle(obj, path: Path, name: str = ""):
    """Persist any Python object to disk via pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    label = f" ({name})" if name else ""
    _print(f"  [SAVE]{label} -> {path}  [{_size_str(path)}]")


def load_pickle(path: Path):
    """Load a pickled object."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_parquet(df: pd.DataFrame, path: Path, name: str = ""):
    """Save DataFrame as parquet with gzip compression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="gzip")
    label = f" ({name})" if name else ""
    _print(f"  [SAVE]{label} -> {path}  [{_size_str(path)}]  [{len(df):,} rows x {len(df.columns)} cols]")


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file into a DataFrame."""
    df = pd.read_parquet(path)
    _print(f"  [LOAD] {path.name}  [{len(df):,} rows x {len(df.columns)} cols]")
    return df


def log_section(title: str):
    """Print a prominent section header."""
    width = 70
    _print()
    _print("=" * width)
    _print(f"  {title}")
    _print("=" * width)


def log_step(msg: str):
    """Print a step-level log message."""
    _print(f"  > {msg}")


class Timer:
    """Simple context-manager timer."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        if elapsed < 60:
            _print(f"    [TIME] {elapsed:.1f}s")
        else:
            _print(f"    [TIME] {elapsed / 60:.1f}min")


def ensure_dir(path: Path):
    """Create directory (and parents) if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# -- private helpers -----------------------------------------------
def _print(*args, **kwargs):
    """Print that handles Windows cp1252 gracefully."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()


def _size_str(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
