"""Dual logging setup: console + file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(output_dir: str | Path, filename: str = "train.log") -> None:
    """Configure the root logger to write to both console and a file.

    Calling this function is idempotent — it removes any handlers previously
    added by this function before attaching new ones.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Remove previous handlers tagged by us
    for h in list(root.handlers):
        if getattr(h, "_hybrid_log", False):
            root.removeHandler(h)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    ch._hybrid_log = True  # type: ignore[attr-defined]
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(output_dir / filename, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    fh._hybrid_log = True  # type: ignore[attr-defined]
    root.addHandler(fh)
