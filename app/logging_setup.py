"""Structured-ish logging to stdout (RunPod captures container stdout)."""

from __future__ import annotations

import logging
import sys


def setup(name: str = "nemotron35", level: int | None = None) -> logging.Logger:
    logging.basicConfig(
        level=level or logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(name)
