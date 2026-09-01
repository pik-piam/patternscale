"""Logging setup for patternscale runs."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import LoggingConfig


def setup_logging(cfg: LoggingConfig) -> None:
    """Configure root logging: console plus optional file handler."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.file:
        Path(cfg.file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(cfg.file))

    logging.basicConfig(
        level=getattr(logging, cfg.level.upper()),
        format="%(asctime)s - %(message)s",
        handlers=handlers,
        force=True,
    )
