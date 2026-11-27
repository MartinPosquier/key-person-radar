"""Frozen logging helper: `log = get_logger(__name__)` in every module."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_FORMAT, datefmt="%H:%M:%S")
    return logging.getLogger(name)
