from __future__ import annotations
"""Lightweight engine hooks.

Allows external modules to register callbacks that are invoked during the
trading process.  Only the hooks required for attribution and stress testing are
implemented."""

from typing import Callable, List
from pathlib import Path
import json


_before_rebalance: List[Callable[[dict], None]] = []
_after_rebalance: List[Callable[[dict], None]] = []
_before_order: List[Callable[[dict], None]] = []
_after_fill: List[Callable[[dict], None]] = []


def register_before_rebalance(fn: Callable[[dict], None]) -> None:
    _before_rebalance.append(fn)


def register_after_rebalance(fn: Callable[[dict], None]) -> None:
    _after_rebalance.append(fn)


def register_before_order(fn: Callable[[dict], None]) -> None:
    _before_order.append(fn)


def register_after_fill(fn: Callable[[dict], None]) -> None:
    _after_fill.append(fn)


def trigger_before_rebalance(weights: dict) -> None:
    for fn in _before_rebalance:
        fn(weights)


def trigger_after_rebalance(weights: dict) -> None:
    for fn in _after_rebalance:
        fn(weights)


def trigger_before_order(order: dict) -> None:
    for fn in _before_order:
        fn(order)


def trigger_after_fill(fill: dict) -> None:
    for fn in _after_fill:
        fn(fill)


# Example default hook storing weights to disk for attribution ----------------

def save_weights_hook(path: Path) -> Callable[[dict], None]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _hook(weights: dict) -> None:
        path.write_text(json.dumps(weights))

    return _hook
