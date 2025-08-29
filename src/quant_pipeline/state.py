"""Persistence helpers for saving and loading trading state snapshots."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple


def save_snapshot(
    path: str | Path,
    model: Any,
    scaler: Any,
    positions: Dict[str, float],
    hidden: Any,
    fb: Any,
) -> None:
    """Persist model, scaler, positions, feature builder and hidden state to ``path``."""

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model,
        "scaler": scaler,
        "positions": positions,
        "hidden": hidden,
        "fb": fb,
    }
    with file.open("wb") as fh:
        pickle.dump(data, fh)


def load_snapshot(path: str | Path) -> Tuple[Any, Any, Dict[str, float], Any, Any]:
    """Load model, scaler, positions, feature builder and hidden state from ``path``."""

    file = Path(path)
    with file.open("rb") as fh:
        data = pickle.load(fh)
    return (
        data.get("model"),
        data.get("scaler"),
        data.get("positions", {}),
        data.get("hidden"),
        data.get("fb"),
    )


__all__ = ["save_snapshot", "load_snapshot"]
