"""Persistence helpers for saving and loading trading state snapshots."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple


def save_snapshot(path: str | Path, model: Any, scaler: Any, positions: Dict[str, float], hidden: Any) -> None:
    """Persist model, scaler, positions and hidden state to ``path``."""

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model,
        "scaler": scaler,
        "positions": positions,
        "hidden": hidden,
    }
    with file.open("wb") as fh:
        pickle.dump(data, fh)


def load_snapshot(path: str | Path) -> Tuple[Any, Any, Dict[str, float], Any]:
    """Load model, scaler, positions and hidden state from ``path``."""

    file = Path(path)
    with file.open("rb") as fh:
        data = pickle.load(fh)
    return data.get("model"), data.get("scaler"), data.get("positions", {}), data.get("hidden")


__all__ = ["save_snapshot", "load_snapshot"]
