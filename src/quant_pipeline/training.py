"""Background model training scheduler."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class AutoTrainer:
    """Periodically trains models on recent data and registers challengers."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        train_every_bars: int,
        history_days: int,
        max_challengers: int,
        build_dataset: Callable[[int], Any],
        train_model: Callable[[Any], Dict[str, str]],
    ) -> None:
        self.registry = registry
        self.train_every_bars = train_every_bars
        self.history_days = history_days
        self.max_challengers = max_challengers
        self.build_dataset = build_dataset
        self.train_model = train_model
        self._bar_count = 0
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        """Start the background scheduler."""

        logger.info("starting AutoTrainer")
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler."""

        self._stop.set()
        self._event.set()
        self._thread.join(timeout=5)
        logger.info("AutoTrainer stopped")

    def notify_bar(self) -> None:
        """Notify the trainer that a new bar has arrived."""

        self._bar_count += 1
        self._event.set()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            self._event.wait()
            self._event.clear()
            if self._bar_count >= self.train_every_bars:
                self._bar_count = 0
                try:
                    self._train_cycle()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("training cycle failed")

    def _train_cycle(self) -> None:
        logger.info("training cycle started")
        dataset = self.build_dataset(self.history_days)
        info = self.train_model(dataset)
        if not info:
            logger.warning("training produced no model")
            return
        model_id = self.registry.register_model(
            model_type=info["type"],
            genes_json=info.get("genes_json", "{}"),
            artifact_path=info["artifact_path"],
            calib_path=info["calib_path"],
            ts=int(time.time()),
        )
        logger.info("registered challenger %s for shadow eval", model_id)
        self.registry.prune_challengers(self.max_challengers)


__all__ = ["AutoTrainer"]
