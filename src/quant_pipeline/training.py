"""Background model training scheduler."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Iterator, Tuple

import numpy as np

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
        num_parallel: int = 1,
    ) -> None:
        self.registry = registry
        self.train_every_bars = train_every_bars
        self.history_days = history_days
        self.max_challengers = max_challengers
        self.build_dataset = build_dataset
        self.train_model = train_model
        self.num_parallel = max(1, num_parallel)
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
        from concurrent.futures import ThreadPoolExecutor

        def _train() -> Dict[str, str]:
            return self.train_model(dataset)

        with ThreadPoolExecutor(max_workers=self.num_parallel) as ex:
            futures = [ex.submit(_train) for _ in range(self.num_parallel)]
            for fut in futures:
                info = fut.result()
                if not info:
                    logger.warning("training produced no model")
                    continue
                model_id = self.registry.register_model(
                    model_type=info["type"],
                    genes_json=info.get("genes_json", "{}"),
                    artifact_path=info["artifact_path"],
                    calib_path=info["calib_path"],
                    ts=int(time.time()),
                )
                logger.info("registered challenger %s for shadow eval", model_id)
        self.registry.prune_challengers(self.max_challengers)


class PurgedKFold:
    """K-Fold cross-validator with purging and optional embargo.

    This splitter avoids look-ahead bias by removing training samples that
    overlap with the test fold and by adding an embargo period after each
    test set.

    Parameters
    ----------
    n_splits : int
        Number of folds. Must be at least 2.
    embargo : int
        Number of observations to exclude after each test fold.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, X: Any) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            embargo_start = min(n_samples, stop + self.embargo)
            train_start = indices[: max(0, start - self.embargo)]
            train_end = indices[embargo_start:]
            train_indices = np.concatenate([train_start, train_end])
            yield train_indices, test_indices
            current = stop

__all__ = ["AutoTrainer", "PurgedKFold"]
