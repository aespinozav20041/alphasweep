"""Background model training scheduler."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Iterator, Tuple, Sequence

from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from .datasets import make_lstm_windows
from .model_registry import ModelRegistry
from .labeling import forward_return, triple_barrier_labels

from .backtest import run_backtest
from .ensemble import blend_horizons
=======
from search.ga_runner import run_engine_sim

from .genetic import GeneticOptimizer

logger = logging.getLogger(__name__)


def train_with_genetic(
    df: pd.DataFrame,
    *,
    generations: int = 3,
    population_size: int = 5,
    rng: np.random.Generator | None = None,
) -> Dict[str, Any]:
    """Explore hyper-parameters using :class:`GeneticOptimizer`.

    Parameters
    ----------
    df:
        Dataset containing at least a ``ret`` column used by the backtest
        fitness function.
    generations, population_size:
        Genetic algorithm settings controlling search effort.
    rng:
        Optional NumPy random generator for deterministic behaviour.

    Returns
    -------
    dict
        Training info compatible with :class:`AutoTrainer`.
    """

    if "ret" not in df.columns:
        raise ValueError("dataset must contain 'ret' column")

    bounds = [
        (1, 3),  # n_lstm
        (8, 64),  # hidden
        (0.0, 0.5),  # dropout
        (5, 50),  # seq_len
        (1, 10),  # horizon
        (1e-4, 1e-2),  # lr
        (0.0, 0.1),  # weight_decay
        (0.0, 1.0),  # umbral
        (0.0, 1.0),  # ema
        (0, 10),  # cooldown
    ]

    def fitness(x: np.ndarray) -> float:
        costs = {
            "threshold": float(x[7]),
            "ema_alpha": float(x[8]),
            "cooldown": int(round(x[9])),
        }
        pnl = run_engine_sim(x[:7], df, costs)
        return float(pnl)

    opt = GeneticOptimizer(
        fitness,
        bounds,
        population_size=population_size,
        rng=rng,
    )
    best, _ = opt.optimise(generations=generations, patience=generations)
    best = best.tolist()

    costs = {
        "threshold": float(best[7]),
        "ema_alpha": float(best[8]),
        "cooldown": int(round(best[9])),
    }
    metrics = run_engine_sim(best[:7], df, costs, return_metrics=True)

    params = {
        "n_lstm": int(round(best[0])),
        "hidden": int(round(best[1])),
        "dropout": float(best[2]),
        "seq_len": int(round(best[3])),
        "horizon": int(round(best[4])),
        "lr": float(best[5]),
        "weight_decay": float(best[6]),
        "umbral": float(best[7]),
        "ema": float(best[8]),
        "cooldown": int(round(best[9])),
    }

    tmp = Path(tempfile.mkdtemp(prefix="ga_model_"))
    art = tmp / "artifact.txt"
    calib = tmp / "calib.txt"
    art.write_text("artifact")
    calib.write_text("calibration")

    return {
        "type": "ga",
        "genes_json": json.dumps(params),
        "artifact_path": str(art),
        "calib_path": str(calib),
        "oos_params": params,
        "oos_metrics": metrics,
    }


class AutoTrainer:
    """Periodically trains models on recent data and registers challengers."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        train_every_bars: int,
        history_days: int,
        max_challengers: int,
        dataset_loader: Callable[[int], pd.DataFrame] | None = None,
        build_dataset: Callable[[int], pd.DataFrame] | None = None,
        train_model: Callable[[Any], Dict[str, str]],
        num_parallel: int = 1,
        seq_len: int = 10,
        cv_splits: int = 5,
        embargo: int = 0,
        label_horizon: int | Sequence[int] | None = None,
        label_up_mult: float = 1.0,
        label_down_mult: float = 1.0,
        label_type: str = "triple_barrier",
    ) -> None:
        self.registry = registry
        self.train_every_bars = train_every_bars
        self.history_days = history_days
        self.max_challengers = max_challengers
        loader = dataset_loader or build_dataset
        if loader is None:
            raise ValueError("dataset_loader or build_dataset must be provided")
        self.dataset_loader = loader
        self.train_model = train_model
        self.seq_len = seq_len
        self.cv = PurgedKFold(n_splits=cv_splits, embargo=embargo)
        self.num_parallel = max(1, num_parallel)
        self.label_horizon = label_horizon
        self.label_up_mult = label_up_mult
        self.label_down_mult = label_down_mult
        self.label_type = label_type
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
        df = self.dataset_loader(self.history_days)
        if self.label_horizon is not None:
            df = df.copy()
            horizons: Sequence[int]
            if isinstance(self.label_horizon, Sequence) and not isinstance(
                self.label_horizon, (str, bytes)
            ):
                horizons = list(self.label_horizon)
            else:
                horizons = [int(self.label_horizon)]

            signals: Dict[int, np.ndarray]
            if self.label_type == "forward_return":
                signals = {h: forward_return(df, h) for h in horizons}
            elif self.label_type == "triple_barrier":
                signals = {
                    h: triple_barrier_labels(
                        df, self.label_up_mult, self.label_down_mult, h
                    )
                    for h in horizons
                }
            else:
                raise ValueError(f"unknown label_type {self.label_type}")

            df["label"] = blend_horizons(signals)

        if "label" in df.columns:
            dataset = self.prepare_dataset(df)
        else:
            dataset = df

        def _train() -> Dict[str, str]:
            return self.train_model(dataset)

        if self.num_parallel == 1:
            infos = [_train()]
        else:
            with ThreadPoolExecutor(max_workers=self.num_parallel) as ex:
                futures = [ex.submit(_train) for _ in range(self.num_parallel)]
                infos = [f.result() for f in futures]

        registered = False
        for info in infos:
            if not info:
                logger.warning("training produced no model")
                continue
            model_id = self.registry.register_model(
                model_type=info["type"],
                genes_json=info.get("genes_json", "{}"),
                artifact_path=info["artifact_path"],
                calib_path=info["calib_path"],
                lstm_path=info.get("lstm_path"),
                scaler_path=info.get("scaler_path"),
                features_path=info.get("features_path"),
                thresholds_path=info.get("thresholds_path"),
                risk_rules_path=info.get("risk_rules_path"),
                ga_version=info.get("ga_version"),
                seed=info.get("seed"),
                data_hash=info.get("data_hash"),
                ts=int(time.time()),
            )
            logger.info("registered challenger %s for shadow eval", model_id)
            params = info.get("oos_params") or json.loads(info.get("genes_json", "{}"))
            metrics = info.get("oos_metrics")
            if metrics:
                self.registry.log_oos_metrics(
                    model_id, params=params, metrics=metrics, ts=int(time.time())
                )
            registered = True

        if registered:
            self.registry.prune_challengers(self.max_challengers)

    def prepare_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, list[Tuple[np.ndarray, np.ndarray]]]:
        """Create LSTM training windows and purged CV splits."""

        X, y = make_lstm_windows(df, self.seq_len)
        splits = list(self.cv.split(X))
        return X, y, splits

    # Backwards compatibility
    def build_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, list[Tuple[np.ndarray, np.ndarray]]]:
        return self.prepare_dataset(df)


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

__all__ = ["AutoTrainer", "PurgedKFold", "train_with_genetic"]
