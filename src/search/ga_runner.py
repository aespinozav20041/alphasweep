"""Helper utilities for genetic algorithm searches."""

from __future__ import annotations

from typing import Iterable, Mapping, Any, Sequence

import pandas as pd

from quant_pipeline.backtest import run_backtest


def run_engine_sim(
    model: Sequence[float],
    oos_stream: Iterable[Mapping[str, Any]] | pd.DataFrame,
    costs: Mapping[str, float],
    *,
    return_metrics: bool = False,
) -> float | dict[str, float]:
    """Evaluate a model on an out-of-sample stream using the backtest engine.

    Parameters
    ----------
    model:
        Sequence of genes describing model hyper-parameters.
    oos_stream:
        Iterable or DataFrame providing market data. When an iterable is
        supplied it is converted to a DataFrame. The data must include a
        ``ret`` column representing returns.
    costs:
        Mapping of post-processing parameters such as ``threshold``,
        ``ema_alpha`` and ``cooldown``.
    return_metrics:
        When ``True`` return rich metrics instead of just PnL.
    """

    if isinstance(oos_stream, pd.DataFrame):
        df = oos_stream
    else:
        df = pd.DataFrame(list(oos_stream))

    return run_backtest(
        df,
        model,
        threshold=float(costs.get("threshold", 0.0)),
        ema_alpha=float(costs.get("ema_alpha", 0.0)),
        cooldown=int(costs.get("cooldown", 0)),
        return_metrics=return_metrics,
    )


__all__ = ["run_engine_sim"]
