"""Lightweight backtesting helpers used for tests.

This module contains a small, dependency free backtester that is intentionally
minimal.  For the purposes of the kata it has been extended so that the
"fitness" of a candidate solution can depend on a number of hyper-parameters
encoded as *genes*.  These genes represent the configuration of a simple LSTM
model used for signal generation.  A full implementation of the model training
is outside the scope of the exercises, but the backtester accepts the gene
vector so that optimisers can explore the search space.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

GENE_NAMES = [
    "n_layers",
    "hidden_size",
    "dropout",
    "seq_len",
    "horizon",
    "lr",
    "weight_decay",
]


def _apply_postprocess(
    signal: pd.Series,
    *,
    threshold: float,
    ema_alpha: float,
    cooldown: int,
) -> pd.Series:
    """Apply post-processing trading rules to a raw signal.

    The rules implemented are very small proxies of what a production
    environment might use:

    ``threshold``
        Only take a long position if the signal is above this threshold and a
        short position otherwise.
    ``ema_alpha``
        If greater than zero, apply exponential moving average smoothing with
        the provided ``alpha`` parameter.
    ``cooldown``
        Number of bars to wait after a position change before another change is
        allowed.
    """

    if ema_alpha > 0:
        signal = signal.ewm(alpha=ema_alpha, adjust=False).mean()

    # Convert to directional signal based on threshold.
    processed = (signal > threshold).astype(int) * 2 - 1

    if cooldown > 0:
        current = 0
        last_flip = -cooldown
        out = []
        for i, s in enumerate(processed):
            if i - last_flip < cooldown:
                out.append(current)
                continue
            if s != current:
                current = s
                last_flip = i
            out.append(current)
        processed = pd.Series(out, index=signal.index)

    return processed


def run_backtest(
    df: pd.DataFrame,
    genes: Sequence[float] | None = None,
    *,
    threshold: float = 0.0,
    ema_alpha: float = 0.0,
    cooldown: int = 0,

    turnover_penalty: float = 0.0,
) -> float:
=======
    spread_col: str | None = "spread",
    volume_col: str | None = "volume",
    volume_cost: float = 0.0,
    slippage: float = 0.0,
    order_latency: int = 0,
    network_latency: int = 0,
    return_metrics: bool = False,
    rng: np.random.Generator | None = None,
) -> float | dict[str, float]:

    """Run a simple backtest using optional gene parameters.

    Parameters
    ----------
    df:
        Must contain a ``ret`` column representing returns.
    genes:
        Sequence of seven values representing ``n_layers``, ``hidden_size``,
        ``dropout``, ``seq_len``, forecast horizon ``horizon``, learning rate
        ``lr`` and ``weight_decay``. The numeric values are primarily
        placeholders but allow optimisers to tune hyper-parameters.
    threshold, ema_alpha, cooldown:
        Post-processing rules applied to the raw signal when computing the
        fitness of a gene vector.
    turnover_penalty:
        Penalty applied per position change to discourage excessive turnover.

    Returns
    -------
    float
        The cumulative return of the strategy after post-processing or,
        when ``return_metrics`` is ``True``, a dictionary containing
        ``pnl``, ``calmar``, ``max_drawdown`` and ``turnover``.
    """

    if "ret" not in df.columns:
        raise ValueError("missing ret column")

    params: dict[str, float] = {}
    if genes is not None:
        params = dict(zip(GENE_NAMES, genes))
    horizon = int(params.get("horizon", 0))

    # In lieu of training an actual model, use future returns as a proxy for a
    # predictive signal so that the optimisation infrastructure has something to
    # operate on. ``horizon`` defines how many steps ahead the signal should
    # look.
    raw_signal = df["ret"].shift(-horizon).fillna(0)

    signal = _apply_postprocess(
        raw_signal, threshold=threshold, ema_alpha=ema_alpha, cooldown=cooldown
    )
    pnl = (signal.shift().fillna(0) * df["ret"]).cumsum().iloc[-1]
    if turnover_penalty > 0:
        turns = (signal != signal.shift()).sum()
        pnl -= turnover_penalty * float(turns)
    return float(pnl)
=======

    # Simulate latency from order queues and network/broker delays.
    total_latency = max(order_latency, 0) + max(network_latency, 0)
    if total_latency > 0:
        exec_signal = signal.shift(total_latency).fillna(0)
    else:
        exec_signal = signal

    trades = exec_signal.diff().abs().fillna(exec_signal.abs())
    spread = df[spread_col] if spread_col and spread_col in df.columns else 0
    volume = df[volume_col] if volume_col and volume_col in df.columns else 1
    cost = trades * (spread + volume_cost / volume)

    if slippage > 0:
        if rng is None:
            rng = np.random.default_rng()
        cost += np.abs(rng.normal(0.0, slippage, size=len(df))) * trades

    strat_ret = exec_signal.shift().fillna(0) * df["ret"] - cost
    pnl_series = strat_ret.cumsum()
    pnl = pnl_series.iloc[-1] if not pnl_series.empty else 0.0

    if not return_metrics:
        return float(pnl)

    dd = pnl_series.cummax() - pnl_series
    max_dd = float(dd.max()) if not dd.empty else 0.0
    calmar = float(pnl / max_dd) if max_dd != 0 else float("inf")
    turnover = float(trades.sum())

    return {
        "pnl": float(pnl),
        "calmar": calmar,
        "max_drawdown": max_dd,
        "turnover": turnover,
    }



__all__ = ["run_backtest", "GENE_NAMES"]
