
"""Lightweight backtesting helpers used for tests.

This module contains a small, dependency free backtester that is intentionally
minimal.  For the purposes of the kata it has been extended so that the
"fitness" of a candidate solution can depend on a number of hyper-parameters
encoded as genes.  These genes represent the configuration of a simple LSTM
model used for signal generation.  A full implementation of the model training
is outside the scope of the exercises, but the backtester accepts the gene
vector so that optimisers can explore the search space.
=======
"""Lightweight backtester with optional cost modelling.

This module provides a very small backtesting utility used throughout the test
suite.  It supports a handful of hyper-parameters encoded as *genes* so that
optimisers can explore different model configurations.  The implementation here
is intentionally simple yet exposes hooks for common trading frictions such as
spreads, volume based costs, latency and slippage.

"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

GENE_NAMES = [
    "n_lstm",
    "hidden_size",
    "dropout",
    "seq_len",
    "horizon",
    "lr",
    "weight_decay",
    "threshold",
    "ema_alpha",
    "cooldown",
]


def _apply_postprocess(
    signal: pd.Series,
    *,
    threshold: float,
    ema_alpha: float,
    cooldown: int,
) -> pd.Series:
    """Apply simple post-processing rules to a raw trading signal."""

    if ema_alpha > 0:
        signal = signal.ewm(alpha=ema_alpha, adjust=False).mean()

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

    threshold: float | None = None,
    ema_alpha: float | None = None,
    cooldown: int | None = None,
=======
    threshold: float = 0.0,
    ema_alpha: float = 0.0,
    cooldown: int = 0,

    turnover_penalty: float = 0.0,
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
        Sequence of ten values representing ``n_lstm``, ``hidden_size``,
        ``dropout``, ``seq_len``, forecast horizon ``horizon``, learning rate
        ``lr`` and ``weight_decay`` along with post-processing rules
        ``threshold``, ``ema_alpha`` and ``cooldown``.  The numeric values are
        primarily placeholders but allow optimisers to tune hyper-parameters.
    threshold, ema_alpha, cooldown:
        Optional overrides for the post-processing rules when not provided in
        ``genes``.
    turnover_penalty:
        Penalty applied per position change to discourage excessive turnover.

    Returns
    -------
    float
        The cumulative return of the strategy after post-processing or,
        when ``return_metrics`` is ``True``, a dictionary containing
        ``pnl``, ``calmar``, ``max_drawdown`` and ``turnover``.
=======
    """Run a tiny backtest using optional gene parameters.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a ``ret`` column with returns.
    genes : sequence of float, optional
        Vector of seven hyper-parameters as defined in ``GENE_NAMES``.
    threshold, ema_alpha, cooldown : float / int
        Post-processing rules applied to the raw signal.
    turnover_penalty : float
        Penalty applied per position change.
    spread_col, volume_col : str, optional
        Column names used to pull spread and volume information.
    volume_cost, slippage : float
        Transaction cost parameters.
    order_latency, network_latency : int
        Delays applied to the execution signal.
    return_metrics : bool
        When ``True`` return rich metrics instead of just PnL.
    rng : numpy.random.Generator, optional
        Random number generator used for slippage simulation.

    """

    if "ret" not in df.columns:
        raise ValueError("missing ret column")

    params: dict[str, float] = {}
    if genes is not None:
        params = dict(zip(GENE_NAMES, genes))

    horizon = int(params.get("horizon", 0))
    thr = threshold if threshold is not None else float(params.get("threshold", 0.0))
    ema = ema_alpha if ema_alpha is not None else float(params.get("ema_alpha", 0.0))
    cd = cooldown if cooldown is not None else int(params.get("cooldown", 0))

    raw_signal = df["ret"].shift(-horizon).fillna(0)

    signal = _apply_postprocess(raw_signal, threshold=thr, ema_alpha=ema, cooldown=cd)
=======
    signal = _apply_postprocess(
        raw_signal, threshold=threshold, ema_alpha=ema_alpha, cooldown=cooldown
    )


    total_latency = max(order_latency, 0) + max(network_latency, 0)
    exec_signal = signal.shift(total_latency).fillna(0) if total_latency > 0 else signal

    trades = exec_signal.diff().abs().fillna(exec_signal.abs())
    spread = df[spread_col] if spread_col and spread_col in df.columns else 0
    volume = df[volume_col] if volume_col and volume_col in df.columns else 1
    cost = trades * (spread + volume_cost / volume + turnover_penalty)

    if slippage > 0:
        if rng is None:
            rng = np.random.default_rng()
        cost += np.abs(rng.normal(0.0, slippage, size=len(df))) * trades

    strat_ret = exec_signal.shift().fillna(0) * df["ret"] - cost
    pnl_series = strat_ret.cumsum()
    pnl = pnl_series.iloc[-1] if not pnl_series.empty else 0.0

    if turnover_penalty > 0:
        turns = (exec_signal != exec_signal.shift()).sum()
        pnl -= turnover_penalty * float(turns)

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

