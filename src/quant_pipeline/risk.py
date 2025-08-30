"""Risk management utilities with kill-switch and limits."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import yaml


logger = logging.getLogger(__name__)


def rolling_covariance(returns: pd.DataFrame, window: int = 60) -> np.ndarray:
    """Compute rolling covariance matrix for recent returns."""

    if len(returns) < 2:
        raise ValueError("need at least two observations")
    win = min(len(returns), window)
    return returns.tail(win).cov().to_numpy()


def risk_parity_weights(
    cov: np.ndarray, *, tol: float = 1e-8, max_iter: int = 1000
) -> np.ndarray:
    """Compute risk-parity weights from a covariance matrix."""

    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        port_var = w @ cov @ w
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n
        diff = rc - target
        if np.all(np.abs(diff) < tol):
            break
        w -= diff / (mrc + 1e-12)
        w = np.clip(w, 1e-12, None)
        w /= w.sum()
    return w


@dataclass
class RiskConfig:
    """Configuration dataclass loaded from ``conf/risk.yaml``."""

    max_dd_daily: float = 0.05
    max_dd_weekly: float = 0.1
    latency_threshold: float = 100.0
    latency_window: int = 3
    pause_minutes: int = 1
    max_position_notional_symbol: Dict[str, float] | None = None
    max_total_notional: float = float("inf")
    max_turnover_day: float = float("inf")
    min_notional_exchange: Dict[str, float] | None = None
    correlation_threshold: float = 0.8
    corr_reduction: float = 0.5
    target_volatility: float = 0.02
    kelly_fraction: float = 1.0
    sl_atr: float = 3.0
    tp_atr: float = 6.0
    atr_window: int = 14
    regime_reduction: float = 0.5


def load_risk_config(path: str | Path = Path("conf/risk.yaml")) -> RiskConfig:
    """Load risk limits configuration from YAML file."""

    cfg = RiskConfig()
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for key, value in data.items():
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, value)
    return cfg


class ATRCalculator:
    """Rolling Average True Range calculator."""

    def __init__(self, window: int = 14) -> None:
        self.window = window
        self.tr = deque(maxlen=window)
        self.prev_close: float | None = None

    def update(self, bar: Dict[str, float]) -> float:
        close = float(bar["close"])
        high = float(bar.get("high", close))
        low = float(bar.get("low", close))
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.tr.append(tr)
        self.prev_close = close
        return sum(self.tr) / len(self.tr) if self.tr else 0.0


class RiskManager:
    """Risk manager enforcing drawdown, latency and notional limits.

    Parameters
    ----------
    max_dd_daily: float
        Maximum acceptable daily drawdown (as ratio, e.g. 0.05 for 5%).
    max_dd_weekly: float
        Maximum acceptable weekly drawdown.
    latency_threshold: float
        Latency threshold in milliseconds triggering the kill-switch when
        exceeded for ``latency_window`` consecutive observations.
    latency_window: int
        Number of recent latency observations considered.
    pause_minutes: int
        Minutes to pause trading once kill-switch is activated.
    max_position_notional_symbol: Dict[str, float]
        Maximum position notional per symbol.
    max_total_notional: float
        Maximum aggregate notional exposure.
    max_turnover_day: float
        Maximum turnover allowed per day.
    min_notional_exchange: Dict[str, float]
        Minimum notional per exchange.
    correlation_threshold: float, optional
        Correlation level above which to throttle BTC/ETH weights.
    corr_reduction: float, optional
        Fraction of aggregated BTC+ETH weights to keep when throttled.
    position_closer: Callable[[], None], optional
        Callback invoked to close positions when kill-switch activates.
    """

    def __init__(
        self,
        *,
        max_dd_daily: float,
        max_dd_weekly: float,
        latency_threshold: float,
        latency_window: int,
        pause_minutes: int,
        max_position_notional_symbol: Dict[str, float] | None = None,
        max_total_notional: float = float("inf"),
        max_turnover_day: float = float("inf"),
        min_notional_exchange: Dict[str, float] | None = None,
        correlation_threshold: float = 0.8,
        corr_reduction: float = 0.5,
        position_closer: Callable[[], None] | None = None,
        target_volatility: float = 0.02,
        kelly_fraction: float = 1.0,
        sl_atr: float = 3.0,
        tp_atr: float = 6.0,
        atr_window: int = 14,
        regime_reduction: float = 0.5,
    ) -> None:
        self.max_dd_daily = max_dd_daily
        self.max_dd_weekly = max_dd_weekly
        self.latency_threshold = latency_threshold
        self.latencies = deque(maxlen=latency_window)
        self.pause_minutes = pause_minutes
        self.max_position_notional_symbol = max_position_notional_symbol or {}
        self.max_total_notional = max_total_notional
        self.max_turnover_day = max_turnover_day
        self.min_notional_exchange = min_notional_exchange or {}
        self.correlation_threshold = correlation_threshold
        self.corr_reduction = corr_reduction
        self.position_closer = position_closer
        self.target_volatility = target_volatility
        self.kelly_fraction = kelly_fraction
        self.sl_atr = sl_atr
        self.tp_atr = tp_atr
        self.atr_window = atr_window
        self.regime_reduction = regime_reduction
        self.pause_until: Optional[datetime] = None
        self._daily_dd = 0.0
        self._weekly_dd = 0.0

    # ------------------------------------------------------------------
    # Kill-switch handling
    # ------------------------------------------------------------------
    def update_drawdowns(self, daily: float, weekly: float) -> None:
        """Update drawdowns and evaluate kill-switch."""

        self._daily_dd = daily
        self._weekly_dd = weekly
        self._check_kill_switch()

    def record_latency(self, ms: float) -> None:
        """Record latency observation and evaluate kill-switch."""

        self.latencies.append(ms)
        self._check_kill_switch()

    def _check_kill_switch(self) -> None:
        if self._daily_dd >= self.max_dd_daily or self._weekly_dd >= self.max_dd_weekly:
            self._trigger_pause("drawdown threshold exceeded")
            return
        if len(self.latencies) == self.latencies.maxlen and all(
            l >= self.latency_threshold for l in self.latencies
        ):
            self._trigger_pause("latency threshold exceeded")

    def _trigger_pause(self, reason: str) -> None:
        if self.is_paused():
            return
        logger.warning("Kill-switch triggered: %s", reason)
        if self.position_closer:
            try:
                self.position_closer()
            except Exception as exc:  # pragma: no cover - logging only
                logger.error("Failed to close positions: %s", exc)
        self.pause_until = datetime.utcnow() + timedelta(minutes=self.pause_minutes)

    def is_paused(self) -> bool:
        return self.pause_until is not None and datetime.utcnow() < self.pause_until

    # ------------------------------------------------------------------
    # Limit enforcement
    # ------------------------------------------------------------------
    def validate_order(
        self,
        *,
        symbol: str,
        notional: float,
        total_notional: float,
        turnover_day: float,
        exchange: str,
    ) -> bool:
        """Validate order against limits and kill-switch state.

        Returns ``True`` if the order is allowed, otherwise ``False`` with a log.
        """

        if self.is_paused():
            logger.warning("Trading paused until %s", self.pause_until)
            return False
        max_sym = self.max_position_notional_symbol.get(symbol)
        if max_sym is not None and abs(notional) > max_sym:
            logger.warning("Max position notional exceeded for %s", symbol)
            return False
        if abs(total_notional + notional) > self.max_total_notional:
            logger.warning("Max total notional exceeded")
            return False
        if abs(turnover_day + notional) > self.max_turnover_day:
            logger.warning("Max daily turnover exceeded")
            return False
        min_ex = self.min_notional_exchange.get(exchange)
        if min_ex is not None and abs(notional) < min_ex:
            logger.warning("Order notional below exchange minimum for %s", exchange)
            return False
        return True

    # ------------------------------------------------------------------
    # Correlation throttle
    # ------------------------------------------------------------------
    def apply_correlation_throttle(
        self, weights: Dict[str, float], *, corr: float, regime: str | None = None
    ) -> Dict[str, float]:
        """Reduce weights based on correlation or market regime."""

        if corr > self.correlation_threshold:
            btc = weights.get("BTC", 0.0)
            eth = weights.get("ETH", 0.0)
            total = btc + eth
            if total:
                scale = self.corr_reduction
                weights["BTC"] = btc * scale
                weights["ETH"] = eth * scale
                logger.info(
                    "Correlation throttle applied: corr %.2f > %.2f",
                    corr,
                    self.correlation_threshold,
                )
        if regime and regime.lower() != "bull":
            weights = {k: v * self.regime_reduction for k, v in weights.items()}
            logger.info("Regime throttle applied: %s", regime)
        return weights

    def kelly_position(self, *, mu: float, sigma: float) -> float:
        """Compute position size using fractional Kelly and target volatility."""

        if sigma <= 0:
            return 0.0
        kelly = mu / (sigma ** 2)
        kelly *= self.kelly_fraction
        return kelly * (self.target_volatility / sigma)

    def target_position(
        self,
        prob: float,
        price: float,
        sigma: float,
        exposure_limits: Dict[str, float],
    ) -> float:
        """Return target position size after throttles and notional limits.

        Parameters
        ----------
        prob: float
            Expected return or probability used by ``kelly_position``.
        price: float
            Current instrument price used to convert limits to quantity.
        sigma: float
            Realized volatility of the signal.
        exposure_limits: Dict[str, float]
            Dictionary with keys ``symbol``, ``current_position`` and
            ``total_notional`` as well as optional ``corr`` and ``regime``.
        """

        symbol = exposure_limits.get("symbol")
        corr = float(exposure_limits.get("corr", 0.0))
        regime = exposure_limits.get("regime")
        current_pos = float(exposure_limits.get("current_position", 0.0))
        total_notional = float(exposure_limits.get("total_notional", 0.0))

        # Base Kelly position scaled to target volatility
        target = self.kelly_position(mu=prob, sigma=sigma)

        # Apply correlation and regime throttles
        weights = self.apply_correlation_throttle({symbol: target}, corr=corr, regime=regime)
        target = weights.get(symbol, 0.0)

        if price <= 0:
            return 0.0

        # Enforce per-symbol notional limits
        max_sym = self.max_position_notional_symbol.get(symbol)
        if max_sym is not None:
            max_qty = max_sym / price
            target = max(-max_qty, min(max_qty, target))

        # Enforce aggregate notional limit
        current_notional = price * abs(current_pos)
        new_total = total_notional - current_notional + price * abs(target)
        if new_total > self.max_total_notional:
            remaining = self.max_total_notional - (total_notional - current_notional)
            remaining = max(0.0, remaining)
            max_qty = remaining / price
            target = min(max_qty, abs(target)) * (1 if target >= 0 else -1)

        return target

    def atr_sl_tp(self, price: float, atr: float) -> tuple[float, float]:
        """Return stop-loss and take-profit levels based on ATR."""

        sl = price - self.sl_atr * atr
        tp = price + self.tp_atr * atr
        return sl, tp


__all__ = ["RiskManager", "ATRCalculator", "RiskConfig", "load_risk_config"]
