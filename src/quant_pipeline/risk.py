"""Risk management utilities with kill-switch and limits."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional


logger = logging.getLogger(__name__)


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
    def apply_correlation_throttle(self, weights: Dict[str, float], corr: float) -> Dict[str, float]:
        """Reduce BTC/ETH weights if correlation exceeds threshold."""

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
        return weights


__all__ = ["RiskManager"]
