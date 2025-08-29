"""Streaming decision loop connecting model, risk manager and OMS."""

from __future__ import annotations

import logging

from typing import Dict, List
from collections import deque
from statistics import median


from .features import FeatureBuilder, Scaler
from .oms import OMS
from .observability import Observability
from .risk import RiskManager, ATRCalculator
from .state import load_snapshot, save_snapshot
from .quality import quality_check

logger = logging.getLogger(__name__)


class SignalToOrdersMapper:
    """Translate target positions into algorithmic order slices."""

    def __init__(
        self,
        oms: OMS,
        *,
        strategy: str = "twap",
        intervals: int = 1,
        participation: float = 0.1,
        vwap_window: int = 5,
    ) -> None:
        self.oms = oms
        self.strategy = strategy
        self.intervals = intervals
        self.participation = participation
        self.vwap_window = vwap_window

    def generate_orders(
        self,
        symbol: str,
        price: float,
        current_position: float,
        target_position: float,
        volume_profile: List[float] | None = None,
    ) -> List[Dict[str, float | str]]:
        diff = target_position - current_position
        if diff == 0:
            return []
        side = "buy" if diff > 0 else "sell"
        qty = abs(diff)
        schedule = self.oms.schedule_child_orders(
            symbol=symbol,
            side=side,
            qty=qty,
            strategy=self.strategy,
            intervals=self.intervals,
            volume_profile=volume_profile,
            participation=self.participation,
        )
        return [{"symbol": symbol, "side": side, "qty": q} for q in schedule]


class DecisionLoop:
    """Consume market data, generate signals and submit orders."""

    def __init__(
        self,
        model,
        risk: RiskManager,
        oms: OMS,
        obs: Observability,
        *,
        order_mapper: SignalToOrdersMapper | None = None,
        ema_span: int = 10,
        threshold: float = 0.0,
        cooldown: int = 0,
        lstm_path: str | None = None,
        snapshot_path: str | None = None,
        snapshot_interval: int = 0,
        median_window: int = 1,
        hysteresis: float = 0.0,
    ) -> None:
        self.model = model
        self.risk = risk
        self.oms = oms
        self.obs = obs
        self.order_mapper = order_mapper or SignalToOrdersMapper(oms)
        self.alpha = 2.0 / (ema_span + 1.0)
        self.threshold = threshold
        self.cooldown = cooldown
        self._ema = 0.0
        self._cooldown = 0
        self.median_window = median_window
        self.hysteresis = hysteresis
        self._median_buf = deque(maxlen=median_window)
        self._signal_state: int | None = None
        # current filled positions per symbol
        self.position: Dict[str, float] = {}
        # feature engineering utilities operating in streaming mode
        self.fb = FeatureBuilder()
        self.scaler = Scaler()
        self.atr = ATRCalculator(window=self.risk.atr_window)
        self.lstm_path = lstm_path
        self.snapshot_path = snapshot_path
        self.snapshot_interval = snapshot_interval
        self._bars_since_snapshot = 0
        if self.lstm_path and hasattr(self.model, "load_state"):
            try:
                self.model.load_state(self.lstm_path)
            except FileNotFoundError:
                pass
        if self.snapshot_path:
            try:
                m, s, pos, hidden = load_snapshot(self.snapshot_path)
                if m is not None:
                    self.model = m
                if s is not None:
                    self.scaler = s
                if pos is not None:
                    self.position = pos
                if hidden is not None and hasattr(self.model, "hidden"):
                    self.model.hidden = hidden
            except FileNotFoundError:
                pass

    def on_bar(self, bar: Dict[str, float]) -> None:
        if not quality_check(bar):
            self.obs.increment_quality_errors()
            return
        feats = self.fb.update(bar)
        scaled = self.scaler.transform(feats[["ret"]])
        self.scaler.update(feats[["ret"]])
        pred = float(self.model.predict(scaled)[0])
        if self.lstm_path and hasattr(self.model, "save_state"):
            self.model.save_state(self.lstm_path)
        self._ema = self.alpha * pred + (1 - self.alpha) * self._ema
        self._median_buf.append(self._ema)
        filtered = (
            median(self._median_buf) if self.median_window > 1 else self._ema
        )
        self._bars_since_snapshot += 1
        if self.snapshot_path and self.snapshot_interval and self._bars_since_snapshot >= self.snapshot_interval:
            self._bars_since_snapshot = 0
            self._save_snapshot()
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        signal = filtered
        if self.hysteresis > 0.0:
            if self._signal_state is None:
                if signal > self.threshold + self.hysteresis:
                    self._signal_state = 1
                elif signal < -self.threshold - self.hysteresis:
                    self._signal_state = -1
                else:
                    return
            else:
                if (
                    self._signal_state == 1
                    and signal < self.threshold - self.hysteresis
                ) or (
                    self._signal_state == -1
                    and signal > -self.threshold + self.hysteresis
                ):
                    self._signal_state = None
                    return
        else:
            if abs(signal) < self.threshold:
                return
        self._cooldown = self.cooldown
        symbol = bar["symbol"]
        price = float(bar["close"])
        sigma = float(self.scaler.std().get("ret", 0.0))
        exposure_limits = {
            "symbol": symbol,
            "current_position": self.position.get(symbol, 0.0),
            "total_notional": price * abs(self.position.get(symbol, 0.0)),
            "corr": 0.0,
            "regime": "bull",
        }
        target = self.risk.target_position(
            prob=signal, price=price, sigma=sigma, exposure_limits=exposure_limits
        )
        atr = self.atr.update(bar)
        _sl, _tp = self.risk.atr_sl_tp(price, atr)
        current = self.position.get(symbol, 0.0)
        diff = target - current
        if diff == 0:
            return
        side = "buy" if diff > 0 else "sell"
        qty = abs(diff)
        notional = price * qty
        if not self.risk.validate_order(
            symbol=symbol,
            notional=notional,
            total_notional=price * abs(current),
            turnover_day=0.0,
            exchange="sim",
        ):
            return
        orders = self.order_mapper.generate_orders(
            symbol=symbol,
            price=price,
            current_position=current,
            target_position=target,
        )
        for i, o in enumerate(orders):
            cid = f"{symbol}-{bar['timestamp']}-{i}"
            try:
                self.oms.submit_order(
                    symbol=o["symbol"],
                    side=o["side"],
                    qty=o["qty"],
                    price=price,
                    client_id=cid,
                )
                self.obs.increment_orders_sent()
            except Exception:  # pragma: no cover - logging
                logger.exception("order submission failed")
                self.obs.increment_order_errors()

    def on_fill(self, order_id: str, qty: float, price: float) -> None:
        """Handle fill events updating position and slippage metric."""

        order = self.oms._by_order_id(order_id)
        self.oms.handle_fill(order_id, qty)
        if order is None:
            return
        sign = 1 if order.side == "buy" else -1
        self.position[order.symbol] = self.position.get(order.symbol, 0.0) + sign * qty
        bps = (price - order.price) / order.price * 1e4 * sign
        self.obs.observe_slippage(bps)

    def reconcile(self) -> None:
        """Delegate reconciliation to OMS."""

        self.oms.reconcile()

    def _save_snapshot(self) -> None:
        if not self.snapshot_path:
            return
        hidden = getattr(self.model, "hidden", None)
        save_snapshot(self.snapshot_path, self.model, self.scaler, self.position, hidden)

    def save(self) -> None:
        """Persist current state to the configured snapshot path."""

        self._save_snapshot()

    def close(self) -> None:
        """Save snapshot before shutting down."""

        self._save_snapshot()


__all__ = ["DecisionLoop", "SignalToOrdersMapper"]
