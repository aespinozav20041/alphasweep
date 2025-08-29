"""Streaming decision loop connecting model, risk manager and OMS."""

from __future__ import annotations

import logging
from typing import Dict

from .features import FeatureBuilder, Scaler
from .oms import OMS
from .observability import Observability
from .risk import RiskManager, ATRCalculator

logger = logging.getLogger(__name__)


class DecisionLoop:
    """Consume market data, generate signals and submit orders."""

    def __init__(
        self,
        model,
        risk: RiskManager,
        oms: OMS,
        obs: Observability,
        *,
        ema_span: int = 10,
        threshold: float = 0.0,
        cooldown: int = 0,
    ) -> None:
        self.model = model
        self.risk = risk
        self.oms = oms
        self.obs = obs
        self.alpha = 2.0 / (ema_span + 1.0)
        self.threshold = threshold
        self.cooldown = cooldown
        self._ema = 0.0
        self._cooldown = 0
        # current filled positions per symbol
        self.position: Dict[str, float] = {}
        # feature engineering utilities operating in streaming mode
        self.fb = FeatureBuilder()
        self.scaler = Scaler()
        self.atr = ATRCalculator(window=self.risk.atr_window)

    def on_bar(self, bar: Dict[str, float]) -> None:
        feats = self.fb.update(bar)
        scaled = self.scaler.transform(feats[["ret"]])
        self.scaler.update(feats[["ret"]])
        pred = float(self.model.predict(scaled)[0])
        self._ema = self.alpha * pred + (1 - self.alpha) * self._ema
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        if abs(self._ema) < self.threshold:
            return
        self._cooldown = self.cooldown
        symbol = bar["symbol"]
        price = float(bar["close"])
        sigma = float(self.scaler.std().get("ret", 0.0))
        target = self.risk.kelly_position(mu=self._ema, sigma=sigma)
        weights = self.risk.apply_correlation_throttle({symbol: target}, corr=0.0, regime="bull")
        target = weights[symbol]
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
        cid = f"{symbol}-{bar['timestamp']}"
        try:
            self.oms.submit_order(
                symbol=symbol,
                side=side,
                qty=qty,
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


__all__ = ["DecisionLoop"]
