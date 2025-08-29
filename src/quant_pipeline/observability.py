"""Observability utilities: Prometheus metrics and alerting."""

from __future__ import annotations

import logging
import os
import smtplib
import time
from email.message import EmailMessage
from typing import Optional

import requests
from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


logger = logging.getLogger(__name__)


class Observability:
    """Helper class exposing Prometheus metrics and simple alerting."""

    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        email_host: Optional[str] = None,
        email_from: Optional[str] = None,
        email_to: Optional[str] = None,
    ) -> None:
        self.registry = CollectorRegistry()
        self.data_missing_bars_ratio = Gauge(
            "data_missing_bars_ratio",
            "Ratio of missing bars in data",
            ["symbol", "timeframe"],
            registry=self.registry,
        )
        self.data_dup_ratio = Gauge(
            "data_dup_ratio",
            "Ratio of duplicate bars",
            ["symbol"],
            registry=self.registry,
        )
        self.data_outliers_count = Gauge(
            "data_outliers_count",
            "Count of detected outliers",
            ["symbol"],
            registry=self.registry,
        )
        self.features_build_seconds = Histogram(
            "features_build_seconds",
            "Time to build features",
            registry=self.registry,
        )
        self.backtest_seconds = Histogram(
            "backtest_seconds",
            "Time spent running backtests",
            registry=self.registry,
        )
        self.trainwf_seconds = Histogram(
            "trainwf_seconds",
            "Training time for walk-forward models",
            registry=self.registry,
        )
        self.orders_sent_total = Counter(
            "orders_sent_total",
            "Total number of orders sent",
            registry=self.registry,
        )
        self.order_errors_total = Counter(
            "order_errors_total",
            "Total number of order send errors",
            registry=self.registry,
        )
        self.slippage_bps = Histogram(
            "slippage_bps",
            "Observed slippage in basis points",
            registry=self.registry,
        )
        self.latency_ms = Histogram(
            "latency_ms",
            "Observed latency in milliseconds",
            registry=self.registry,
        )
        self.sharpe_ratio = Gauge(
            "sharpe_ratio",
            "Current strategy Sharpe ratio",
            registry=self.registry,
        )
        self.broker_api_latency_ms = Histogram(
            "broker_api_latency_ms",
            "Latency of broker API calls in milliseconds",
            registry=self.registry,
        )
        self.risk_fetch_failures_total = Counter(
            "risk_fetch_failures_total",
            "Total number of risk fetch failures",
            registry=self.registry,
        )
        self.stable_sweep_amount_usd = Gauge(
            "stable_sweep_amount_usd",
            "USD amount swept in last stable sweep",
            registry=self.registry,
        )
        self.stable_sweep_blocked_total = Counter(
            "stable_sweep_blocked_total",
            "Total number of stable sweeps blocked",
            registry=self.registry,
        )
        self.stable_orders_filled_total = Counter(
            "stable_orders_filled_total",
            "Total number of stable orders filled",
            registry=self.registry,
        )
        self._heartbeat_gauge = Gauge(
            "heartbeat_timestamp",
            "Timestamp of last heartbeat",
            registry=self.registry,
        )
        self._heartbeat_ts = 0.0

        # alerting configuration
        self.slack_webhook = slack_webhook or os.getenv("SLACK_WEBHOOK_URL")
        self.email_host = email_host or os.getenv("ALERT_SMTP_HOST")
        self.email_from = email_from or os.getenv("ALERT_EMAIL_FROM")
        self.email_to = email_to or os.getenv("ALERT_EMAIL_TO")

        # FastAPI application exposing /metrics
        self.app = FastAPI()

        @self.app.get("/metrics")
        def metrics() -> Response:
            return Response(
                generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST
            )

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    def report_missing_bars(
        self, symbol: str, timeframe: str, ratio: float, threshold: float | None = None
    ) -> None:
        """Record missing bar ratio and alert if above threshold."""

        self.data_missing_bars_ratio.labels(symbol=symbol, timeframe=timeframe).set(
            ratio
        )
        if threshold is not None and ratio > threshold:
            self._send_alert(
                f"Missing bars ratio {ratio:.2%} for {symbol} {timeframe} exceeds {threshold:.2%}"
            )

    def report_duplicate_ratio(self, symbol: str, ratio: float) -> None:
        self.data_dup_ratio.labels(symbol=symbol).set(ratio)

    def report_outliers(self, symbol: str, count: int) -> None:
        self.data_outliers_count.labels(symbol=symbol).set(count)

    def observe_features_build(self, seconds: float) -> None:
        self.features_build_seconds.observe(seconds)

    def observe_backtest(self, seconds: float, success: bool = True) -> None:
        self.backtest_seconds.observe(seconds)
        if not success:
            self._send_alert("Backtest failed")

    def observe_trainwf(self, seconds: float) -> None:
        self.trainwf_seconds.observe(seconds)

    def increment_orders_sent(self, n: int = 1) -> None:
        self.orders_sent_total.inc(n)

    def increment_order_errors(self, n: int = 1) -> None:
        self.order_errors_total.inc(n)
        self._send_alert("Order error encountered")

    def observe_slippage(self, bps: float, threshold: float | None = None) -> None:
        self.slippage_bps.observe(bps)
        if threshold is not None and bps > threshold:
            self._send_alert(
                f"Slippage {bps:.2f}bps exceeds {threshold:.2f}bps"
            )

    def observe_latency(self, ms: float, threshold: float | None = None) -> None:
        self.latency_ms.observe(ms)
        if threshold is not None and ms > threshold:
            self._send_alert(f"Latency {ms:.0f}ms exceeds {threshold:.0f}ms")

    def observe_sharpe(self, sharpe: float, threshold: float | None = None) -> None:
        """Record Sharpe ratio and alert if below ``threshold``."""

        self.sharpe_ratio.set(sharpe)
        if threshold is not None and sharpe < threshold:
            self._send_alert(
                f"Sharpe ratio {sharpe:.2f} below {threshold:.2f}"
            )

    def observe_broker_latency(self, ms: float, threshold: float | None = None) -> None:
        """Record broker API latency and alert if above ``threshold``."""

        self.broker_api_latency_ms.observe(ms)
        if threshold is not None and ms > threshold:
            self._send_alert(f"Broker API latency {ms:.0f}ms exceeds {threshold:.0f}ms")

    def increment_risk_fetch_failures(self, n: int = 1) -> None:
        """Increment risk fetch failure counter and alert."""

        self.risk_fetch_failures_total.inc(n)
        self._send_alert("Risk fetch failure")

    def alert_connection_failure(self, service: str) -> None:
        """Send connection failure alert for ``service``."""

        self._send_alert(f"Connection failure: {service}")

    def alert_timeout(self, operation: str) -> None:
        """Send timeout alert for ``operation``."""

        self._send_alert(f"Timeout during {operation}")

    def alert_anomalous_ratio(self, name: str, ratio: float, threshold: float) -> None:
        """Alert if ``ratio`` for ``name`` exceeds ``threshold``."""

        if ratio > threshold:
            self._send_alert(f"{name} ratio {ratio:.2%} exceeds {threshold:.2%}")

    def report_stable_sweep(self, amount_usd: float) -> None:
        """Record the USD amount swept in the last stable sweep."""

        self.stable_sweep_amount_usd.set(amount_usd)

    def increment_stable_blocked(self, n: int = 1) -> None:
        """Increment the counter for blocked stable sweeps."""

        self.stable_sweep_blocked_total.inc(n)

    def increment_stable_filled(self, n: int = 1) -> None:
        """Increment the counter for filled stable orders."""

        self.stable_orders_filled_total.inc(n)

    def heartbeat(self) -> None:
        """Update the heartbeat timestamp."""

        ts = time.time()
        self._heartbeat_ts = ts
        self._heartbeat_gauge.set(ts)

    def check_heartbeat(self, max_age_seconds: float) -> None:
        """Alert if the heartbeat has not been updated in ``max_age_seconds``."""

        if self._heartbeat_ts and time.time() - self._heartbeat_ts > max_age_seconds:
            self._send_alert("No heartbeat received")

    # ------------------------------------------------------------------
    # Alerting helpers
    # ------------------------------------------------------------------
    def _send_alert(self, message: str) -> None:
        """Send alert message via configured channels."""

        if self.slack_webhook:
            try:
                requests.post(self.slack_webhook, json={"text": message}, timeout=5)
            except Exception as exc:  # pragma: no cover - logging only
                logger.error("Failed to send Slack alert: %s", exc)
        if self.email_host and self.email_from and self.email_to:
            try:
                msg = EmailMessage()
                msg["Subject"] = "quant-pipeline alert"
                msg["From"] = self.email_from
                msg["To"] = self.email_to
                msg.set_content(message)
                with smtplib.SMTP(self.email_host) as smtp:
                    smtp.send_message(msg)
            except Exception as exc:  # pragma: no cover - logging only
                logger.error("Failed to send email alert: %s", exc)


__all__ = ["Observability"]
