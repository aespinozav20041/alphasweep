"""Prometheus metrics for trading dashboards."""

from __future__ import annotations

from prometheus_client import Gauge, Histogram, Counter, start_http_server

PNL = Gauge("pnl", "Profit and Loss")
EXPOSURE = Gauge("exposure", "Current market exposure")
HIT_RATE = Gauge("hit_rate", "Trade hit rate")
DRAWDOWN = Gauge("drawdown", "Max drawdown")
FILLS = Counter("fills", "Number of order fills")
ORDERS = Counter("orders", "Number of orders sent")
LATENCY = Histogram("latency_seconds", "Order latency in seconds")
ORDER_QUEUE = Gauge("order_queue", "Current length of the order queue")


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server."""

    start_http_server(port)


__all__ = [
    "PNL",
    "EXPOSURE",
    "HIT_RATE",
    "DRAWDOWN",
    "FILLS",
    "ORDERS",
    "LATENCY",
    "ORDER_QUEUE",
    "start_metrics_server",
]
