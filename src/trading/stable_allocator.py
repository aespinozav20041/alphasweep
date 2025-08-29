"""Stable allocation sweeping utilities."""

from __future__ import annotations

from datetime import date, datetime
import os
import csv
from pathlib import Path

import click

from .config import StableAllocConfig, load_stable_cfg
from . import ledger, brokers
from .market_calendar import is_market_open, next_market_open
from quant_pipeline.observability import Observability


DD_WEEKLY_THRESHOLD = float(os.getenv("STABLE_DD_WEEKLY_THRESHOLD", 0.08))
MIN_CASH_USD = float(os.getenv("STABLE_MIN_CASH_USD", 0.0))

REPORT_PATH = Path("reports/stable_sweep_daily.csv")


def _fetch_risk_data(day: date) -> tuple[float, float]:
    """Fetch risk metrics for the given date.

    Returns tuple ``(dd_weekly, cash_available)``. This function acts as a
    placeholder for a real risk/treasury service and can be monkeypatched in
    tests.
    """

    return 0.0, float("inf")


def _fetch_net_pnl(day: date) -> float:
    """Fetch net PnL for the given ``day``.

    This is a placeholder that can be monkeypatched in tests. In production it
    would query a treasury ledger or accounting system.
    """

    return 0.0


def _append_report(
    day: date,
    net_pnl: float,
    sweep_amount: float,
    ticker: str,
    status: str,
    order_id: str,
) -> None:
    """Append a sweep report row to ``REPORT_PATH``."""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not REPORT_PATH.exists()
    with REPORT_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date", "net_pnl", "sweep_amount", "ticker", "status", "order_id"])
        writer.writerow(
            [
                day.isoformat(),
                f"{net_pnl:.2f}",
                f"{sweep_amount:.2f}",
                ticker,
                status,
                order_id,
            ]
        )


def compute_sweep_amount(net_pnl_usd: float, cfg: StableAllocConfig) -> float:
    """Compute sweep amount based on net PnL and configuration.

    Parameters
    ----------
    net_pnl_usd: float
        Net profit and loss in USD.
    cfg: StableAllocConfig
        Stable allocation configuration.
    """
    net = max(net_pnl_usd, 0.0)
    amount = min(max(net * cfg.pct, 0.0), cfg.cap_daily_usd)
    return amount if amount >= cfg.min_usd else 0.0


def already_swept(day: date, ticker: str) -> bool:
    """Check if a sweep has already occurred for the given day and ticker."""
    return any(entry.date == day and entry.ticker == ticker for entry in ledger.entries)


def should_block_sweep(day: date) -> tuple[bool, str]:
    """Determine whether sweep should be blocked based on risk metrics.

    Parameters
    ----------
    day: date
        Evaluation day.
    Returns
    -------
    tuple[bool, str]
        ``(True, reason)`` if sweep must be blocked. Reason is "dd_weekly" or
        "cash_available". If not blocked, returns ``(False, "")``.
    """

    dd_weekly, cash_available = _fetch_risk_data(day)
    if dd_weekly > DD_WEEKLY_THRESHOLD:
        ledger.record(
            order_id=f"block-{day.isoformat()}",
            ticker="SWEEP",
            usd=0.0,
            status="blocked",
            day=day,
            note="dd_weekly",
        )
        return True, "dd_weekly"
    if cash_available < MIN_CASH_USD:
        ledger.record(
            order_id=f"block-{day.isoformat()}",
            ticker="SWEEP",
            usd=0.0,
            status="blocked",
            day=day,
            note="cash_available",
        )
        return True, "cash_available"
    return False, ""


def place_or_schedule(day: date, amount_usd: float, cfg: StableAllocConfig) -> None:
    """Place an order if the market is open, otherwise schedule it.

    Ensures idempotency by not duplicating planned or sent orders for the same
    ``day`` and ``ticker``.
    """

    if amount_usd <= 0:
        return

    # Idempotency: avoid duplicates if already actioned for the day
    for e in ledger.entries:
        if e.date == day and e.ticker == cfg.ticker and e.status in {
            "sent",
            "filled",
            "planned",
        }:
            return

    now = datetime.utcnow()
    if is_market_open(now, cfg.timezone):
        broker_mod = getattr(brokers, cfg.broker)
        if cfg.order_style == "market_open":
            order_id, status = broker_mod.place_market_on_open(cfg.ticker, amount_usd)
        elif cfg.order_style == "twap_30m":
            order_id, status = broker_mod.place_twap(cfg.ticker, amount_usd, minutes=30)
        elif cfg.order_style == "limit_vwap":
            order_id, status = broker_mod.place_limit_vwap(cfg.ticker, amount_usd)
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"unknown order_style {cfg.order_style}")
        for e in reversed(ledger.entries):
            if e.order_id == order_id:
                e.date = day
                break
    else:
        scheduled = next_market_open(now, cfg.timezone)
        ledger.record(
            order_id=f"plan-{day.isoformat()}",
            ticker=cfg.ticker,
            usd=amount_usd,
            status="planned",
            day=day,
            scheduled_at=scheduled,
        )


def run_scheduler(cfg: StableAllocConfig | None = None, obs: Observability | None = None) -> None:
    """Execute planned sweeps whose scheduled time has arrived."""

    if cfg is None:
        cfg = load_stable_cfg()

    broker_mod = getattr(brokers, cfg.broker)

    for entry in list(ledger.entries):
        if entry.status != "planned" or entry.scheduled_at is None:
            continue
        now_local = datetime.now(tz=entry.scheduled_at.tzinfo)
        if entry.scheduled_at > now_local:
            continue

        before = len(ledger.entries)
        if cfg.order_style == "market_open":
            order_id, status = broker_mod.place_market_on_open(entry.ticker, entry.usd)
        elif cfg.order_style == "twap_30m":
            order_id, status = broker_mod.place_twap(entry.ticker, entry.usd, minutes=30)
        elif cfg.order_style == "limit_vwap":
            order_id, status = broker_mod.place_limit_vwap(entry.ticker, entry.usd)
        else:  # pragma: no cover
            raise ValueError(f"unknown order_style {cfg.order_style}")

        if len(ledger.entries) > before:
            ledger.entries.pop()
        entry.order_id = order_id
        entry.status = status
        entry.scheduled_at = None
        if obs is not None:
            obs.report_stable_sweep(entry.usd)
            if status == "filled":
                obs.increment_stable_filled()


def perform_sweep(
    day: date, net_pnl: float, cfg: StableAllocConfig, obs: Observability | None = None
) -> None:
    """Core sweep logic used by CLI and tests."""

    amount = compute_sweep_amount(net_pnl, cfg)
    if already_swept(day, cfg.ticker):
        return
    blocked, _ = should_block_sweep(day)
    if blocked:
        entry = ledger.entries[-1]
        _append_report(day, net_pnl, 0.0, cfg.ticker, entry.status, entry.order_id)
        if obs is not None:
            obs.report_stable_sweep(0.0)
            obs.increment_stable_blocked()
        return
    place_or_schedule(day, amount, cfg)
    if not ledger.entries:
        return
    entry = ledger.entries[-1]
    _append_report(day, net_pnl, amount, cfg.ticker, entry.status, entry.order_id)
    if obs is not None:
        obs.report_stable_sweep(amount)
        if entry.status == "filled":
            obs.increment_stable_filled()


@click.group()
def cli() -> None:
    """Stable allocator command line interface."""


@cli.command()
@click.option("--date", "day", required=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--pnl", type=float, default=None, help="Net PnL override in USD")
@click.option("--dry-run", is_flag=True, help="Compute without executing or recording")
def sweep(day: datetime, pnl: float | None, dry_run: bool) -> None:
    """Calculate sweep amount and place or schedule the order."""

    cfg = load_stable_cfg()
    day_date = day.date()
    net_pnl = pnl if pnl is not None else _fetch_net_pnl(day_date)
    amount = compute_sweep_amount(net_pnl, cfg)

    if dry_run:
        click.echo(f"Would sweep {amount:.2f} USD")
        return

    perform_sweep(day_date, net_pnl, cfg)


@cli.command(name="run-scheduler")
def run_scheduler_cmd() -> None:
    """Execute any due planned sweeps."""

    run_scheduler()


if __name__ == "__main__":
    cli()
