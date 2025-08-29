"""Profit routing utilities.

This module exposes a ``settle`` function that is expected to run after the
trading day closes.  Once settlement bookkeeping is complete, the function
triggers the stable sweep logic for the previous day via the CLI of
:mod:`trading.stable_allocator`.

The sweep is executed as a subprocess using ``python -m trading.stable_allocator``
so that the same command can be reused by external schedulers or operators.
"""

from __future__ import annotations

from datetime import date, timedelta
import subprocess
import click


def settle(day: date | None = None) -> None:
    """Finalize allocations for ``day`` and trigger the stable sweep."""

    if day is None:
        day = date.today()

    # Placeholder for real allocation logic. In production this function would
    # record ``allocation_hype`` or other accounting entries here.

    sweep_day = day - timedelta(days=1)
    subprocess.run(
        [
            "python",
            "-m",
            "trading.stable_allocator",
            "sweep",
            "--date",
            sweep_day.isoformat(),
        ],
        check=True,
    )


@click.group()
def cli() -> None:
    """Profit router command line interface."""


@cli.command()
@click.option("--date", "day", type=click.DateTime(formats=["%Y-%m-%d"]), default=None)
def settle_cmd(day: date | None) -> None:
    """Settle allocations for the given day (defaults to today)."""

    settle(day.date() if day else None)


if __name__ == "__main__":  # pragma: no cover
    cli()
