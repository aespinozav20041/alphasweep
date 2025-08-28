"""Command line interface for quant-pipeline."""

import click

from pathlib import Path

import pandas as pd

from .ingest import ingest_ohlcv_ccxt
from .logging import get_logger
from .settings import settings
from .storage import LAKE_PATH
=======
from .logging import get_logger
from .settings import settings


logger = get_logger(__name__)


@click.group()
def cli():
    """Quant Pipeline CLI."""


@cli.command()
def info():
    """Display current settings."""

    click.echo(f"Environment: {settings.env_name}")



@cli.command("ingest-ohlcv")
@click.option("--exchange", required=True, type=str)
@click.option("--symbols", required=True, type=str, help="Comma separated symbols")
@click.option("--timeframe", default="1h", show_default=True)
@click.option("--since", type=str)
@click.option("--until", type=str)
@click.option("--out", type=click.Path(), default=str(LAKE_PATH), show_default=True)
def ingest_ohlcv_cmd(exchange, symbols, timeframe, since, until, out):
    """Download and store OHLCV data from CCXT exchanges."""

    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    since_ms = (
        int(pd.Timestamp(since, tz="UTC").value // 1_000_000) if since else None
    )
    until_ms = (
        int(pd.Timestamp(until, tz="UTC").value // 1_000_000) if until else None
    )
    ingest_ohlcv_ccxt(
        exchange,
        sym_list,
        timeframe=timeframe,
        since=since_ms,
        until=until_ms,
        out=Path(out),
    )



if __name__ == "__main__":
    cli()
