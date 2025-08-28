"""Command line interface for quant-pipeline."""

import click

from pathlib import Path

import pandas as pd

from .ingest import ingest_ohlcv_ccxt

from .quality import doctor_ohlcv
from .logging import get_logger
from .settings import settings
from .storage import LAKE_PATH
=======
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



=======


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



@cli.command("doctor-ohlcv")
@click.option("--symbol", required=True, type=str)
@click.option("--timeframe", required=True, type=str)
@click.option("--start", required=True, type=str)
@click.option("--end", required=True, type=str)
@click.option("--gap-threshold", default=0.01, show_default=True)
@click.option("--lake", type=click.Path(), default=str(LAKE_PATH), show_default=True)
@click.option("--runs", type=click.Path(), default=str(Path(LAKE_PATH).parents[1] / "runs"), show_default=True)
def doctor_ohlcv_cmd(symbol, timeframe, start, end, gap_threshold, lake, runs):
    """Validate and repair OHLCV data, writing cleaned bars."""
    start_ms = int(pd.Timestamp(start, tz="UTC").value // 1_000_000)
    end_ms = int(pd.Timestamp(end, tz="UTC").value // 1_000_000)
    doctor_ohlcv(
        symbol,
        timeframe,
        start_ms,
        end_ms,
        gap_threshold=gap_threshold,
        lake_path=Path(lake),
        runs_path=Path(runs),
    )



if __name__ == "__main__":
    cli()
