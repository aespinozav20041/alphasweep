"""Command line interface for quant-pipeline."""

import click

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


if __name__ == "__main__":
    cli()
