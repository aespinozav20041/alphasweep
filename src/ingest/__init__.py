"""High level data ingestion utilities.

This module orchestrates downloading market data from external providers,
validates it with the Data Doctor, and exposes simple scheduling helpers.
"""

from .service import IngestService, schedule_job

__all__ = ["IngestService", "schedule_job"]
