"""Ingestion service integrating validation and metrics."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from quant_pipeline.doctor import validate_ohlcv
from quant_pipeline.ingest import ingest_ohlcv_ccxt
from quant_pipeline.observability import Observability
from quant_pipeline.storage import read_table


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tf_to_ms(timeframe: str) -> int:
    units = {"m": 60, "h": 60 * 60, "d": 24 * 60 * 60}
    unit = timeframe[-1]
    if unit not in units:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return int(timeframe[:-1]) * units[unit] * 1000


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------


class IngestService:
    """Fetch data from providers, validate and store it."""

    def __init__(self, lake_path: Path | None = None, observability: Observability | None = None) -> None:
        self.lake_path = lake_path or Path(__file__).resolve().parents[2] / "lake"
        self.obs = observability or Observability()

    def ingest_ccxt(
        self,
        exchange: str,
        symbols: Iterable[str],
        *,
        timeframe: str = "1h",
        since: Optional[int] = None,
        until: Optional[int] = None,
        db_path: Optional[Path] = None,
    ) -> None:
        """Ingest OHLCV data via CCXT and run quality checks."""

        start = time.time()
        db_path = db_path or Path(__file__).resolve().parents[2] / "data" / "ingest.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ingest_ohlcv_ccxt(
            exchange,
            symbols,
            timeframe=timeframe,
            since=since,
            until=until,
            out=self.lake_path,
            db_path=db_path,
        )
        tf_ms = _tf_to_ms(timeframe)
        end_ts = until or int(time.time() * 1000)

        for sym in symbols:
            canon = sym.replace("/", "-")
            df = read_table(
                "ohlcv",
                symbols=[canon],
                start=since or 0,
                end=end_ts,
                timeframe=timeframe,
                base_path=self.lake_path,
            )
            validated = validate_ohlcv(df, tf_ms)
            diffs = validated["timestamp"].diff().dropna()
            missing = (diffs > tf_ms).sum()
            missing_ratio = missing / max(len(diffs), 1)
            dup_ratio = df["timestamp"].duplicated().sum() / len(df)
            self.obs.report_missing_bars(canon, timeframe, missing_ratio)
            self.obs.report_duplicate_ratio(canon, dup_ratio)

        latency_ms = (time.time() - start) * 1000.0
        self.obs.observe_latency(latency_ms)


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------

def schedule_job(interval_seconds: float, job: Callable, *args, **kwargs) -> threading.Event:
    """Run ``job`` every ``interval_seconds`` seconds in a background thread."""

    stop_event = threading.Event()

    def _runner() -> None:
        while not stop_event.is_set():
            job(*args, **kwargs)
            if stop_event.wait(interval_seconds):
                break

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    return stop_event


__all__ = ["IngestService", "schedule_job"]
