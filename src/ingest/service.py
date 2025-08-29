"""Ingestion service integrating validation and metrics."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from quant_pipeline.doctor import validate_ohlcv
from quant_pipeline.ingest import (
    CorporateMacroAdapter,
    ingest_ohlcv_ccxt,
    ingest_orderbook,
    ingest_news,
)
from quant_pipeline.observability import Observability
from quant_pipeline.storage import read_table, to_parquet


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

    def ingest_orderbook(
        self,
        fetcher: Callable[[str, str, Optional[int], Optional[int]], "pd.DataFrame"],
        symbols: Iterable[str],
        *,
        timeframe: str = "1s",
        since: Optional[int] = None,
        until: Optional[int] = None,
    ) -> None:
        """Ingest order book snapshots using ``fetcher`` adapter."""

        start = time.time()
        ingest_orderbook(
            fetcher,
            symbols,
            timeframe=timeframe,
            start=since,
            end=until,
            out=self.lake_path,
        )
        end_ts = until or int(time.time() * 1000)

        for sym in symbols:
            canon = sym.replace("/", "-")
            df = read_table(
                "orderbook_best",
                symbols=[canon],
                start=since or 0,
                end=end_ts,
                timeframe=timeframe,
                base_path=self.lake_path,
            )
            if not df["timestamp"].is_monotonic_increasing:
                raise ValueError("timestamps not monotonic")
            depth_missing = (
                (df["bid_sz1"] <= 0) | (df["ask_sz1"] <= 0)
            ).sum()
            depth_ratio = depth_missing / max(len(df), 1)
            dup_ratio = df["timestamp"].duplicated().sum() / len(df)
            self.obs.report_missing_bars(canon, timeframe, depth_ratio)
            self.obs.report_duplicate_ratio(canon, dup_ratio)

        latency_ms = (time.time() - start) * 1000.0
        self.obs.observe_latency(latency_ms)

    def ingest_news(
        self,
        provider: str,
        symbols: Iterable[str],
        *,
        timeframe: str = "1h",
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """Ingest news/sentiment data from ``provider``."""

        start_ts = time.time()
        ingest_news(
            provider,
            symbols,
            start=start,
            end=end,
            timeframe=timeframe,
            out=self.lake_path,
        )
        end_ts = end or int(time.time() * 1000)

        for sym in symbols:
            canon = sym.replace("/", "-")
            df = read_table(
                "news",
                symbols=[canon],
                start=start or 0,
                end=end_ts,
                timeframe=timeframe,
                base_path=self.lake_path,
            )
            if not df["timestamp"].is_monotonic_increasing:
                raise ValueError("timestamps not monotonic")
            dup_ratio = df["timestamp"].duplicated().sum() / len(df)
            self.obs.report_duplicate_ratio(canon, dup_ratio)

        latency_ms = (time.time() - start_ts) * 1000.0
        self.obs.observe_latency(latency_ms)

    def ingest_corporate(
        self,
        adapter: CorporateMacroAdapter,
        symbols: Iterable[str],
        *,
        timeframe: str = "1d",
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """Ingest corporate actions using ``adapter``."""

        start_ts = time.time()
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            df = adapter.fetch_corporate(sym, start=start, end=end)
            if df.empty:
                continue
            if "timestamp" not in df.columns:
                raise ValueError("payload must contain 'timestamp'")
            df = df.copy()
            df["timestamp"] = df["timestamp"].astype("int64")
            df["symbol"] = sym
            df["source"] = adapter.__class__.__name__
            df["timeframe"] = timeframe
            frames.append(df)

        if not frames:
            return

        out_df = pd.concat(frames, ignore_index=True)
        out_df = out_df.drop_duplicates(subset=["timestamp", "symbol"]).sort_values(
            "timestamp"
        )
        to_parquet(out_df, "corporate", base_path=self.lake_path)

        tf_ms = _tf_to_ms(timeframe)
        for sym in symbols:
            sym_df = out_df[out_df["symbol"] == sym]
            if sym_df.empty:
                continue
            diffs = sym_df["timestamp"].diff().dropna()
            missing = (diffs > tf_ms).sum()
            missing_ratio = missing / max(len(diffs), 1)
            dup_ratio = sym_df["timestamp"].duplicated().sum() / len(sym_df)
            self.obs.report_missing_bars(sym, timeframe, missing_ratio)
            self.obs.report_duplicate_ratio(sym, dup_ratio)

        latency_ms = (time.time() - start_ts) * 1000.0
        self.obs.observe_latency(latency_ms)

    def ingest_macro(
        self,
        adapter: CorporateMacroAdapter,
        indicators: Iterable[str],
        *,
        timeframe: str = "1d",
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """Ingest macroeconomic indicators using ``adapter``."""

        start_ts = time.time()
        frames: list[pd.DataFrame] = []
        for ind in indicators:
            df = adapter.fetch_macro(ind, start=start, end=end)
            if df.empty:
                continue
            required = {"timestamp", "value"}
            if not required.issubset(df.columns):
                missing = required - set(df.columns)
                raise ValueError(f"missing columns: {missing}")
            df = df.copy()
            df = df.astype({"timestamp": "int64", "value": "float64"})
            df["symbol"] = ind
            df["source"] = adapter.__class__.__name__
            df["timeframe"] = timeframe
            frames.append(df)

        if not frames:
            return

        out_df = pd.concat(frames, ignore_index=True)
        out_df = out_df.drop_duplicates(subset=["timestamp", "symbol"]).sort_values(
            "timestamp"
        )
        to_parquet(out_df, "macro", base_path=self.lake_path)

        tf_ms = _tf_to_ms(timeframe)
        for ind in indicators:
            ind_df = out_df[out_df["symbol"] == ind]
            if ind_df.empty:
                continue
            diffs = ind_df["timestamp"].diff().dropna()
            missing = (diffs > tf_ms).sum()
            missing_ratio = missing / max(len(diffs), 1)
            dup_ratio = ind_df["timestamp"].duplicated().sum() / len(ind_df)
            self.obs.report_missing_bars(ind, timeframe, missing_ratio)
            self.obs.report_duplicate_ratio(ind, dup_ratio)

        latency_ms = (time.time() - start_ts) * 1000.0
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
