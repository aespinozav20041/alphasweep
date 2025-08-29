"""Market calendar utilities for NYSE sessions.

Provides helpers to determine whether the market is open at a given
time and when the next session will start.  If ``pandas_market_calendars``
is available it is used to account for holidays and special schedules; if
not, a simple 9:30–16:00 Eastern time schedule excluding weekends is used.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

try:  # pragma: no cover - optional dependency
    import pandas_market_calendars as mcal

    _NY_CAL = mcal.get_calendar("NYSE")
except Exception:  # pragma: no cover - fallback when library missing
    mcal = None
    _NY_CAL = None


def _to_local(now: datetime, tz: str) -> datetime:
    """Return ``now`` converted to the timezone ``tz``."""

    tzinfo = ZoneInfo(tz)
    return now.astimezone(tzinfo) if now.tzinfo else now.replace(tzinfo=tzinfo)


def is_market_open(now: datetime, tz: str = "America/New_York") -> bool:
    """Return ``True`` if the NYSE is trading at ``now``.

    Parameters
    ----------
    now: datetime
        The time to evaluate.
    tz: str
        Timezone for interpretation of ``now`` and market hours.
    """

    now_local = _to_local(now, tz)

    if _NY_CAL is not None:
        schedule = _NY_CAL.schedule(
            start_date=now_local.date(), end_date=now_local.date(), tz=tz
        )
        if schedule.empty:
            return False
        open_dt = schedule.iloc[0]["market_open"]
        close_dt = schedule.iloc[0]["market_close"]
        return open_dt <= now_local < close_dt

    open_time = time(9, 30)
    close_time = time(16, 0)
    return (
        now_local.weekday() < 5
        and open_time <= now_local.time() < close_time
    )


def next_market_open(now: datetime, tz: str = "America/New_York") -> datetime:
    """Return the next NYSE market open time after ``now``.

    The returned ``datetime`` is timezone-aware in ``tz``.
    """

    now_local = _to_local(now, tz)

    if _NY_CAL is not None:
        schedule = _NY_CAL.schedule(
            start_date=now_local.date(),
            end_date=(now_local + pd.Timedelta(days=7)).date(),
            tz=tz,
        )
        future = schedule[schedule["market_open"] > now_local]
        if future.empty:
            # Should not happen for the 7-day window but fall back if it does
            next_open = schedule.iloc[-1]["market_open"]
        else:
            next_open = future.iloc[0]["market_open"]
        return next_open.to_pydatetime()

    open_time = time(9, 30)
    today_open = now_local.replace(
        hour=open_time.hour, minute=open_time.minute, second=0, microsecond=0
    )
    if now_local.weekday() < 5 and now_local < today_open:
        return today_open

    candidate = now_local + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.replace(
        hour=open_time.hour, minute=open_time.minute, second=0, microsecond=0
    )


__all__ = ["is_market_open", "next_market_open"]

