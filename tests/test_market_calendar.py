"""Tests for market calendar utilities."""

from datetime import datetime
from zoneinfo import ZoneInfo

from trading.market_calendar import is_market_open, next_market_open


def test_weekend_and_next_open():
    tz = "America/New_York"
    saturday = datetime(2023, 9, 9, 12, 0, tzinfo=ZoneInfo(tz))
    assert is_market_open(saturday, tz) is False

    monday_open = next_market_open(saturday, tz)
    assert monday_open == datetime(2023, 9, 11, 9, 30, tzinfo=ZoneInfo(tz))

