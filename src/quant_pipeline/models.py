"""Pydantic data models for canonical tables."""

from pydantic import BaseModel, validator
import math


class BarOHLCV(BaseModel):
    timestamp: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    timeframe: str

    @validator("timestamp")
    def ts_ms(cls, v: int) -> int:
        if v < 0:
            raise ValueError("timestamp must be positive milliseconds")
        return v

    @validator("open", "high", "low", "close")
    def no_nan(cls, v: float) -> float:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            raise ValueError("OHLC values cannot be NaN")
        return v

    @validator("volume", pre=True, always=True)
    def volume_default(cls, v) -> float:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 0.0
        return v


class OrderBookBest(BaseModel):
    timestamp: int
    symbol: str
    bid1: float
    ask1: float
    bid_sz1: float
    ask_sz1: float
    trades_buy_vol: float
    trades_sell_vol: float
    source: str
    timeframe: str

    @validator("timestamp")
    def ts_ms(cls, v: int) -> int:
        if v < 0:
            raise ValueError("timestamp must be positive milliseconds")
        return v


class PerpMetrics(BaseModel):
    timestamp: int
    symbol: str
    funding_rate: float
    open_interest: float
    basis: float
    liquidations_usd: float
    source: str
    timeframe: str

    @validator("timestamp")
    def ts_ms(cls, v: int) -> int:
        if v < 0:
            raise ValueError("timestamp must be positive milliseconds")
        return v


class NewsSentiment(BaseModel):
    timestamp: int
    symbol: str
    sentiment: float
    source: str
    timeframe: str

    @validator("timestamp")
    def ts_ms(cls, v: int) -> int:
        if v < 0:
            raise ValueError("timestamp must be positive milliseconds")
        return v

    @validator("sentiment")
    def sentiment_range(cls, v: float) -> float:
        if v < -1 or v > 1:
            raise ValueError("sentiment must be between -1 and 1")
        return v
