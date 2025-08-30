import pytest
from execution import CostModel, FeeSchedule


def test_cost_model_uses_fee_schedule():
    schedule = FeeSchedule(
        fees={
            "binance": {
                "vip0": {"maker": 1.0, "taker": 2.0},
                "vip1": {"maker": 0.5, "taker": 1.5},
            }
        }
    )
    model = CostModel(fee_schedule=schedule)
    price = 100.0
    qty = 1.0

    maker_cost = model.apply(price, qty, side="maker", exchange="binance", tier="vip1")
    taker_cost = model.apply(price, qty, side="taker", exchange="binance", tier="vip0")

    expected_maker = price * qty + price * qty * 0.5 / 10_000
    expected_taker = price * qty + price * qty * 2.0 / 10_000

    assert maker_cost == pytest.approx(expected_maker)
    assert taker_cost == pytest.approx(expected_taker)
