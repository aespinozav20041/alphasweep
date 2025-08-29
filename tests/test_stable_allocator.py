from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from trading import ledger
from trading.config import StableAllocConfig
import trading.stable_allocator as sa
from click.testing import CliRunner
from trading.stable_allocator import (
    compute_sweep_amount,
    already_swept,
    should_block_sweep,
    place_or_schedule,
    perform_sweep,
)
from quant_pipeline.observability import Observability
from fastapi.testclient import TestClient


def test_compute_sweep_amount_defaults():
    cfg = StableAllocConfig()
    assert compute_sweep_amount(1000, cfg) == 500
    assert compute_sweep_amount(60, cfg) == 0
    assert compute_sweep_amount(100000, cfg) == cfg.cap_daily_usd


def test_already_swept_checks_ledger():
    ledger.clear()
    today = date.today()
    iid = ledger.record_intent("SPY", 100, day=today)
    ledger.record_result(iid, "oid", "filled", day=today)
    assert already_swept(today, "SPY")
    assert not already_swept(today, "VOO")


def test_should_block_sweep_dd_weekly(monkeypatch):
    ledger.clear()
    today = date.today()

    def fake_fetch(day, obs=None):
        return 0.10, 1000.0  # dd_weekly, cash_available

    monkeypatch.setattr(sa, "_fetch_risk_data", fake_fetch)
    monkeypatch.setattr(sa, "DD_WEEKLY_THRESHOLD", 0.08)
    monkeypatch.setattr(sa, "MIN_CASH_USD", 0.0)

    blocked, reason = should_block_sweep(today)
    assert blocked and reason == "dd_weekly"
    assert any(e.status == "blocked" and e.note == "dd_weekly" for e in ledger.entries())


def test_place_or_schedule_plans_when_closed(monkeypatch):
    ledger.clear()
    today = date.today()
    cfg = StableAllocConfig()

    monkeypatch.setattr(sa, "is_market_open", lambda now, tz: False)
    next_open = datetime(2024, 1, 1, 9, 30, tzinfo=ZoneInfo(cfg.timezone))
    monkeypatch.setattr(sa, "next_market_open", lambda now, tz: next_open)

    place_or_schedule(today, 100.0, cfg)
    entries = ledger.entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.status == "planned"
    assert entry.scheduled_at == next_open

    # Idempotency: second call should not add another entry
    place_or_schedule(today, 100.0, cfg)
    assert len(ledger.entries()) == 1


def test_cli_sweep_and_scheduler(monkeypatch):
    ledger.clear()
    today = date.today()
    cfg = StableAllocConfig(broker="mock")
    monkeypatch.setattr(sa, "load_stable_cfg", lambda: cfg)
    monkeypatch.setattr(sa, "is_market_open", lambda now, tz: False)
    past = datetime.now(ZoneInfo(cfg.timezone)) - timedelta(minutes=1)
    monkeypatch.setattr(sa, "next_market_open", lambda now, tz: past)

    runner = CliRunner()
    res = runner.invoke(sa.cli, ["sweep", "--date", today.isoformat(), "--pnl", "100"])
    assert res.exit_code == 0
    entries = ledger.entries()
    assert len(entries) == 1
    assert entries[0].status == "planned"

    def fake_place(ticker, usd, obs=None, *, day=None, intent_id=None):
        iid = intent_id or ledger.record_intent(ticker, usd, day=day)
        ledger.record_result(iid, "mock-oid", "filled", day=day)
        return "mock-oid", "filled"

    monkeypatch.setattr(sa.brokers.mock, "place_market_on_open", fake_place)
    res = runner.invoke(sa.cli, ["run-scheduler"])
    assert res.exit_code == 0
    entries = ledger.entries()
    assert len(entries) == 2
    e = entries[-1]
    assert e.status == "filled"
    assert e.order_id == "mock-oid"


def test_report_and_metrics(monkeypatch, tmp_path):
    ledger.clear()
    cfg = StableAllocConfig(broker="mock")
    obs = Observability()
    monkeypatch.setattr(sa, "REPORT_PATH", tmp_path / "stable_sweep_daily.csv")
    monkeypatch.setattr(sa, "is_market_open", lambda now, tz: True)
    day = date.today()
    perform_sweep(day, 100.0, cfg, obs)

    report_file = tmp_path / "stable_sweep_daily.csv"
    content = report_file.read_text().strip().splitlines()
    assert content[0].startswith("date,net_pnl,sweep_amount,ticker,status,order_id")
    row = content[1].split(",")
    assert row[0] == day.isoformat()
    assert row[1] == "100.00"
    assert row[2] == "50.00"
    assert row[3] == cfg.ticker
    assert row[4] == "filled"
    assert row[5].startswith("mock-")

    client = TestClient(obs.app)
    metrics = client.get("/metrics").text
    assert "stable_sweep_amount_usd" in metrics
    assert "stable_orders_filled_total 1.0" in metrics
    assert "stable_sweep_blocked_total" in metrics


def test_fetch_risk_data_failure(monkeypatch):
    def fake_get(self, url, params=None, headers=None, timeout=None):
        raise sa.requests.Timeout("boom")

    monkeypatch.setattr(sa.requests.Session, "get", fake_get)
    obs = Observability()
    called: dict[str, str] = {}
    monkeypatch.setattr(obs, "_send_alert", lambda msg: called.setdefault("msg", msg))
    dd, cash = sa._fetch_risk_data(date.today(), obs)
    assert dd == 0.0 and cash == float("inf")
    assert "risk_data_fetch_error" in called["msg"]


def test_fetch_net_pnl_failure(monkeypatch):
    def fake_get(self, url, params=None, headers=None, timeout=None):
        raise sa.requests.ConnectionError("boom")

    monkeypatch.setattr(sa.requests.Session, "get", fake_get)
    obs = Observability()
    called: dict[str, str] = {}
    monkeypatch.setattr(obs, "_send_alert", lambda msg: called.setdefault("msg", msg))
    pnl = sa._fetch_net_pnl(date.today(), obs)
    assert pnl == 0.0
    assert "net_pnl_fetch_error" in called["msg"]
