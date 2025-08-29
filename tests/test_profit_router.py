from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from click.testing import CliRunner

import trading.profit_router as pr
import trading.stable_allocator as sa
from trading import ledger
from trading.config import StableAllocConfig


def test_settle_triggers_sweep_and_scheduler(monkeypatch):
    ledger.clear()
    cfg = StableAllocConfig(broker="mock")
    monkeypatch.setattr(sa, "load_stable_cfg", lambda: cfg)

    # yesterday's PnL is positive
    yesterday = date.today() - timedelta(days=1)

    def fake_pnl(day: date) -> float:
        assert day == yesterday
        return 100.0

    monkeypatch.setattr(sa, "_fetch_net_pnl", fake_pnl)
    monkeypatch.setattr(sa, "is_market_open", lambda now, tz: False)
    past = datetime.now(ZoneInfo(cfg.timezone)) - timedelta(minutes=1)
    monkeypatch.setattr(sa, "next_market_open", lambda now, tz: past)

    runner = CliRunner()

    def fake_run(cmd, check):
        # execute the sweep command via click instead of spawning a process
        assert cmd[-2] == "--date"
        res = runner.invoke(sa.cli, ["sweep", "--date", cmd[-1]])
        assert res.exit_code == 0

    monkeypatch.setattr(pr.subprocess, "run", fake_run)

    pr.settle(date.today())
    assert len(ledger.entries) == 1
    assert ledger.entries[0].status == "planned"

    # run scheduler to fill
    monkeypatch.setattr(sa.brokers.mock, "place_market_on_open", lambda t, u: ("oid", "filled"))
    sa.run_scheduler(cfg)
    assert ledger.entries[0].status == "filled"
