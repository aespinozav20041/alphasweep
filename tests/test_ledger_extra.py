from datetime import date

from trading import ledger


def test_record_fill_and_pnl(tmp_path):
    ledger.clear()
    day = date(2024, 1, 1)
    ledger.record_fill("oid1", "AAPL", qty=10, price=5.0, day=day, note="test")
    ledger.record_pnl(100.0, day=day, model="m1")
    conn = ledger._connection()
    fill_rows = conn.execute("SELECT order_id, ticker, qty, price, usd, day, note FROM fills").fetchall()
    pnl_rows = conn.execute("SELECT day, pnl, model FROM pnl").fetchall()
    assert fill_rows[0]["order_id"] == "oid1"
    assert fill_rows[0]["usd"] == 50.0
    assert pnl_rows[0]["pnl"] == 100.0
    assert pnl_rows[0]["model"] == "m1"
    ledger.clear()
    assert conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM pnl").fetchone()[0] == 0
