from pathlib import Path

from src.stress.test_harness import StressScenario, run_stress_scenarios


def test_sharpe_drops(tmp_path: Path) -> None:
    scenarios = [
        StressScenario("baseline"),
        StressScenario("wider", spread_factor=3.0),
    ]
    df = run_stress_scenarios({}, scenarios, out_dir=tmp_path)
    base = df[df.scenario == "baseline"].sharpe.iloc[0]
    wide = df[df.scenario == "wider"].sharpe.iloc[0]
    assert wide < base
    assert (tmp_path / "stress_summary.md").exists()
