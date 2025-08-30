from pathlib import Path

from src.cc.champion_challenger import ChampionChallenger, ModelMetrics


def test_promotion(tmp_path: Path) -> None:
    cc = ChampionChallenger(1.0, 0.5, 0.2, out_dir=tmp_path)
    cc.register("m1", ModelMetrics(1.1, 1.2, 0.6, 0.1))
    cc.register("m2", ModelMetrics(0.8, 0.7, 0.4, 0.3))
    champ = cc.evaluate()
    assert champ == "m1"
    assert (tmp_path / "promotion_report.md").exists()
