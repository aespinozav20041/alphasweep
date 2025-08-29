import json
import numpy as np

from quant_pipeline.model_registry import ModelRegistry
from quant_pipeline.training import AutoTrainer


def _mk_files(tmp_path, tag):
    art = tmp_path / f"{tag}.bin"
    calib = tmp_path / f"{tag}.calib"
    art.write_text(tag)
    calib.write_text(tag)
    return str(art), str(calib)


def test_autotrainer_genetic_flow(tmp_path):
    """AutoTrainer should evaluate genes via GA and register each model."""
    db = tmp_path / "reg.db"
    reg = ModelRegistry(str(db))

    gene_bounds = [(0.0, 1.0), (0.0, 1.0)]
    pop = 3
    gens = 2
    trained = []

    def build_dataset(days):
        return {"days": days}

    def train_model(_, genes):
        idx = len(trained)
        art, calib = _mk_files(tmp_path, f"m{idx}")
        trained.append(genes)
        score = -((genes[0] - 0.5) ** 2 + (genes[1] - 0.5) ** 2)
        return {
            "type": "lstm",
            "genes_json": json.dumps(list(genes)),
            "artifact_path": art,
            "calib_path": calib,
            "score": score,
        }

    def fitness(info):
        return info["score"]

    trainer = AutoTrainer(
        reg,
        train_every_bars=1,
        history_days=1,
        max_challengers=pop * (gens + 1),
        build_dataset=build_dataset,
        train_model=train_model,
        gene_bounds=gene_bounds,
        fitness_fn=fitness,
        ga_generations=gens,
        ga_population=pop,
        ga_rng=np.random.default_rng(0),
    )

    trainer._train_cycle()

    models = reg.list_models(status="challenger")
    assert len(models) == pop * (gens + 1)
    assert len(trained) == pop * (gens + 1)
    for m in models:
        genes = json.loads(m["genes_json"])
        assert 0 <= genes[0] <= 1
        assert 0 <= genes[1] <= 1
