import numpy as np
import pandas as pd

from quant_pipeline.backtest import GENE_NAMES, run_backtest
from quant_pipeline.genetic import GeneticOptimizer


def test_run_backtest_with_genes():
    df = pd.DataFrame({"ret": [0.01, -0.02, 0.03, -0.01]})
    genes = [1, 16, 0.1, 5, 1, 0.001, 0.01, 0.0, 0.5, 1]
    assert len(genes) == len(GENE_NAMES)
    pnl = run_backtest(df, genes)
    assert isinstance(pnl, float)


def test_turnover_penalty():
    df = pd.DataFrame({"ret": [0.01, -0.02, 0.03, -0.01]})
    genes = [1, 16, 0.1, 5, 1, 0.001, 0.01, 0.0, 0.0, 0]
    pnl_raw = run_backtest(df, genes)
    pnl_penalised = run_backtest(df, genes, turnover_penalty=1.0)
    assert pnl_penalised <= pnl_raw


def test_genetic_optimizer_vector_bounds():
    def fitness(x):
        return -(x[0] - 0.5) ** 2 - (x[1] - 0.5) ** 2

    bounds = [(0, 1), (0, 1)]
    opt = GeneticOptimizer(fitness, bounds, population_size=20, rng=np.random.default_rng(0))
    best, score = opt.optimise(generations=5, patience=3)
    assert best.shape == (2,)
    assert 0 <= best[0] <= 1 and 0 <= best[1] <= 1

