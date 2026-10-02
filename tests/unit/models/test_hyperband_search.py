import numpy as np

from quant_fund.models.hyperband_search import (
    bench_hyperband,
    hyperband,
    successive_halving,
)


def _sample(rng):
    return {"x": float(rng.uniform())}


def _loss(cfg, budget):
    return abs(cfg["x"] - 0.5)


def test_sha_returns_config():
    r = successive_halving(_sample, _loss, n=27, budget=81, seed=0)
    assert "x" in r["config"]
    assert isinstance(r["loss"], float)


def test_hb_beats_coarse_grid():
    hb = hyperband(_sample, _loss, max_iter=27, eta=3, seed=0)
    cfg = hb["config"]
    assert isinstance(cfg["x"], float)


def test_hb_finds_good_region():
    r = hyperband(_sample, _loss, max_iter=81, eta=3, seed=7)
    assert abs(r["config"]["x"] - 0.5) < 0.3


def test_bench_hyperband_runs():
    out = bench_hyperband(seed=545)
    assert np.isfinite(out["synthetic_hb_best_loss"])
    assert out["synthetic_hb_x_err"] < 0.5
