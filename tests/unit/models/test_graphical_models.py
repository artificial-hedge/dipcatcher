import numpy as np

from quant_fund.models.graphical_models import (
    bench_graphical_models,
    cpt_fit,
    infer_marginal,
    learn_bayes_net,
)


def test_chain_skeleton_recovered():
    rng = np.random.default_rng(0)
    n = 2000
    a = (rng.uniform(0, 1, n) < 0.5).astype(np.int64)
    b = (rng.uniform(0, 1, n) < np.where(a == 1, 0.9, 0.1)).astype(np.int64)
    data = np.c_[a, b]
    parents = learn_bayes_net(data, max_parents=1, it=30)
    edges = {frozenset((p, j)) for j, ps in enumerate(parents) for p in ps}
    assert frozenset((0, 1)) in edges


def test_marginal_matches_empirical():
    rng = np.random.default_rng(1)
    n = 2000
    a = (rng.uniform(0, 1, n) < 0.3).astype(np.int64)
    data = np.c_[a]
    parents = learn_bayes_net(data, max_parents=1, it=10)
    cpts = cpt_fit(data, parents)
    marg = infer_marginal(cpts, parents, 0)
    assert abs(marg[1] - a.mean()) < 0.05


def test_bench_graphical_models():
    out = bench_graphical_models(seed=559)
    assert out["synthetic_bn_edges_found"] == 2
    assert out["synthetic_bn_marg_err"] < 0.1
