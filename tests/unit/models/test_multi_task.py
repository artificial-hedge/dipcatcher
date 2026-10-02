import numpy as np

from quant_fund.models.multi_task import (
    bench_multi_task,
    mtl_ridge,
    mtl_shared,
)


def _tasks(seed=0):
    rng = np.random.default_rng(seed)
    d, t = 30, 4
    shared = np.arange(8)
    w = np.zeros((d, t))
    base = rng.standard_normal(len(shared)) * 1.5
    xs, ys = [], []
    for i in range(t):
        w[shared, i] = base
        xs.append(rng.standard_normal((25, d)))
        ys.append(xs[-1] @ w[:, i] + 0.2 * rng.standard_normal(25))
    return xs, ys, w, shared


def test_mtl_beats_ind():
    xs, ys, w, _ = _tasks()
    wi = mtl_ridge(xs, ys, lam=5e-3)
    wm, _ = mtl_shared(xs, ys, lam=5e-3, lam_share=20.0, it=60)
    x = np.random.default_rng(9).standard_normal((500, 30))
    ei = np.mean([np.mean((x @ wi[:, i] - x @ w[:, i]) ** 2) for i in range(4)])
    em = np.mean([np.mean((x @ wm[:, i] - x @ w[:, i]) ** 2) for i in range(4)])
    assert em < ei


def test_bench_keys():
    out = bench_multi_task()
    assert out["synthetic_mtl_gain"] > 0.0
    assert out["synthetic_mtl_shared_hit"] >= 10
