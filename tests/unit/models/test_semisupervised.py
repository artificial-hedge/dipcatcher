import numpy as np

from quant_fund.models.semisupervised import (
    bench_semisupervised,
    label_propagation,
    self_training,
)


def _moons(seed=0, n=200, noise=0.08):
    rng = np.random.default_rng(seed)
    cls = (rng.uniform(0, 1, n) > 0.5).astype(np.float64)
    t = rng.uniform(0, np.pi, n)
    x = np.zeros((n, 2))
    m0 = cls == 0
    x[m0, 0] = np.cos(t[m0])
    x[m0, 1] = np.sin(t[m0])
    x[~m0, 0] = 1.0 - np.cos(t[~m0])
    x[~m0, 1] = -np.sin(t[~m0]) + 0.5
    x += noise * rng.standard_normal(x.shape)
    return x, cls


def test_label_prop_moons():
    x, y = _moons()
    rng = np.random.default_rng(3)
    lab = np.zeros(len(y), bool)
    lab[rng.permutation(len(y))[:20]] = True
    f = label_propagation(x, y, lab)
    acc = np.mean((f[~lab] >= 0.5) == (y[~lab] >= 0.5))
    assert acc > 0.85


def test_self_training_runs():
    x, y = _moons()
    rng = np.random.default_rng(5)
    lab = np.zeros(len(y), bool)
    lab[rng.permutation(len(y))[:40]] = True
    p = self_training(x, y, lab, threshold=0.9)
    assert np.isfinite(p).all()
    assert np.all((p >= 0) & (p <= 1))


def test_bench_keys():
    out = bench_semisupervised()
    assert out["synthetic_labelprop_acc"] > 0.9
    assert out["synthetic_lp_gain"] > 0.0
