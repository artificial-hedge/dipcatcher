import numpy as np

from quant_fund.models.self_organizing_maps import (
    bench_som,
    lvq_predict,
    lvq_train,
    som_bmus,
    som_train,
    u_matrix,
)


def test_som_weights_shape():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 2))
    r = som_train(x, 5, 6, it=50, seed=0)
    assert np.asarray(r["weights"]).shape == (30, 2)
    assert np.asarray(r["coords"]).shape == (30, 2)


def test_som_bmus_in_range():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 2))
    r = som_train(x, 4, 4, it=50, seed=1)
    b = som_bmus(np.asarray(r["weights"]), x)
    assert b.min() >= 0 and b.max() < 16


def test_u_matrix_shape():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 2))
    r = som_train(x, 4, 5, it=30, seed=2)
    u = u_matrix(np.asarray(r["weights"]), 4, 5)
    assert u.shape == (4, 5)
    assert (u >= 0).all()


def test_lvq_separates_blobs():
    rng = np.random.default_rng(3)
    x = np.vstack([rng.normal([0, 0], 0.3, (40, 2)), rng.normal([4, 4], 0.3, (40, 2))])
    y = np.r_[np.zeros(40), np.ones(40)].astype(np.int64)
    mdl = lvq_train(x, y, n_proto=3, it=200, seed=3)
    acc = (lvq_predict(mdl, x) == y).mean()
    assert acc > 0.95


def test_bench_som_runs():
    out = bench_som(seed=543)
    assert out["synthetic_som_qe"] < out["synthetic_som_qe_random"]
    assert out["synthetic_lvq_acc"] > 0.9
