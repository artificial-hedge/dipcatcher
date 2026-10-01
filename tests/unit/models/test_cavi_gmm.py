import numpy as np
import pytest

from quant_fund.models.cavi_gmm import bench_cavi_gmm, cavi_gmm, synth_gmm


def test_recovers_means() -> None:
    d = synth_gmm(seed=0)
    out = cavi_gmm(np.asarray(d["x"]), k=2, seed=0)
    mu = np.asarray(out["mu"])
    assert abs(mu[0] + 1.5) < 0.3 and abs(mu[1] - 1.5) < 0.3


def test_elbo_increases() -> None:
    d = synth_gmm(seed=1)
    out = cavi_gmm(np.asarray(d["x"]), k=2, seed=1)
    elbo = np.asarray(out["elbo"])
    assert elbo[-1] > elbo[0]
    diffs = np.diff(elbo)
    assert np.all(diffs > -0.01)


def test_pi_recovery() -> None:
    d = synth_gmm(seed=2)
    out = cavi_gmm(np.asarray(d["x"]), k=2, seed=2)
    pi = np.asarray(out["pi"])
    assert abs(pi[0] - 0.4) < 0.15


def test_three_components() -> None:
    d = synth_gmm(
        n=2000,
        mus=(-2.0, 0.0, 2.5),
        sigmas=(0.4, 0.5, 0.4),
        pis=(0.3, 0.4, 0.3),
        seed=3,
    )
    out = cavi_gmm(np.asarray(d["x"]), k=3, seed=3)
    mu = np.asarray(out["mu"])
    assert abs(mu[0] + 2.0) < 0.35 and abs(mu[2] - 2.5) < 0.35


def test_responsibilities_sum_to_one() -> None:
    d = synth_gmm(n=300, seed=4)
    out = cavi_gmm(np.asarray(d["x"]), k=2, seed=4)
    r = np.asarray(out["r"])
    assert np.allclose(r.sum(axis=1), 1.0)


def test_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        cavi_gmm(rng.normal(0, 1, 50), k=2)
    with pytest.raises(ValueError):
        cavi_gmm(rng.normal(0, 1, 200) * np.nan, k=2)
    with pytest.raises(ValueError):
        cavi_gmm(rng.normal(0, 1, 200), k=1)
    with pytest.raises(ValueError):
        cavi_gmm(rng.normal(0, 1, 200), k=9)


def test_bench() -> None:
    out = bench_cavi_gmm()
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
    assert all(np.isfinite(v) for v in out.values())
