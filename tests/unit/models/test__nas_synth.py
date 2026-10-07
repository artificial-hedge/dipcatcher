"""Probes for _nas_synth."""

import numpy as np
import pytest

from quant_fund.models._nas_synth import all_archs, noisy_labels, rand_arch


def test_rand_arch_within_space():
    rng = np.random.default_rng(0)
    space = set(all_archs())
    for _ in range(20):
        assert rand_arch(rng) in space


def test_rand_arch_deterministic():
    rng0 = np.random.default_rng(9)
    a = [rand_arch(rng0) for _ in range(5)]
    rng1 = np.random.default_rng(9)
    rng2 = np.random.default_rng(9)
    seq1 = [rand_arch(rng1) for _ in range(5)]
    seq2 = [rand_arch(rng2) for _ in range(5)]
    assert seq1 == seq2
    assert a == seq1


def test_noisy_labels_flips_exact_frac():
    y = np.zeros(100, dtype=np.int64)
    out = noisy_labels(y, seed=0, frac=0.3)
    assert int((out != y).sum()) == 30
    assert set(np.unique(out)) <= {0, 1}
    np.testing.assert_array_equal(y, np.zeros(100, dtype=np.int64))


@pytest.mark.parametrize(
    "y,frac",
    [(np.zeros(0, dtype=np.int64), 0.1), (np.array([0, 1, 2]), 0.1)],
)
def test_noisy_labels_hostile(y, frac):
    with pytest.raises(ValueError):
        noisy_labels(y, 0, frac)


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan])
def test_noisy_labels_bad_frac(bad):
    with pytest.raises(ValueError):
        noisy_labels(np.zeros(4, dtype=np.int64), 0, bad)


def test_build_net_rejects_bad_arch():
    torch = pytest.importorskip("torch")
    from quant_fund.models._nas_synth import build_net

    for arch in [(4, 1, 2), (0, 1, 0), (4, 0, 0), (-2, 1, 1)]:
        with pytest.raises(ValueError):
            build_net(torch, arch)


def test_eval_arch_rejects_vacuous():
    from quant_fund.models._nas_synth import eval_arch

    for kw in [{"iters": 0}, {"iters": -1}, {"lr": 0.0}, {"lr": np.inf}]:
        with pytest.raises(ValueError):
            eval_arch((4, 1, 0), None, None, None, None, **kw)
