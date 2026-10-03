"""Unit tests for wave-285 information-theory canon modules."""

import numpy as np

from quant_fund.models.blahut_arimoto import _bsc, _h2, capacity
from quant_fund.models.elias_gamma import delta_decode, delta_encode, gamma_decode, gamma_encode
from quant_fund.models.kl_knn import kl_nn
from quant_fund.models.markov_entropy import _analytic
from quant_fund.models.miller_madow import miller_madow
from quant_fund.models.type_class import empirical_prob


def test_markov_iid_limit():
    assert abs(_analytic(0.5, 0.5) - 1.0) < 1e-9


def test_markov_sticky():
    assert _analytic(0.99, 0.99) < 0.1


def test_ba_bsc():
    assert abs(capacity(_bsc(0.11)) - (1 - _h2(0.11))) < 5e-3


def test_kl_same():
    rng = np.random.RandomState(0)
    x, y = rng.randn(2000), rng.randn(2000)
    assert abs(kl_nn(x, y)) < 0.15


def test_type_prob():
    # n=2 fair ternary: all 6 types sum to 1
    p = np.array([1 / 3, 1 / 3, 1 / 3])
    tot = sum(
        empirical_prob(p, np.array([a, b, 2 - a - b])) for a in range(3) for b in range(3 - a)
    )
    assert abs(tot - 1.0) < 1e-12


def test_elias_roundtrip():
    assert gamma_decode(gamma_encode(17)) == (17, len(gamma_encode(17)))
    assert delta_decode(delta_encode(37)) == (37, len(delta_encode(37)))


def test_miller_madow_beats_plugin():
    rng = np.random.RandomState(1)
    cnt = np.bincount(rng.randint(0, 10, 20), minlength=10)
    assert miller_madow(cnt, 10) > 0
