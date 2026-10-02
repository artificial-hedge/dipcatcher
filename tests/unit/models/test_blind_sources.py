"""Blind source separation (SOBI, JADE, FOBI)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.blind_sources import (
    _match_corr,
    fobi,
    jade,
    sobi,
    whiten,
)


def _mixture(seed: int = 533):
    rng = np.random.default_rng(seed)
    t = np.arange(600)
    s = np.vstack(
        [
            np.sin(t * 0.05),
            np.sign(np.sin(t * 0.03)),
            rng.uniform(-1, 1, t.size),
        ]
    )
    a = rng.normal(0, 1, (3, 3))
    while abs(np.linalg.cond(a)) > 8:
        a = rng.normal(0, 1, (3, 3))
    return a @ s, s


def test_whiten_gives_identity_cov():
    x, _ = _mixture()
    z, _ = whiten(x)
    cov = z @ z.T / z.shape[1]
    assert np.allclose(cov, np.eye(3), atol=1e-8)


@pytest.mark.parametrize("fn", [sobi, jade, fobi])
def test_methods_separate_mixed_sources(fn):
    x, s = _mixture()
    est = np.asarray(fn(x)["sources"])
    scores = _match_corr(est, s)
    assert scores.min() > 0.85


def test_unmixing_recovers_inverse():
    x, s = _mixture()
    rng_a = x @ np.linalg.pinv(s)  # recover A = x s^+
    out = sobi(x)
    w = np.asarray(out["unmixing"])
    prod = w @ rng_a
    # each row should be dominated by one column
    dom = np.abs(prod).max(axis=1) / (np.abs(prod).sum(axis=1) + 1e-12)
    assert dom.min() > 0.8


def test_input_validation():
    with pytest.raises(ValueError):
        sobi(np.ones((2, 5)))
    with pytest.raises(ValueError):
        jade(np.full((3, 100), np.nan))
    with pytest.raises(ValueError):
        fobi(np.ones(10))
