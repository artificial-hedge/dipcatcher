"""Unit tests for quant_fund.models._dfx_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._dfx_synth import gauss_mmd


def test_gauss_mmd_deterministic_and_nonnegative() -> None:
    rng = np.random.default_rng(0)
    Xte = rng.standard_normal((100, 3))
    a = gauss_mmd(np.random.default_rng(1), Xte, n=50)
    b = gauss_mmd(np.random.default_rng(1), Xte, n=50)
    assert a == b
    assert a >= 0.0


def test_gauss_mmd_rejects_empty_draw() -> None:
    Xte = np.zeros((10, 2))
    with pytest.raises(ValueError, match="n must be >= 1"):
        gauss_mmd(np.random.default_rng(0), Xte, n=0)


def test_gauss_mmd_same_dist_small() -> None:
    # MMD between two normal draws should be small relative to shifted data
    rng = np.random.default_rng(0)
    Xte_same = rng.standard_normal((300, 2))
    Xte_shift = rng.standard_normal((300, 2)) + 5.0
    near = gauss_mmd(np.random.default_rng(2), Xte_same, n=300)
    far = gauss_mmd(np.random.default_rng(2), Xte_shift, n=300)
    assert far > near
