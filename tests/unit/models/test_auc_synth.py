"""Unit tests for quant_fund.models._auc_synth."""

from __future__ import annotations

import numpy as np

from quant_fund.models._auc_synth import (
    GSP_CTR,
    GSP_VALUES,
    iid_values,
    uniform_bne_bid,
)


def test_iid_values_deterministic_and_bounded() -> None:
    a = iid_values(5, n=4, m=100)
    b = iid_values(5, n=4, m=100)
    assert np.array_equal(a, b)
    assert a.shape == (100, 4)
    assert np.all((a >= 0.0) & (a <= 1.0))


def test_bne_bid_is_shaded_and_linear() -> None:
    v = np.array([0.0, 0.5, 1.0])
    b = uniform_bne_bid(v, n=4)
    # U[0,1] first-price BNE: b(v) = (n-1)/n v = 0.75 v
    assert np.allclose(b, 0.75 * v)
    assert np.all(b <= v)


def test_gsp_fixture_sorted() -> None:
    assert np.all(np.diff(GSP_CTR) < 0)  # CTRs descending by slot
    assert np.all(np.diff(GSP_VALUES) < 0)  # values descending by bidder
