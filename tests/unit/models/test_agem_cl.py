"""Tests for A-GEM projection (models/agem_cl.py)."""

import numpy as np


def test_projected_gradient_respects_memory():
    from quant_fund.models.agem_cl import _agem_projected

    # g clips to (-50, 50) which still violates g·gref < 0 — the projection
    # must run AFTER the clip or the non-interference guarantee silently
    # re-breaks (the defect this guards).
    gref = np.array([1.0, 0.001])
    g = np.array([-60.0, 60.0])
    ga = _agem_projected(g, gref)
    assert float(ga @ gref) >= -1e-9


def test_projected_gradient_no_memory():
    from quant_fund.models.agem_cl import _agem_projected

    g = np.array([-100.0, 100.0])
    np.testing.assert_allclose(_agem_projected(g, None), np.clip(g, -50, 50))


def test_projected_gradient_already_aligned():
    from quant_fund.models.agem_cl import _agem_projected

    gref = np.array([1.0, 0.0])
    g = np.array([2.0, -1.0])
    np.testing.assert_allclose(_agem_projected(g, gref), g)
