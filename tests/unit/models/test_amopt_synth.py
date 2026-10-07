"""Unit tests for quant_fund.models._amopt_synth."""

from __future__ import annotations

import numpy as np

from quant_fund.models._amopt_synth import (
    S0,
    K,
    amopt_params,
    crr_price,
    european_put,
    gbm_path,
)


def test_gbm_path_deterministic_and_positive() -> None:
    p1 = gbm_path(5, steps=50)
    p2 = gbm_path(5, steps=50)
    p3 = gbm_path(6, steps=50)
    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, p3)
    assert np.all(p1 > 0)


def test_american_geq_european_put() -> None:
    # early exercise can only add value: the honest-negative baseline
    # must never exceed the American price.
    am, boundary = crr_price(400)
    eu = european_put()
    assert eu > 0.0
    assert am >= eu - 1e-6
    assert boundary.shape == (401,)


def test_boundary_nonincreasing_and_below_strike() -> None:
    _, boundary = crr_price(400)
    assert np.all(boundary <= K + 1e-9)
    assert np.all(boundary > 0.0)
    assert np.all(boundary <= S0 * 10)


def test_params_match_module_constants() -> None:
    p = amopt_params()
    assert p["S0"] == S0 and p["K"] == K
