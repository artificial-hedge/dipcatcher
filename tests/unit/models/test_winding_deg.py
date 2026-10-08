"""Winding-degree probes."""

from __future__ import annotations

import numpy as np

from quant_fund.models.winding_deg import bench_winding_deg, degree


def test_degree_counts_signed_wraps():
    for k in (-4, -2, -1, 0, 1, 2, 3, 4):
        f = lambda t, k=k: k * t + 0.4 + 0.2 * np.sin(3 * t)  # noqa: E731
        assert degree(f) == k


def test_degree_public_helper_not_mixed_units():
    """Former bug added (f(2pi)-f(0))/(2pi) into a radian sum."""
    assert degree(lambda t: 2.0 * t) == 2
    assert degree(lambda t: -1.0 * t + 0.5) == -1


def test_bench():
    assert bench_winding_deg()["synthetic_winding_deg"] == 1.0
