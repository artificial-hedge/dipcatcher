"""Tests for models/banach_alaoglu.py — cluster point must be a real limit."""

from __future__ import annotations

import numpy as np


def test_cluster_is_a_sequence_element() -> None:
    """The Cesàro mean of an alternating ±1 sequence is 0 — a point no
    subsequence reaches, hence not a cluster point. The returned witness
    must be an actual element of the sequence."""
    from quant_fund.models.banach_alaoglu import weak_star_cluster

    alt = np.tile(np.array([[1.0, 0.0], [-1.0, 0.0]]), (8, 1))
    c = weak_star_cluster(alt)
    assert any(np.array_equal(c, row) for row in alt)


def test_cluster_constant_sequence() -> None:
    from quant_fund.models.banach_alaoglu import weak_star_cluster

    const = np.tile(np.array([0.5, -0.5]), (10, 1))
    assert np.allclose(weak_star_cluster(const), [0.5, -0.5])


def test_cluster_respects_unit_ball() -> None:
    from quant_fund.models.banach_alaoglu import weak_star_cluster

    rng = np.random.default_rng(0)
    seq = rng.uniform(-1, 1, (200, 4))
    seq = seq / np.maximum(np.linalg.norm(seq, axis=1, keepdims=True), 1.0)
    assert float(np.linalg.norm(weak_star_cluster(seq))) <= 1.0 + 1e-12
