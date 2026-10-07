"""Tests for models/approx_ndcg_ltr.py — soft-NDCG ideal must sort gains."""

from __future__ import annotations

import numpy as np

from quant_fund.models.approx_ndcg_ltr import bench_approx_ndcg_ltr


def test_soft_ndcg_perfect_ranking_is_one() -> None:
    """A near-perfect soft ranking must score ~1, not >1: the ideal DCG
    uses descending-sorted gains."""
    from quant_fund.models.approx_ndcg_ltr import _soft_ndcg, _torch

    torch = _torch()
    s = torch.tensor([[0.0, 0.0, 10.0]])  # ranks doc2 first
    gains = torch.tensor([[0.0, 3.0, 7.0]])  # doc2 is truly best
    ndcg = _soft_ndcg(s, gains, 1.0, torch)
    # an unsorted ideal would report ~1.26 here
    assert float(ndcg[0]) <= 1.0 + 1e-5


def test_ndcg_never_exceeds_one() -> None:
    """The ideal-DCG denominator must sort gains descending; unsorted
    discounting lets NDCG exceed 1."""
    out = bench_approx_ndcg_ltr(iters=40)
    assert out["synthetic_andcg_ndcg10"] <= 1.0 + 1e-6
    assert out["synthetic_andcg_base_ndcg10"] <= 1.0 + 1e-6


def test_ideal_dcg_sorted_manually() -> None:
    """Reference check: canonical NDCG ideal uses sorted gains; an
    unsorted ideal inflates the ratio past 1 on permuted labels."""
    gains = np.array([0.0, 1.0, 3.0, 7.0])
    pos = 1.0 / np.log2(np.arange(2, 6))
    unsorted_ideal = float((gains * pos).sum())
    sorted_ideal = float((np.sort(gains)[::-1] * pos).sum())
    assert unsorted_ideal < sorted_ideal
    assert sorted_ideal / unsorted_ideal > 1.5  # the defect magnitude


def test_bench_deterministic() -> None:
    a = bench_approx_ndcg_ltr(iters=5)
    b = bench_approx_ndcg_ltr(iters=5)
    assert a == b
