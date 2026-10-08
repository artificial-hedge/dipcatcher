"""Tests for concrete_dropout — concrete mask keep-probability."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_concrete_keep_unbiased():
    # E[keep] must be ~1 for an unbiased dropout mask at drop rate p.
    # The old offset log(p) keeps with probability ~p (the drop rate
    # itself, inverted) giving E[keep] ~= p/(1-p) = 0.43 at p=0.3.
    from quant_fund.models.concrete_dropout import _concrete_keep

    g = torch.Generator().manual_seed(0)
    p = torch.tensor(0.3)
    u = torch.rand((400000,), generator=g)
    keep = _concrete_keep(u, p, 0.067)
    assert 0.9 < float(keep.mean()) < 1.1


def test_concrete_keep_rate_matches_one_minus_p():
    # the fraction of near-1 mask entries approximates the keep prob 1-p.
    from quant_fund.models.concrete_dropout import _concrete_keep

    g = torch.Generator().manual_seed(1)
    p = torch.tensor(0.25)
    u = torch.rand((400000,), generator=g)
    conc = _concrete_keep(u, p, 1e-3) * (1 - p)
    assert abs(float((conc > 0.5).float().mean()) - 0.75) < 0.02


def test_bench_smoke():
    pytest.importorskip("torch")
    from quant_fund.models.concrete_dropout import bench_concrete_dropout

    out = bench_concrete_dropout(iters=60, T=8)
    assert out["synthetic_cd_nll"] == out["synthetic_cd_nll"]  # finite, not NaN
    assert 0.0 < out["synthetic_cd_learned_p"] < 1.0
