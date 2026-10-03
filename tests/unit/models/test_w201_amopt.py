"""Wave-201 stochastic-control / American-option canon tests."""

from __future__ import annotations

from quant_fund.models.crr_tree import bench_crr_tree
from quant_fund.models.dual_american import bench_dual_american
from quant_fund.models.exercise_boundary import bench_exercise_boundary
from quant_fund.models.hjb_penalty import bench_hjb_penalty
from quant_fund.models.kushner_mca import bench_kushner_mca
from quant_fund.models.psor_american import bench_psor_american

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestPSOR:
    def test_bench(self) -> None:
        out = bench_psor_american()
        _clean(out)
        assert out["synthetic_psor_err"] < 0.05
        assert out["synthetic_psor_err"] < out["synthetic_psor_eur_floor_err"]


class TestCRR:
    def test_bench(self) -> None:
        out = bench_crr_tree()
        _clean(out)
        assert out["synthetic_crr_err"] < 0.05
        assert out["synthetic_crr_price"] > 0.0


class TestKushnerMCA:
    def test_bench(self) -> None:
        out = bench_kushner_mca()
        _clean(out)
        assert out["synthetic_mca_err"] < 0.05


class TestHJBPenalty:
    def test_bench(self) -> None:
        out = bench_hjb_penalty()
        _clean(out)
        assert out["synthetic_hjb_err"] < 0.05


class TestDual:
    def test_bench(self) -> None:
        out = bench_dual_american()
        _clean(out)
        assert out["synthetic_dual_err"] < 0.2


class TestBoundary:
    def test_bench(self) -> None:
        out = bench_exercise_boundary()
        _clean(out)
        assert out["synthetic_boundary_l2"] < out["synthetic_boundary_naive_l2"]
