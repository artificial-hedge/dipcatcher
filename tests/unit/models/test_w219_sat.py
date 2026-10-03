"""Wave-219 SAT/symbolic-reasoning canon tests."""

from __future__ import annotations

from quant_fund.models.bdd_ops import bench_bdd_ops
from quant_fund.models.cdcl_solver import bench_cdcl_solver
from quant_fund.models.ltl_mc import bench_ltl_mc
from quant_fund.models.twosat_scc import bench_twosat_scc
from quant_fund.models.unit_propagation import bench_unit_propagation
from quant_fund.models.walksat import bench_walksat

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestCDCL:
    def test_bench(self) -> None:
        out = bench_cdcl_solver()
        _clean(out)
        assert out["synthetic_cdcl_agree"] == 1.0


class TestWalkSAT:
    def test_bench(self) -> None:
        out = bench_walksat()
        _clean(out)
        assert out["synthetic_ws_agree"] >= 0.95


class TestUnitProp:
    def test_bench(self) -> None:
        out = bench_unit_propagation()
        _clean(out)
        assert out["synthetic_up_consistent"] == 1.0


class TestTwoSat:
    def test_bench(self) -> None:
        out = bench_twosat_scc()
        _clean(out)
        assert out["synthetic_2sat_agree"] == 1.0


class TestBDD:
    def test_bench(self) -> None:
        out = bench_bdd_ops()
        _clean(out)
        assert out["synthetic_bdd_agree"] == out["synthetic_bdd_forms"]


class TestLTL:
    def test_bench(self) -> None:
        out = bench_ltl_mc()
        _clean(out)
        assert out["synthetic_ltl_states"] > 0
