"""Wave-239 programming-languages canon tests."""

from __future__ import annotations

from quant_fund.models.cps_transform import bench_cps_transform
from quant_fund.models.gc_marksweep import bench_gc_marksweep
from quant_fund.models.hm_inference import bench_hm_inference
from quant_fund.models.macro_expand import bench_macro_expand
from quant_fund.models.simple_types import bench_simple_types
from quant_fund.models.tree_walk_interp import bench_tree_walk_interp

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestHM:
    def test_bench(self) -> None:
        out = bench_hm_inference()
        _clean(out)
        assert out["synthetic_infers_principal"] == 1.0
        assert out["synthetic_occurs_rejects"] == 1.0


class TestInterp:
    def test_bench(self) -> None:
        out = bench_tree_walk_interp()
        _clean(out)
        assert out["synthetic_lexical_closure"] == 1.0
        assert out["synthetic_recursion"] == 1.0


class TestCPS:
    def test_bench(self) -> None:
        out = bench_cps_transform()
        _clean(out)
        assert out["synthetic_cps_equals_direct"] == 1.0


class TestMacro:
    def test_bench(self) -> None:
        out = bench_macro_expand()
        _clean(out)
        assert out["synthetic_when_correct"] == 1.0
        assert out["synthetic_and_lazy"] == 1.0


class TestGC:
    def test_bench(self) -> None:
        out = bench_gc_marksweep()
        _clean(out)
        assert out["synthetic_frees_unreachable"] == 1.0
        assert out["synthetic_copy_compacts"] == 1.0


class TestSTLC:
    def test_bench(self) -> None:
        out = bench_simple_types()
        _clean(out)
        assert out["synthetic_typed_accept"] == 1.0
        assert out["synthetic_typed_reject"] == 1.0
