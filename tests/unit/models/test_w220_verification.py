"""Wave-220 program-verification canon tests."""

from __future__ import annotations

from quant_fund.models.bmc_unroll import bench_bmc_unroll
from quant_fund.models.cegar_loop import bench_cegar_loop
from quant_fund.models.hoare_logic import bench_hoare_logic
from quant_fund.models.ic3_pdr import bench_ic3_pdr
from quant_fund.models.invariant_synth import bench_invariant_synth
from quant_fund.models.k_induction import bench_k_induction
from quant_fund.models.ranking_function import bench_ranking_function

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestKInduction:
    def test_bench(self) -> None:
        out = bench_k_induction()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_safe"] == 1.0


class TestIC3:
    def test_bench(self) -> None:
        out = bench_ic3_pdr()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_agree2"] == 1.0


class TestBMC:
    def test_bench(self) -> None:
        out = bench_bmc_unroll()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_hit_exact"] == 1.0
        assert out["synthetic_hit_shallow"] == 0.0


class TestInvSynth:
    def test_bench(self) -> None:
        out = bench_invariant_synth()
        _clean(out)
        assert out["synthetic_agree"] == 1.0


class TestHoare:
    def test_bench(self) -> None:
        out = bench_hoare_logic()
        _clean(out)
        assert out["synthetic_wlp_valid"] == 1.0
        assert out["synthetic_bad_rejected"] == 1.0


class TestRanking:
    def test_bench(self) -> None:
        out = bench_ranking_function()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_cycle_rejected"] == 1.0


class TestCEGAR:
    def test_bench(self) -> None:
        out = bench_cegar_loop()
        _clean(out)
        assert out["synthetic_agree"] == 1.0
        assert out["synthetic_agree2"] == 1.0
