"""Wave-209 reliability-engineering canon tests."""

from __future__ import annotations

from quant_fund.models.fault_tree import bench_fault_tree
from quant_fund.models.fmea_rpn import bench_fmea_rpn
from quant_fund.models.life_stress import bench_life_stress
from quant_fund.models.ram_markov import bench_ram_markov
from quant_fund.models.redundancy_block import bench_redundancy_block
from quant_fund.models.weibull_life import bench_weibull_life

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestWeibullLife:
    def test_bench(self) -> None:
        out = bench_weibull_life()
        _clean(out)
        assert out["synthetic_wb_beta_err"] < 0.5
        assert out["synthetic_wb_eta_rel"] < 0.2


class TestFaultTree:
    def test_bench(self) -> None:
        out = bench_fault_tree()
        _clean(out)
        assert out["synthetic_ft_top_err"] < 1e-9
        assert out["synthetic_ft_mc_err"] < 0.02


class TestRAMMarkov:
    def test_bench(self) -> None:
        out = bench_ram_markov()
        _clean(out)
        assert out["synthetic_ram_avail"] > 0.9
        assert out["synthetic_ram_avail_mc_err"] < 0.02


class TestFMEA:
    def test_bench(self) -> None:
        out = bench_fmea_rpn()
        _clean(out)
        assert out["synthetic_fmea_top_is_mount"] == 1.0
        assert out["synthetic_fmea_rpn_range"] > 0.0


class TestLifeStress:
    def test_bench(self) -> None:
        out = bench_life_stress()
        _clean(out)
        assert out["synthetic_arr_ea_err"] < 0.1
        assert out["synthetic_arr_use_rel"] < 0.3


class TestRBD:
    def test_bench(self) -> None:
        out = bench_redundancy_block()
        _clean(out)
        assert out["synthetic_rbd_series_mc_err"] < 0.02
        assert out["synthetic_rbd_par_over_ser"] > 0.0
