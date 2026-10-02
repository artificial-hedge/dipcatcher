"""Wave-216 advanced-MC-sampling canon tests."""

from __future__ import annotations

from quant_fund.models.metadynamics import bench_metadynamics
from quant_fund.models.parallel_tempering import bench_parallel_tempering
from quant_fund.models.thermo_integration import bench_thermo_integration
from quant_fund.models.umbrella_sampling import bench_umbrella_sampling
from quant_fund.models.wang_landau import bench_wang_landau
from quant_fund.models.wham import bench_wham

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestPT:
    def test_bench(self) -> None:
        out = bench_parallel_tempering()
        _clean(out)
        assert out["synthetic_pt_err"] < out["synthetic_pt_sc_err"]


class TestWL:
    def test_bench(self) -> None:
        out = bench_wang_landau()
        _clean(out)
        assert out["synthetic_wl_logcorr"] > 0.95


class TestUS:
    def test_bench(self) -> None:
        out = bench_umbrella_sampling()
        _clean(out)
        assert out["synthetic_us_err"] < 0.01


class TestMeta:
    def test_bench(self) -> None:
        out = bench_metadynamics()
        _clean(out)
        assert out["synthetic_md_crosses"] > out["synthetic_md_base_crosses"]


class TestWHAM:
    def test_bench(self) -> None:
        out = bench_wham()
        _clean(out)
        assert out["synthetic_wham_err"] < 0.5


class TestTI:
    def test_bench(self) -> None:
        out = bench_thermo_integration()
        _clean(out)
        assert out["synthetic_ti_err"] < 0.2
