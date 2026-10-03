"""Wave-213 SDP/relaxation canon tests."""

from __future__ import annotations

from quant_fund.models.eigenvalue_opt import bench_eigenvalue_opt
from quant_fund.models.hoffman_bound import bench_hoffman_bound
from quant_fund.models.qcqp_relax import bench_qcqp_relax
from quant_fund.models.sdp_maxcut import bench_sdp_maxcut
from quant_fund.models.sos_certificate import bench_sos_certificate
from quant_fund.models.spectral_bisection import bench_spectral_bisection

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestSDPMaxCut:
    def test_bench(self) -> None:
        out = bench_sdp_maxcut()
        _clean(out)
        assert out["synthetic_gw_ratio"] >= 0.878


class TestEigOpt:
    def test_bench(self) -> None:
        out = bench_eigenvalue_opt()
        _clean(out)
        assert out["synthetic_eo_lam"] <= out["synthetic_eo_grid_lb"] + 1e-6


class TestSOS:
    def test_bench(self) -> None:
        out = bench_sos_certificate()
        _clean(out)
        assert out["synthetic_sos_residual"] < 1e-8
        assert out["synthetic_sos_p_min"] > 0.0


class TestQCQP:
    def test_bench(self) -> None:
        out = bench_qcqp_relax()
        _clean(out)
        assert out["synthetic_qcqp_sdp_bound"] >= out["synthetic_qcqp_truth"] - 1e-6


class TestSpectralBisect:
    def test_bench(self) -> None:
        out = bench_spectral_bisection()
        _clean(out)
        assert out["synthetic_sb_cut"] >= out["synthetic_sb_truth"] - 1e-9
        assert out["synthetic_sb_fiedler"] > 0.0


class TestHoffman:
    def test_bench(self) -> None:
        out = bench_hoffman_bound()
        _clean(out)
        assert out["synthetic_hb_bound"] >= out["synthetic_hb_alpha"] - 1e-9
        assert out["synthetic_hb_valid"] == 1.0
