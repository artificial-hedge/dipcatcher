"""Wave-217 estimation/filtering canon tests."""

from __future__ import annotations

from quant_fund.models.cubature_kalman import bench_cubature_kalman
from quant_fund.models.hinf_filter import bench_hinf_filter
from quant_fund.models.huber_filter import bench_huber_filter
from quant_fund.models.mhe import bench_mhe
from quant_fund.models.particle_smoother import bench_particle_smoother
from quant_fund.models.variational_bayes import bench_variational_bayes

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestHinf:
    def test_bench(self) -> None:
        out = bench_hinf_filter()
        _clean(out)
        assert out["synthetic_hinf_gain"] == 1.0


class TestCKF:
    def test_bench(self) -> None:
        out = bench_cubature_kalman()
        _clean(out)
        assert out["synthetic_ckf_rmse"] < out["synthetic_ckf_ekf_rmse"]


class TestMHE:
    def test_bench(self) -> None:
        out = bench_mhe()
        _clean(out)
        assert out["synthetic_mhe_gain"] == 1.0


class TestVB:
    def test_bench(self) -> None:
        out = bench_variational_bayes()
        _clean(out)
        assert out["synthetic_vb_err"] < 0.01


class TestHuber:
    def test_bench(self) -> None:
        out = bench_huber_filter()
        _clean(out)
        assert out["synthetic_hf_gain"] == 1.0


class TestSmoother:
    def test_bench(self) -> None:
        out = bench_particle_smoother()
        _clean(out)
        assert out["synthetic_ps_smooth_rmse"] <= out["synthetic_ps_filt_rmse"] + 0.1
