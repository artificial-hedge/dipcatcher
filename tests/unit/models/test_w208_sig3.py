"""Wave-208 signal-processing-3 canon tests."""

from __future__ import annotations

from quant_fund.models.cepstrum_pitch import bench_cepstrum_pitch
from quant_fund.models.cwt_ridge import bench_cwt_ridge
from quant_fund.models.goertzel_detect import bench_goertzel_detect
from quant_fund.models.hilbert_instant import bench_hilbert_instant
from quant_fund.models.lpc_formant import bench_lpc_formant
from quant_fund.models.mvdr_beamformer import bench_mvdr_beamformer

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestCWTRidge:
    def test_bench(self) -> None:
        out = bench_cwt_ridge()
        _clean(out)
        assert out["synthetic_cwt_med_err"] < 0.1


class TestCepstrum:
    def test_bench(self) -> None:
        out = bench_cepstrum_pitch()
        _clean(out)
        assert out["synthetic_cep_err"] < 5.0


class TestMVDR:
    def test_bench(self) -> None:
        out = bench_mvdr_beamformer()
        _clean(out)
        assert out["synthetic_mvdr_constraint_err"] < 1e-6
        assert out["synthetic_mvdr_null_gain"] > 0.0


class TestHilbert:
    def test_bench(self) -> None:
        out = bench_hilbert_instant()
        _clean(out)
        assert out["synthetic_hil_med_err"] < 0.5
        assert out["synthetic_hil_env_mean"] > 0.5


class TestLPC:
    def test_bench(self) -> None:
        out = bench_lpc_formant()
        _clean(out)
        assert out["synthetic_lpc_f1_err"] < 80.0
        assert out["synthetic_lpc_f2_err"] < 80.0


class TestGoertzel:
    def test_bench(self) -> None:
        out = bench_goertzel_detect()
        _clean(out)
        assert out["synthetic_goe_941_err"] < 0.1
        assert out["synthetic_goe_top1"] == 941.0
        assert out["synthetic_goe_top2"] == 1336.0
