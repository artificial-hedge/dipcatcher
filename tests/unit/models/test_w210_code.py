"""Wave-210 coding-theory canon tests."""

from __future__ import annotations

from quant_fund.models.bch_code import bench_bch_code
from quant_fund.models.conv_interleaver import bench_conv_interleaver
from quant_fund.models.crc_check import bench_crc_check
from quant_fund.models.ldpc_decoder import bench_ldpc_decoder
from quant_fund.models.polar_code import bench_polar_code
from quant_fund.models.turbo_decoder import bench_turbo_decoder

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestLDPC:
    def test_bench(self) -> None:
        out = bench_ldpc_decoder()
        _clean(out)
        assert out["synthetic_ldpc_ber_gain"] >= 0.0
        assert out["synthetic_ldpc_syndrome"] == 0.0


class TestTurbo:
    def test_bench(self) -> None:
        out = bench_turbo_decoder()
        _clean(out)
        assert out["synthetic_turbo_ber"] <= out["synthetic_turbo_single_ber"]


class TestPolar:
    def test_bench(self) -> None:
        out = bench_polar_code()
        _clean(out)
        assert out["synthetic_polar_gain"] > 0.0
        assert out["synthetic_polar_info"] == 8.0


class TestBCH:
    def test_bench(self) -> None:
        out = bench_bch_code()
        _clean(out)
        assert out["synthetic_bch_single_frac"] == 1.0
        assert out["synthetic_bch_double_frac"] == 1.0
        assert out["synthetic_bch_clean_ok"] == 1.0


class TestCRC:
    def test_bench(self) -> None:
        out = bench_crc_check()
        _clean(out)
        assert out["synthetic_crc32_kat"] == 1.0
        assert out["synthetic_crc16_kat"] == 1.0
        assert out["synthetic_crc32_single_detect"] == 1.0


class TestInterleaver:
    def test_bench(self) -> None:
        out = bench_conv_interleaver()
        _clean(out)
        assert out["synthetic_int_gain"] > 0.0
