"""Wave-227 compression canon tests."""

from __future__ import annotations

from quant_fund.models.arithmetic_coding import bench_arithmetic_coding
from quant_fund.models.golomb_rice import bench_golomb_rice
from quant_fund.models.huffman_codes import bench_huffman_codes
from quant_fund.models.lz78_dict import bench_lz78_dict
from quant_fund.models.lzw_compress import bench_lzw_compress
from quant_fund.models.rans_coder import bench_rans_coder

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestHuffman:
    def test_bench(self) -> None:
        out = bench_huffman_codes()
        _clean(out)
        assert out["synthetic_roundtrip"] == 1.0
        assert out["synthetic_prefix_free"] == 1.0
        assert out["synthetic_within_bit"] == 1.0


class TestArithmetic:
    def test_bench(self) -> None:
        out = bench_arithmetic_coding()
        _clean(out)
        assert out["synthetic_roundtrip"] == 1.0
        assert out["synthetic_near_entropy"] == 1.0
        assert out["synthetic_mean_overhead_bits"] < 1.0


class TestLZW:
    def test_bench(self) -> None:
        out = bench_lzw_compress()
        _clean(out)
        assert out["synthetic_roundtrip"] == 1.0
        assert out["synthetic_beats_literal"] == 1.0


class TestGolomb:
    def test_bench(self) -> None:
        out = bench_golomb_rice()
        _clean(out)
        assert out["synthetic_roundtrip"] == 1.0
        assert out["synthetic_near_entropy"] == 1.0


class TestRANS:
    def test_bench(self) -> None:
        out = bench_rans_coder()
        _clean(out)
        assert out["synthetic_roundtrip"] == 1.0
        assert out["synthetic_near_entropy"] == 1.0


class TestLZ78:
    def test_bench(self) -> None:
        out = bench_lz78_dict()
        _clean(out)
        assert out["synthetic_roundtrip"] == 1.0
        assert out["synthetic_valid_indices"] == 1.0
        assert out["synthetic_edge_cases"] == 1.0
