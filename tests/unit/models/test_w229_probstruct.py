"""Wave-229 probabilistic-membership canon tests."""

from __future__ import annotations

from quant_fund.models.bloom_filter import bench_bloom_filter
from quant_fund.models.cuckoo_filter import bench_cuckoo_filter
from quant_fund.models.minhash_lsh import bench_minhash_lsh
from quant_fund.models.quotient_filter import bench_quotient_filter
from quant_fund.models.simhash import bench_simhash
from quant_fund.models.xor_filter import bench_xor_filter

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _clean(out: dict[str, float]) -> None:
    assert not any(b in k.lower() for k in out for b in _FORBIDDEN)
    assert all(v == v and abs(v) < 1e18 for v in out.values())


class TestBloom:
    def test_bench(self) -> None:
        out = bench_bloom_filter()
        _clean(out)
        assert out["synthetic_no_fn"] == 1.0
        assert out["synthetic_fpr_within"] == 1.0


class TestCuckoo:
    def test_bench(self) -> None:
        out = bench_cuckoo_filter()
        _clean(out)
        assert out["synthetic_no_fn"] == 1.0
        assert out["synthetic_delete_ok"] == 1.0
        assert out["synthetic_fpr_bound"] == 1.0


class TestXor:
    def test_bench(self) -> None:
        out = bench_xor_filter()
        _clean(out)
        assert out["synthetic_no_fn"] == 1.0
        assert out["synthetic_fpr_bound"] == 1.0


class TestQuotient:
    def test_bench(self) -> None:
        out = bench_quotient_filter()
        _clean(out)
        assert out["synthetic_no_fn"] == 1.0
        assert out["synthetic_fpr_bound"] == 1.0
        assert out["synthetic_delete_ok"] == 1.0


class TestMinHash:
    def test_bench(self) -> None:
        out = bench_minhash_lsh()
        _clean(out)
        assert out["synthetic_err_ok"] == 1.0
        assert out["synthetic_recall_high_j"] == 1.0


class TestSimHash:
    def test_bench(self) -> None:
        out = bench_simhash()
        _clean(out)
        assert out["synthetic_near_detects"] == 1.0
        assert out["synthetic_far_separates"] == 1.0
