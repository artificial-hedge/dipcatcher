"""Wave-197 inventory canon tests."""

from __future__ import annotations

from quant_fund.models.base_stock import bench_base_stock
from quant_fund.models.clark_scarf import bench_clark_scarf
from quant_fund.models.eoq_model import bench_eoq_model
from quant_fund.models.newsvendor import bench_newsvendor
from quant_fund.models.ss_policy import bench_ss_policy
from quant_fund.models.wagner_whitin import bench_wagner_whitin


class TestEOQ:
    def test_bench(self) -> None:
        out = bench_eoq_model()
        assert out["synthetic_eoq_gain"] > 0.0
        assert out["synthetic_eoq_fill"] > 0.9


class TestNewsvendor:
    def test_bench(self) -> None:
        out = bench_newsvendor()
        assert abs(out["synthetic_nv_qstar"] - out["synthetic_nv_q_emp"]) <= 1.0
        assert out["synthetic_nv_gain"] > 0.0


class TestSS:
    def test_bench(self) -> None:
        out = bench_ss_policy()
        assert out["synthetic_ss_gain"] >= 0.0


class TestWW:
    def test_bench(self) -> None:
        out = bench_wagner_whitin()
        assert out["synthetic_ww_cost"] <= out["synthetic_lfl_cost"]


class TestBaseStock:
    def test_bench(self) -> None:
        out = bench_base_stock()
        assert out["synthetic_bs_fill"] > 0.95


class TestClarkScarf:
    def test_bench(self) -> None:
        out = bench_clark_scarf()
        assert out["synthetic_cs_gain"] > 0.0
        assert out["synthetic_cs_fill"] > out["synthetic_myopic_fill"]
