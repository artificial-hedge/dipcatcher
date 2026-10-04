"""Wave-986 Calderon-Zygmund canon tests."""

from __future__ import annotations

from quant_fund.models.ap_weight import bench_ap_weight
from quant_fund.models.calderon_zygmund import bench_calderon_zygmund
from quant_fund.models.cotlar_ineq import bench_cotlar_ineq
from quant_fund.models.cz_decomp import bench_cz_decomp
from quant_fund.models.good_lambda import bench_good_lambda
from quant_fund.models.reverse_holder import bench_reverse_holder


def test_calderon_zygmund():
    assert bench_calderon_zygmund()["synthetic_calderon_zygmund"] == 1.0


def test_cz_decomp():
    assert bench_cz_decomp()["synthetic_cz_decomp"] == 1.0


def test_cotlar_ineq():
    assert bench_cotlar_ineq()["synthetic_cotlar_ineq"] == 1.0


def test_good_lambda():
    assert bench_good_lambda()["synthetic_good_lambda"] == 1.0


def test_ap_weight():
    assert bench_ap_weight()["synthetic_ap_weight"] == 1.0


def test_reverse_holder():
    assert bench_reverse_holder()["synthetic_reverse_holder"] == 1.0
