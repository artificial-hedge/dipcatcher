from quant_fund.models.eight_geometries import bench_eight_geometries
from quant_fund.models.haken_mfd import bench_haken_mfd
from quant_fund.models.jsj_decomp import bench_jsj_decomp
from quant_fund.models.ricci_flow import bench_ricci_flow
from quant_fund.models.seifert_fibered import bench_seifert_fibered
from quant_fund.models.thurston_geometrization import (
    bench_thurston_geometrization,
)


def test_thurston_geometrization():
    out = bench_thurston_geometrization()
    assert out["synthetic_thurston_geometrization"] == 1.0


def test_eight_geometries():
    assert bench_eight_geometries()["synthetic_eight_geometries"] == 1.0


def test_seifert_fibered():
    assert bench_seifert_fibered()["synthetic_seifert_fibered"] == 1.0


def test_haken_mfd():
    assert bench_haken_mfd()["synthetic_haken_mfd"] == 1.0


def test_jsj_decomp():
    assert bench_jsj_decomp()["synthetic_jsj_decomp"] == 1.0


def test_ricci_flow():
    assert bench_ricci_flow()["synthetic_ricci_flow"] == 1.0
