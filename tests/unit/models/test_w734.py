from quant_fund.models.garmadon_sle import bench_garmadon_sle
from quant_fund.models.lawler_werner import bench_lawler_werner
from quant_fund.models.miller_sheffield import (
    bench_miller_sheffield,
)
from quant_fund.models.osgood_schramm import (
    bench_osgood_schramm,
)
from quant_fund.models.smirnov_parafermion import (
    bench_smirnov_parafermion,
)
from quant_fund.models.werner_wilson import bench_werner_wilson


def test_osgood_schramm():
    assert bench_osgood_schramm()["synthetic_osgood_schramm"] == 1.0


def test_lawler_werner():
    assert bench_lawler_werner()["synthetic_lawler_werner"] == 1.0


def test_werner_wilson():
    assert bench_werner_wilson()["synthetic_werner_wilson"] == 1.0


def test_smirnov_parafermion():
    assert bench_smirnov_parafermion()["synthetic_smirnov_parafermion"] == 1.0


def test_garmadon_sle():
    assert bench_garmadon_sle()["synthetic_garmadon_sle"] == 1.0


def test_miller_sheffield():
    assert bench_miller_sheffield()["synthetic_miller_sheffield"] == 1.0
