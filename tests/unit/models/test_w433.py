from quant_fund.models.adic_space import bench_adic_space
from quant_fund.models.berkovich_space import (
    bench_berkovich_space,
)
from quant_fund.models.diamond_toy import bench_diamond_toy
from quant_fund.models.etale_ph2 import bench_etale_ph2
from quant_fund.models.perfectoid_space import (
    bench_perfectoid_space,
)
from quant_fund.models.rigid_analytic import bench_rigid_analytic


def test_rigid_analytic():
    assert bench_rigid_analytic()["synthetic_rigid_analytic"] == 1.0


def test_berkovich_space():
    assert bench_berkovich_space()["synthetic_berkovich_space"] == 1.0


def test_perfectoid_space():
    assert bench_perfectoid_space()["synthetic_perfectoid_space"] == 1.0


def test_adic_space():
    assert bench_adic_space()["synthetic_adic_space"] == 1.0


def test_etale_ph2():
    assert bench_etale_ph2()["synthetic_etale_ph2"] == 1.0


def test_diamond_toy():
    assert bench_diamond_toy()["synthetic_diamond_toy"] == 1.0
