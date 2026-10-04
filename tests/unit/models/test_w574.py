from quant_fund.models.exotic_sphere import bench_exotic_sphere
from quant_fund.models.immersion_thm import bench_immersion_thm
from quant_fund.models.kervaire_milnor import (
    bench_kervaire_milnor,
)
from quant_fund.models.smale_hcob import bench_smale_hcob
from quant_fund.models.surgery_theory import bench_surgery_theory
from quant_fund.models.whitney_trick import bench_whitney_trick


def test_exotic_sphere():
    assert bench_exotic_sphere()["synthetic_exotic_sphere"] == 1.0


def test_kervaire_milnor():
    assert bench_kervaire_milnor()["synthetic_kervaire_milnor"] == 1.0


def test_surgery_theory():
    assert bench_surgery_theory()["synthetic_surgery_theory"] == 1.0


def test_smale_hcob():
    assert bench_smale_hcob()["synthetic_smale_hcob"] == 1.0


def test_whitney_trick():
    assert bench_whitney_trick()["synthetic_whitney_trick"] == 1.0


def test_immersion_thm():
    assert bench_immersion_thm()["synthetic_immersion_thm"] == 1.0
