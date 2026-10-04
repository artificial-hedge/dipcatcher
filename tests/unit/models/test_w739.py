from quant_fund.models.bernardi_bijection import (
    bench_bernardi_bijection,
)
from quant_fund.models.bonzom_combe import bench_bonzom_combe
from quant_fund.models.bouttier_guiter import bench_bouttier_guiter
from quant_fund.models.caraceni_curien import bench_caraceni_curien
from quant_fund.models.mullin_bijection import (
    bench_mullin_bijection,
)
from quant_fund.models.schaeffer_bijection import (
    bench_schaeffer_bijection,
)


def test_caraceni_curien():
    assert bench_caraceni_curien()["synthetic_caraceni_curien"] == 1.0


def test_bonzom_combe():
    assert bench_bonzom_combe()["synthetic_bonzom_combe"] == 1.0


def test_mullin_bijection():
    assert bench_mullin_bijection()["synthetic_mullin_bijection"] == 1.0


def test_bernardi_bijection():
    assert bench_bernardi_bijection()["synthetic_bernardi_bijection"] == 1.0


def test_schaeffer_bijection():
    assert bench_schaeffer_bijection()["synthetic_schaeffer_bijection"] == 1.0


def test_bouttier_guiter():
    assert bench_bouttier_guiter()["synthetic_bouttier_guiter"] == 1.0
