from quant_fund.models.breuil_mod import bench_breuil_mod
from quant_fund.models.etale_phi import bench_etale_phi
from quant_fund.models.finite_height import (
    bench_finite_height,
)
from quant_fund.models.galois_lattice import (
    bench_galois_lattice,
)
from quant_fund.models.kisin_mod import bench_kisin_mod
from quant_fund.models.padic_hodge import bench_padic_hodge


def test_breuil_mod():
    assert bench_breuil_mod()["synthetic_breuil_mod"] == 1.0


def test_kisin_mod():
    assert bench_kisin_mod()["synthetic_kisin_mod"] == 1.0


def test_galois_lattice():
    assert bench_galois_lattice()["synthetic_galois_lattice"] == 1.0


def test_padic_hodge():
    assert bench_padic_hodge()["synthetic_padic_hodge"] == 1.0


def test_finite_height():
    assert bench_finite_height()["synthetic_finite_height"] == 1.0


def test_etale_phi():
    assert bench_etale_phi()["synthetic_etale_phi"] == 1.0
