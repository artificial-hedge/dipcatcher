from quant_fund.models.contact_homology3 import (
    bench_contact_homology3,
)
from quant_fund.models.eliashberg_givental import (
    bench_eliashberg_givental,
)
from quant_fund.models.floer_homol import bench_floer_homol
from quant_fund.models.reeb_orbit import bench_reeb_orbit
from quant_fund.models.sft_algebra import bench_sft_algebra
from quant_fund.models.symplectic_field import (
    bench_symplectic_field,
)


def test_symplectic_field():
    assert bench_symplectic_field()["synthetic_symplectic_field"] == 1.0


def test_contact_homology3():
    assert bench_contact_homology3()["synthetic_contact_homology3"] == 1.0


def test_floer_homol():
    assert bench_floer_homol()["synthetic_floer_homol"] == 1.0


def test_reeb_orbit():
    assert bench_reeb_orbit()["synthetic_reeb_orbit"] == 1.0


def test_sft_algebra():
    assert bench_sft_algebra()["synthetic_sft_algebra"] == 1.0


def test_eliashberg_givental():
    assert bench_eliashberg_givental()["synthetic_eliashberg_givental"] == 1.0
