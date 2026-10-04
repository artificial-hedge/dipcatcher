from quant_fund.models.contact_geom import bench_contact_geom
from quant_fund.models.gromov_nonsq import bench_gromov_nonsq
from quant_fund.models.hamiltonian_flow import bench_hamiltonian_flow
from quant_fund.models.lagrangian_mfd import bench_lagrangian_mfd
from quant_fund.models.poisson_bracket import bench_poisson_bracket
from quant_fund.models.symplectic_form import bench_symplectic_form


def test_symplectic_form():
    assert bench_symplectic_form()["synthetic_symplectic_form"] == 1.0


def test_lagrangian_mfd():
    assert bench_lagrangian_mfd()["synthetic_lagrangian_mfd"] == 1.0


def test_hamiltonian_flow():
    assert bench_hamiltonian_flow()["synthetic_hamiltonian_flow"] == 1.0


def test_poisson_bracket():
    assert bench_poisson_bracket()["synthetic_poisson_bracket"] == 1.0


def test_contact_geom():
    assert bench_contact_geom()["synthetic_contact_geom"] == 1.0


def test_gromov_nonsq():
    assert bench_gromov_nonsq()["synthetic_gromov_nonsq"] == 1.0
