from quant_fund.models.dof_management import (
    bench_dof_management,
)
from quant_fund.models.edge_elements import (
    bench_edge_elements,
)
from quant_fund.models.fem_assembly import (
    bench_fem_assembly,
)
from quant_fund.models.isoparametric_map import (
    bench_isoparametric_map,
)
from quant_fund.models.quadrature_rules import (
    bench_quadrature_rules,
)
from quant_fund.models.triangular_basis import (
    bench_triangular_basis,
)


def test_fem_assembly():
    assert bench_fem_assembly()["synthetic_fem_assembly"] == 1.0


def test_isoparametric_map():
    assert bench_isoparametric_map()["synthetic_isoparametric_map"] == 1.0


def test_quadrature_rules():
    assert bench_quadrature_rules()["synthetic_quadrature_rules"] == 1.0


def test_triangular_basis():
    assert bench_triangular_basis()["synthetic_triangular_basis"] == 1.0


def test_edge_elements():
    assert bench_edge_elements()["synthetic_edge_elements"] == 1.0


def test_dof_management():
    assert bench_dof_management()["synthetic_dof_management"] == 1.0
