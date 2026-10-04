from quant_fund.models.dag_representation import bench_dag_representation
from quant_fund.models.derived_deformation import bench_derived_deformation
from quant_fund.models.derived_moduli import bench_derived_moduli
from quant_fund.models.formal_deformation import bench_formal_deformation
from quant_fund.models.obstruction_2 import bench_obstruction_2
from quant_fund.models.tangent_coh import bench_tangent_coh


def test_derived_deformation():
    assert bench_derived_deformation()["synthetic_derived_deformation"] == 1.0


def test_formal_deformation():
    assert bench_formal_deformation()["synthetic_formal_deformation"] == 1.0


def test_dag_representation():
    assert bench_dag_representation()["synthetic_dag_representation"] == 1.0


def test_derived_moduli():
    assert bench_derived_moduli()["synthetic_derived_moduli"] == 1.0


def test_tangent_coh():
    assert bench_tangent_coh()["synthetic_tangent_coh"] == 1.0


def test_obstruction_2():
    assert bench_obstruction_2()["synthetic_obstruction_2"] == 1.0
