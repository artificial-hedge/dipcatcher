from quant_fund.models.deformation_functor import (
    bench_deformation_functor,
)
from quant_fund.models.maurer_cartan import bench_maurer_cartan
from quant_fund.models.obstruction_theory import (
    bench_obstruction_theory,
)
from quant_fund.models.schlessinger import bench_schlessinger
from quant_fund.models.tangent_space_def import (
    bench_tangent_space_def,
)
from quant_fund.models.versal_deformation import (
    bench_versal_deformation,
)


def test_deformation_functor():
    assert bench_deformation_functor()["synthetic_deformation_functor"] == 1.0


def test_schlessinger():
    assert bench_schlessinger()["synthetic_schlessinger"] == 1.0


def test_tangent_space_def():
    assert bench_tangent_space_def()["synthetic_tangent_space_def"] == 1.0


def test_obstruction_theory():
    assert bench_obstruction_theory()["synthetic_obstruction_theory"] == 1.0


def test_versal_deformation():
    assert bench_versal_deformation()["synthetic_versal_deformation"] == 1.0


def test_maurer_cartan():
    assert bench_maurer_cartan()["synthetic_maurer_cartan"] == 1.0
