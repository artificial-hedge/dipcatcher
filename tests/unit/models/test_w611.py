from quant_fund.models.mate_dual import bench_mate_dual
from quant_fund.models.modification import bench_modification
from quant_fund.models.pasting_diag import bench_pasting_diag
from quant_fund.models.pseudo_naturality import (
    bench_pseudo_naturality,
)
from quant_fund.models.two_adjoint import bench_two_adjoint
from quant_fund.models.whisker_comp import bench_whisker_comp


def test_pasting_diag():
    assert bench_pasting_diag()["synthetic_pasting_diag"] == 1.0


def test_mate_dual():
    assert bench_mate_dual()["synthetic_mate_dual"] == 1.0


def test_whisker_comp():
    assert bench_whisker_comp()["synthetic_whisker_comp"] == 1.0


def test_pseudo_naturality():
    assert bench_pseudo_naturality()["synthetic_pseudo_naturality"] == 1.0


def test_two_adjoint():
    assert bench_two_adjoint()["synthetic_two_adjoint"] == 1.0


def test_modification():
    assert bench_modification()["synthetic_modification"] == 1.0
