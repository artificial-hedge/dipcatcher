from quant_fund.models.ding_dupias import bench_ding_dupias
from quant_fund.models.gaines_sle import bench_gaines_sle
from quant_fund.models.gwynne_miller import bench_gwynne_miller
from quant_fund.models.miller_wu import bench_miller_wu
from quant_fund.models.rhoade_vargas import bench_rhoade_vargas
from quant_fund.models.sheffield_quantum import (
    bench_sheffield_quantum,
)


def test_sheffield_quantum():
    assert bench_sheffield_quantum()["synthetic_sheffield_quantum"] == 1.0


def test_gaines_sle():
    assert bench_gaines_sle()["synthetic_gaines_sle"] == 1.0


def test_miller_wu():
    assert bench_miller_wu()["synthetic_miller_wu"] == 1.0


def test_rhoade_vargas():
    assert bench_rhoade_vargas()["synthetic_rhoade_vargas"] == 1.0


def test_ding_dupias():
    assert bench_ding_dupias()["synthetic_ding_dupias"] == 1.0


def test_gwynne_miller():
    assert bench_gwynne_miller()["synthetic_gwynne_miller"] == 1.0
