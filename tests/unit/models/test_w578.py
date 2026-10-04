from quant_fund.models.bogomolov_conj import bench_bogomolov_conj
from quant_fund.models.canonical_height import (
    bench_canonical_height,
)
from quant_fund.models.equidistribution_thm import (
    bench_equidistribution_thm,
)
from quant_fund.models.global_height import bench_global_height
from quant_fund.models.nevanlinna_th import bench_nevanlinna_th
from quant_fund.models.vojta_conj import bench_vojta_conj


def test_global_height():
    assert bench_global_height()["synthetic_global_height"] == 1.0


def test_bogomolov_conj():
    assert bench_bogomolov_conj()["synthetic_bogomolov_conj"] == 1.0


def test_equidistribution_thm():
    assert bench_equidistribution_thm()["synthetic_equidistribution_thm"] == 1.0


def test_canonical_height():
    assert bench_canonical_height()["synthetic_canonical_height"] == 1.0


def test_nevanlinna_th():
    assert bench_nevanlinna_th()["synthetic_nevanlinna_th"] == 1.0


def test_vojta_conj():
    assert bench_vojta_conj()["synthetic_vojta_conj"] == 1.0
