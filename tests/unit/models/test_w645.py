from quant_fund.models.balmer_k import bench_balmer_k
from quant_fund.models.hermitian_k3 import (
    bench_hermitian_k3,
)
from quant_fund.models.schlichting_k import (
    bench_schlichting_k,
)
from quant_fund.models.thomason_les import (
    bench_thomason_les,
)
from quant_fund.models.vishik_k import bench_vishik_k
from quant_fund.models.witt_k import bench_witt_k


def test_witt_k():
    assert bench_witt_k()["synthetic_witt_k"] == 1.0


def test_schlichting_k():
    assert bench_schlichting_k()["synthetic_schlichting_k"] == 1.0


def test_balmer_k():
    assert bench_balmer_k()["synthetic_balmer_k"] == 1.0


def test_hermitian_k3():
    assert bench_hermitian_k3()["synthetic_hermitian_k3"] == 1.0


def test_thomason_les():
    assert bench_thomason_les()["synthetic_thomason_les"] == 1.0


def test_vishik_k():
    assert bench_vishik_k()["synthetic_vishik_k"] == 1.0
