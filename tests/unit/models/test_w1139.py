"""Wave-1139 medicine-6 canon tests."""

from __future__ import annotations

from quant_fund.models.dentistry_2 import bench_dentistry_2
from quant_fund.models.dietetics import bench_dietetics
from quant_fund.models.occupational_therapy import bench_occupational_therapy
from quant_fund.models.optometry import bench_optometry
from quant_fund.models.physiotherapy import bench_physiotherapy
from quant_fund.models.podiatry import bench_podiatry


def test_optometry():
    assert bench_optometry()["synthetic_optometry"] == 1.0


def test_dentistry_2():
    assert bench_dentistry_2()["synthetic_dentistry_2"] == 1.0


def test_podiatry():
    assert bench_podiatry()["synthetic_podiatry"] == 1.0


def test_dietetics():
    assert bench_dietetics()["synthetic_dietetics"] == 1.0


def test_physiotherapy():
    assert bench_physiotherapy()["synthetic_physiotherapy"] == 1.0


def test_occupational_therapy():
    assert bench_occupational_therapy()["synthetic_occupational_therapy"] == 1.0
