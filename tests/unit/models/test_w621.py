from quant_fund.models.motivic_borel import bench_motivic_borel
from quant_fund.models.motivic_chow import bench_motivic_chow
from quant_fund.models.motivic_class import bench_motivic_class
from quant_fund.models.motivic_height import (
    bench_motivic_height,
)
from quant_fund.models.motivic_homology import (
    bench_motivic_homology,
)
from quant_fund.models.motivic_k import bench_motivic_k


def test_motivic_k():
    assert bench_motivic_k()["synthetic_motivic_k"] == 1.0


def test_motivic_borel():
    assert bench_motivic_borel()["synthetic_motivic_borel"] == 1.0


def test_motivic_height():
    assert bench_motivic_height()["synthetic_motivic_height"] == 1.0


def test_motivic_chow():
    assert bench_motivic_chow()["synthetic_motivic_chow"] == 1.0


def test_motivic_homology():
    assert bench_motivic_homology()["synthetic_motivic_homology"] == 1.0


def test_motivic_class():
    assert bench_motivic_class()["synthetic_motivic_class"] == 1.0
