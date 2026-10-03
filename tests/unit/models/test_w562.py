from quant_fund.models.ekeland_hofer import bench_ekeland_hofer
from quant_fund.models.gromov_width import bench_gromov_width
from quant_fund.models.hofer_metric import bench_hofer_metric
from quant_fund.models.mcduff_polterovich import (
    bench_mcduff_polterovich,
)
from quant_fund.models.symplectic_capacity import (
    bench_symplectic_capacity,
)
from quant_fund.models.symplectic_packing import (
    bench_symplectic_packing,
)


def test_gromov_width():
    assert bench_gromov_width()["synthetic_gromov_width"] == 1.0


def test_hofer_metric():
    assert bench_hofer_metric()["synthetic_hofer_metric"] == 1.0


def test_symplectic_capacity():
    assert bench_symplectic_capacity()["synthetic_symplectic_capacity"] == 1.0


def test_symplectic_packing():
    assert bench_symplectic_packing()["synthetic_symplectic_packing"] == 1.0


def test_mcduff_polterovich():
    assert bench_mcduff_polterovich()["synthetic_mcduff_polterovich"] == 1.0


def test_ekeland_hofer():
    assert bench_ekeland_hofer()["synthetic_ekeland_hofer"] == 1.0
