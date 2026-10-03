from quant_fund.models.bicgstab2 import (
    bench_bicgstab2,
)
from quant_fund.models.block_cg import (
    bench_block_cg,
)
from quant_fund.models.cgs_solver import (
    bench_cgs_solver,
)
from quant_fund.models.minres_solver import (
    bench_minres_solver,
)
from quant_fund.models.qmr_solver import (
    bench_qmr_solver,
)
from quant_fund.models.tfqmr import (
    bench_tfqmr,
)


def test_minres_solver():
    assert bench_minres_solver()["synthetic_minres_solver"] == 1.0


def test_cgs_solver():
    assert bench_cgs_solver()["synthetic_cgs_solver"] == 1.0


def test_tfqmr():
    assert bench_tfqmr()["synthetic_tfqmr"] == 1.0


def test_qmr_solver():
    assert bench_qmr_solver()["synthetic_qmr_solver"] == 1.0


def test_bicgstab2():
    assert bench_bicgstab2()["synthetic_bicgstab2"] == 1.0


def test_block_cg():
    assert bench_block_cg()["synthetic_block_cg"] == 1.0
