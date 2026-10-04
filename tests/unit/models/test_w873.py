from quant_fund.models.asm_precond import (
    bench_asm_precond,
)
from quant_fund.models.baldding_dd import (
    bench_baldding_dd,
)
from quant_fund.models.dd_partition import (
    bench_dd_partition,
)
from quant_fund.models.feti_dp import (
    bench_feti_dp,
)
from quant_fund.models.neumann_dd import (
    bench_neumann_dd,
)
from quant_fund.models.subspace_dd import (
    bench_subspace_dd,
)


def test_dd_partition():
    assert bench_dd_partition()["synthetic_dd_partition"] == 1.0


def test_baldding_dd():
    assert bench_baldding_dd()["synthetic_baldding_dd"] == 1.0


def test_neumann_dd():
    assert bench_neumann_dd()["synthetic_neumann_dd"] == 1.0


def test_feti_dp():
    assert bench_feti_dp()["synthetic_feti_dp"] == 1.0


def test_subspace_dd():
    assert bench_subspace_dd()["synthetic_subspace_dd"] == 1.0


def test_asm_precond():
    assert bench_asm_precond()["synthetic_asm_precond"] == 1.0
