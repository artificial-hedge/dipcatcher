from quant_fund.models.dualizing_cmplx import (
    bench_dualizing_cmplx,
)
from quant_fund.models.dualizing_sheaf import (
    bench_dualizing_sheaf,
)
from quant_fund.models.groth_duality import (
    bench_groth_duality,
)
from quant_fund.models.relative_duality import (
    bench_relative_duality,
)
from quant_fund.models.residue_thm import bench_residue_thm
from quant_fund.models.verdier_duality import (
    bench_verdier_duality,
)


def test_groth_duality():
    assert bench_groth_duality()["synthetic_groth_duality"] == 1.0


def test_dualizing_cmplx():
    assert bench_dualizing_cmplx()["synthetic_dualizing_cmplx"] == 1.0


def test_residue_thm():
    assert bench_residue_thm()["synthetic_residue_thm"] == 1.0


def test_verdier_duality():
    assert bench_verdier_duality()["synthetic_verdier_duality"] == 1.0


def test_dualizing_sheaf():
    assert bench_dualizing_sheaf()["synthetic_dualizing_sheaf"] == 1.0


def test_relative_duality():
    assert bench_relative_duality()["synthetic_relative_duality"] == 1.0
