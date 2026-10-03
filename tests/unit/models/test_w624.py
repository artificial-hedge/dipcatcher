from quant_fund.models.adic_formal import bench_adic_formal
from quant_fund.models.algebraization import (
    bench_algebraization,
)
from quant_fund.models.formal_completion import (
    bench_formal_completion,
)
from quant_fund.models.formal_neighborhood import (
    bench_formal_neighborhood,
)
from quant_fund.models.groth_existence import (
    bench_groth_existence,
)
from quant_fund.models.raynaud_formal import (
    bench_raynaud_formal,
)


def test_raynaud_formal():
    assert bench_raynaud_formal()["synthetic_raynaud_formal"] == 1.0


def test_formal_completion():
    assert bench_formal_completion()["synthetic_formal_completion"] == 1.0


def test_adic_formal():
    assert bench_adic_formal()["synthetic_adic_formal"] == 1.0


def test_formal_neighborhood():
    assert bench_formal_neighborhood()["synthetic_formal_neighborhood"] == 1.0


def test_groth_existence():
    assert bench_groth_existence()["synthetic_groth_existence"] == 1.0


def test_algebraization():
    assert bench_algebraization()["synthetic_algebraization"] == 1.0
