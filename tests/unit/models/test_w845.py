from quant_fund.models.abel_transform import (
    bench_abel_transform,
)
from quant_fund.models.hankel_transform import (
    bench_hankel_transform,
)
from quant_fund.models.hilbert_transform import (
    bench_hilbert_transform,
)
from quant_fund.models.laplace_transform import (
    bench_laplace_transform,
)
from quant_fund.models.mellin_transform import (
    bench_mellin_transform,
)
from quant_fund.models.z_transform import (
    bench_z_transform,
)


def test_laplace_transform():
    assert bench_laplace_transform()["synthetic_laplace_transform"] == 1.0


def test_mellin_transform():
    assert bench_mellin_transform()["synthetic_mellin_transform"] == 1.0


def test_hankel_transform():
    assert bench_hankel_transform()["synthetic_hankel_transform"] == 1.0


def test_z_transform():
    assert bench_z_transform()["synthetic_z_transform"] == 1.0


def test_hilbert_transform():
    assert bench_hilbert_transform()["synthetic_hilbert_transform"] == 1.0


def test_abel_transform():
    assert bench_abel_transform()["synthetic_abel_transform"] == 1.0
