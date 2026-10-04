from quant_fund.models.motivic_cartier import (
    bench_motivic_cartier,
)
from quant_fund.models.motivic_frobenius import (
    bench_motivic_frobenius,
)
from quant_fund.models.motivic_hodge import (
    bench_motivic_hodge,
)
from quant_fund.models.motivic_lax import bench_motivic_lax
from quant_fund.models.motivic_span import bench_motivic_span
from quant_fund.models.motivic_street import (
    bench_motivic_street,
)


def test_motivic_frobenius():
    assert bench_motivic_frobenius()["synthetic_motivic_frobenius"] == 1.0


def test_motivic_cartier():
    assert bench_motivic_cartier()["synthetic_motivic_cartier"] == 1.0


def test_motivic_hodge():
    assert bench_motivic_hodge()["synthetic_motivic_hodge"] == 1.0


def test_motivic_span():
    assert bench_motivic_span()["synthetic_motivic_span"] == 1.0


def test_motivic_lax():
    assert bench_motivic_lax()["synthetic_motivic_lax"] == 1.0


def test_motivic_street():
    assert bench_motivic_street()["synthetic_motivic_street"] == 1.0
