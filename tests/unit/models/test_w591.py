from quant_fund.models.friedlander_voev import (
    bench_friedlander_voev,
)
from quant_fund.models.motivic_descent import (
    bench_motivic_descent,
)
from quant_fund.models.motivic_eilenberg import (
    bench_motivic_eilenberg,
)
from quant_fund.models.motivic_invert import (
    bench_motivic_invert,
)
from quant_fund.models.motivic_purity import (
    bench_motivic_purity,
)
from quant_fund.models.motivic_zeta import bench_motivic_zeta


def test_friedlander_voev():
    assert bench_friedlander_voev()["synthetic_friedlander_voev"] == 1.0


def test_motivic_eilenberg():
    assert bench_motivic_eilenberg()["synthetic_motivic_eilenberg"] == 1.0


def test_motivic_zeta():
    assert bench_motivic_zeta()["synthetic_motivic_zeta"] == 1.0


def test_motivic_purity():
    assert bench_motivic_purity()["synthetic_motivic_purity"] == 1.0


def test_motivic_descent():
    assert bench_motivic_descent()["synthetic_motivic_descent"] == 1.0


def test_motivic_invert():
    assert bench_motivic_invert()["synthetic_motivic_invert"] == 1.0
