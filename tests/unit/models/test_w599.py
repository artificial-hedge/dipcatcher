from quant_fund.models.algebra_cat import bench_algebra_cat
from quant_fund.models.codensity_monad import (
    bench_codensity_monad,
)
from quant_fund.models.distributive_law import (
    bench_distributive_law,
)
from quant_fund.models.klesli_cat import bench_klesli_cat
from quant_fund.models.monad_theorem import (
    bench_monad_theorem,
)
from quant_fund.models.monadicity import bench_monadicity


def test_monad_theorem():
    assert bench_monad_theorem()["synthetic_monad_theorem"] == 1.0


def test_klesli_cat():
    assert bench_klesli_cat()["synthetic_klesli_cat"] == 1.0


def test_codensity_monad():
    assert bench_codensity_monad()["synthetic_codensity_monad"] == 1.0


def test_monadicity():
    assert bench_monadicity()["synthetic_monadicity"] == 1.0


def test_distributive_law():
    assert bench_distributive_law()["synthetic_distributive_law"] == 1.0


def test_algebra_cat():
    assert bench_algebra_cat()["synthetic_algebra_cat"] == 1.0
