from quant_fund.models.boolean_model import (
    bench_boolean_model,
)
from quant_fund.models.germ_grain import (
    bench_germ_grain,
)
from quant_fund.models.intrinsic_volumes import (
    bench_intrinsic_volumes,
)
from quant_fund.models.miles_matheron import (
    bench_miles_matheron,
)
from quant_fund.models.poisson_voronoi import (
    bench_poisson_voronoi,
)
from quant_fund.models.steiner_formula import (
    bench_steiner_formula,
)


def test_poisson_voronoi():
    assert bench_poisson_voronoi()["synthetic_poisson_voronoi"] == 1.0


def test_boolean_model():
    assert bench_boolean_model()["synthetic_boolean_model"] == 1.0


def test_germ_grain():
    assert bench_germ_grain()["synthetic_germ_grain"] == 1.0


def test_steiner_formula():
    assert bench_steiner_formula()["synthetic_steiner_formula"] == 1.0


def test_miles_matheron():
    assert bench_miles_matheron()["synthetic_miles_matheron"] == 1.0


def test_intrinsic_volumes():
    assert bench_intrinsic_volumes()["synthetic_intrinsic_volumes"] == 1.0
