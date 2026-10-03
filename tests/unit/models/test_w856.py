from quant_fund.models.halton_seq import (
    bench_halton_seq,
)
from quant_fund.models.latin_hypercube import (
    bench_latin_hypercube,
)
from quant_fund.models.monte_carlo_quad import (
    bench_monte_carlo_quad,
)
from quant_fund.models.quasi_mc import (
    bench_quasi_mc,
)
from quant_fund.models.sobol_seq import (
    bench_sobol_seq,
)
from quant_fund.models.stratified_mc import (
    bench_stratified_mc,
)


def test_monte_carlo_quad():
    assert bench_monte_carlo_quad()["synthetic_monte_carlo_quad"] == 1.0


def test_quasi_mc():
    assert bench_quasi_mc()["synthetic_quasi_mc"] == 1.0


def test_halton_seq():
    assert bench_halton_seq()["synthetic_halton_seq"] == 1.0


def test_sobol_seq():
    assert bench_sobol_seq()["synthetic_sobol_seq"] == 1.0


def test_latin_hypercube():
    assert bench_latin_hypercube()["synthetic_latin_hypercube"] == 1.0


def test_stratified_mc():
    assert bench_stratified_mc()["synthetic_stratified_mc"] == 1.0
