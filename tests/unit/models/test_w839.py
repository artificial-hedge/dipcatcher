from quant_fund.models.bernstein_poly import (
    bench_bernstein_poly,
)
from quant_fund.models.chebyshev_alternation import (
    bench_chebyshev_alternation,
)
from quant_fund.models.fourier_decay import (
    bench_fourier_decay,
)
from quant_fund.models.jackson_direct import (
    bench_jackson_direct,
)
from quant_fund.models.kolmogorov_nwidth import (
    bench_kolmogorov_nwidth,
)
from quant_fund.models.markov_brothers import (
    bench_markov_brothers,
)


def test_jackson_direct():
    assert bench_jackson_direct()["synthetic_jackson_direct"] == 1.0


def test_chebyshev_alternation():
    assert bench_chebyshev_alternation()["synthetic_chebyshev_alternation"] == 1.0


def test_kolmogorov_nwidth():
    assert bench_kolmogorov_nwidth()["synthetic_kolmogorov_nwidth"] == 1.0


def test_bernstein_poly():
    assert bench_bernstein_poly()["synthetic_bernstein_poly"] == 1.0


def test_markov_brothers():
    assert bench_markov_brothers()["synthetic_markov_brothers"] == 1.0


def test_fourier_decay():
    assert bench_fourier_decay()["synthetic_fourier_decay"] == 1.0
