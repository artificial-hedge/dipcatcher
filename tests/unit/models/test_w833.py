from quant_fund.models.brownian_approx import (
    bench_brownian_approx,
)
from quant_fund.models.donsker_invariance import (
    bench_donsker_invariance,
)
from quant_fund.models.fclt_invariance import (
    bench_fclt_invariance,
)
from quant_fund.models.martingale_fclt import (
    bench_martingale_fclt,
)
from quant_fund.models.stable_limit import (
    bench_stable_limit,
)
from quant_fund.models.strassen_flln import (
    bench_strassen_flln,
)


def test_fclt_invariance():
    assert bench_fclt_invariance()["synthetic_fclt_invariance"] == 1.0


def test_donsker_invariance():
    assert bench_donsker_invariance()["synthetic_donsker_invariance"] == 1.0


def test_martingale_fclt():
    assert bench_martingale_fclt()["synthetic_martingale_fclt"] == 1.0


def test_stable_limit():
    assert bench_stable_limit()["synthetic_stable_limit"] == 1.0


def test_brownian_approx():
    assert bench_brownian_approx()["synthetic_brownian_approx"] == 1.0


def test_strassen_flln():
    assert bench_strassen_flln()["synthetic_strassen_flln"] == 1.0
