from quant_fund.models.chung_series import (
    bench_chung_series,
)
from quant_fund.models.ito_nisio import (
    bench_ito_nisio,
)
from quant_fund.models.kolmogorov_two import (
    bench_kolmogorov_two,
)
from quant_fund.models.ortega_series import (
    bench_ortega_series,
)
from quant_fund.models.salem_zygmund import (
    bench_salem_zygmund,
)
from quant_fund.models.three_series import (
    bench_three_series,
)


def test_three_series():
    assert bench_three_series()["synthetic_three_series"] == 1.0


def test_kolmogorov_two():
    assert bench_kolmogorov_two()["synthetic_kolmogorov_two"] == 1.0


def test_ito_nisio():
    assert bench_ito_nisio()["synthetic_ito_nisio"] == 1.0


def test_chung_series():
    assert bench_chung_series()["synthetic_chung_series"] == 1.0


def test_ortega_series():
    assert bench_ortega_series()["synthetic_ortega_series"] == 1.0


def test_salem_zygmund():
    assert bench_salem_zygmund()["synthetic_salem_zygmund"] == 1.0
