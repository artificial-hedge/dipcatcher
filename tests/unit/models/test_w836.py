from quant_fund.models.buffon_needle import (
    bench_buffon_needle,
)
from quant_fund.models.crofton_formula import (
    bench_crofton_formula,
)
from quant_fund.models.hadwiger_chars import (
    bench_hadwiger_chars,
)
from quant_fund.models.kinematic_measure import (
    bench_kinematic_measure,
)
from quant_fund.models.kubota_mean_width import (
    bench_kubota_mean_width,
)
from quant_fund.models.santalo_measure import (
    bench_santalo_measure,
)


def test_crofton_formula():
    assert bench_crofton_formula()["synthetic_crofton_formula"] == 1.0


def test_kinematic_measure():
    assert bench_kinematic_measure()["synthetic_kinematic_measure"] == 1.0


def test_buffon_needle():
    assert bench_buffon_needle()["synthetic_buffon_needle"] == 1.0


def test_santalo_measure():
    assert bench_santalo_measure()["synthetic_santalo_measure"] == 1.0


def test_kubota_mean_width():
    assert bench_kubota_mean_width()["synthetic_kubota_mean_width"] == 1.0


def test_hadwiger_chars():
    assert bench_hadwiger_chars()["synthetic_hadwiger_chars"] == 1.0
