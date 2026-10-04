from quant_fund.models.form_analysis import (
    bench_form_analysis,
)
from quant_fund.models.greedy_marking import (
    bench_greedy_marking,
)
from quant_fund.models.hp_adaptive import (
    bench_hp_adaptive,
)
from quant_fund.models.residual_marking import (
    bench_residual_marking,
)
from quant_fund.models.space_time_adapt import (
    bench_space_time_adapt,
)
from quant_fund.models.wavelet_adapt import (
    bench_wavelet_adapt,
)


def test_space_time_adapt():
    assert bench_space_time_adapt()["synthetic_space_time_adapt"] == 1.0


def test_greedy_marking():
    assert bench_greedy_marking()["synthetic_greedy_marking"] == 1.0


def test_form_analysis():
    assert bench_form_analysis()["synthetic_form_analysis"] == 1.0


def test_hp_adaptive():
    assert bench_hp_adaptive()["synthetic_hp_adaptive"] == 1.0


def test_wavelet_adapt():
    assert bench_wavelet_adapt()["synthetic_wavelet_adapt"] == 1.0


def test_residual_marking():
    assert bench_residual_marking()["synthetic_residual_marking"] == 1.0
