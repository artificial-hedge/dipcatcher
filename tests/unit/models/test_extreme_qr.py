import numpy as np
import pytest

from quant_fund.models.extreme_qr import (
    bench_extreme_qr,
    extremal_qr,
    hill_tail_index,
    synth_extreme_qr,
)


def test_bench_extreme_qr_passes():
    r = bench_extreme_qr()
    assert r["score"] == 1.0


def test_hill_positive_on_pareto():
    rng = np.random.default_rng(0)
    x = rng.pareto(4.0, size=2000) + 1.0
    xi = hill_tail_index(x, k=50)
    assert 0.05 < xi < 1.0


def test_extreme_deeper_than_intermediate():
    y, x = synth_extreme_qr(seed=3)
    r = extremal_qr(y, x[:, None])
    assert r["q_extreme_mean"] < r["q_intermediate_mean"]


def test_extrap_factor_gt_one():
    y, x = synth_extreme_qr(seed=8)
    r = extremal_qr(y, x[:, None])
    assert r["extrap_factor"] > 1.0


def test_rejects_bad_taus():
    y, x = synth_extreme_qr(seed=1)
    with pytest.raises(ValueError):
        extremal_qr(y, x[:, None], tau_n=0.01, tau_out=0.10)


def test_rejects_short_series():
    with pytest.raises(ValueError):
        extremal_qr(np.zeros(50), np.zeros((50, 1)))
