import numpy as np
import pytest

from quant_fund.models.kalman_em import bench_kalman_em, kalman_em, synth_kalman_em


def test_bench_kalman_em_passes():
    r = bench_kalman_em()
    assert r["synthetic_score"] == 1.0


def test_em_recovers_phi():
    y, _ = synth_kalman_em(seed=5)
    r = kalman_em(y)
    assert abs(r["phi"] - 0.85) < 0.2


def test_em_variances_positive():
    y, _ = synth_kalman_em(seed=2)
    r = kalman_em(y)
    assert r["q"] > 0 and r["r"] > 0


def test_em_rejects_short():
    with pytest.raises(ValueError):
        kalman_em(np.arange(50.0))


def test_em_deterministic():
    y, _ = synth_kalman_em(seed=9)
    a = kalman_em(y)
    b = kalman_em(y)
    assert a == b
