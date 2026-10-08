"""Tests for ivx (wave-57)."""

import numpy as np
import pytest

from quant_fund.models.ivx import bench_ivx, ivx_wald, synth_ivx


def test_coupled_predictor_rejects() -> None:
    x, y, _ = synth_ivx(seed=1)
    r = ivx_wald(y[1:], x[1:])
    assert r["reject5"] == 1.0
    assert r["wald"] > 10


def test_null_predictor_accepts() -> None:
    x, _, y_null = synth_ivx(seed=1)
    r = ivx_wald(y_null[1:], x[1:])
    assert r["reject5"] == 0.0


def test_fail_closed() -> None:
    x, y, _ = synth_ivx(seed=3)
    with pytest.raises(ValueError):
        ivx_wald(y[:50], x[:50])
    with pytest.raises(ValueError):
        ivx_wald(np.full(200, np.nan), x)
    with pytest.raises(ValueError):
        ivx_wald(y, np.ones(400))


def test_determinism() -> None:
    x, y, _ = synth_ivx(seed=4)
    assert ivx_wald(y[1:], x[1:]) == ivx_wald(y[1:], x[1:])


def test_bench_schema_and_score() -> None:
    r = bench_ivx()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["synthetic_score"] == 1.0
