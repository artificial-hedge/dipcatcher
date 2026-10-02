"""Tests for echo_state (wave-58)."""

import numpy as np
import pytest

from quant_fund.models.echo_state import (
    bench_echo_state,
    echo_state_esn,
    synth_esn,
)


def test_narma_learnable() -> None:
    y, _ = synth_esn(seed=1)
    r = echo_state_esn(y, n_res=60, washout=200, seed=5)
    assert r["nmse"] < 0.7


def test_iid_unlearnable() -> None:
    _, iid = synth_esn(seed=2)
    r = echo_state_esn(iid, n_res=60, washout=200, seed=5)
    assert r["nmse"] > 0.8


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        echo_state_esn(np.ones(50))
    with pytest.raises(ValueError):
        echo_state_esn(np.full(500, np.nan))
    with pytest.raises(ValueError):
        echo_state_esn(np.ones(300), sr=2.0)


def test_determinism() -> None:
    y, _ = synth_esn(seed=3)
    assert echo_state_esn(y, n_res=40, seed=9) == echo_state_esn(y, n_res=40, seed=9)


def test_bench_schema_and_score() -> None:
    r = bench_echo_state()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["score"] == 1.0
