"""Tests for microstructure/hmm_learn.py — Baum–Welch on the tape."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.hmm_learn import (
    HMM_LEARN_SCHEMA,
    baum_welch,
    discretize_tape,
    hmm_learn_bench,
)
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    RegimeState,
    ZILobConfig,
)


def _planted_obs(n: int = 1500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = np.array([[0.97, 0.03], [0.06, 0.94]])
    B = np.array([[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.3, 0.7]])
    s = 0
    out = np.empty(n, dtype=np.int64)
    for t in range(n):
        out[t] = rng.choice(4, p=B[s])
        s = rng.choice(2, p=A[s])
    return out


def test_baum_welch_recovers_planted_chain() -> None:
    obs = _planted_obs()
    fit = baum_welch(obs, 4, seed=0)
    hi = int(np.argmax(fit.B[:, 1] + fit.B[:, 3]))
    lo = 1 - hi
    assert fit.A[hi, hi] == pytest.approx(0.94, abs=0.03)
    assert fit.A[lo, lo] == pytest.approx(0.97, abs=0.03)
    assert fit.B[hi, 3] > 0.5  # trend state emits hi-buy symbol
    assert fit.B[lo, 0] > 0.5  # calm state emits the lo symbol


def test_loglik_nondecreasing() -> None:
    obs = _planted_obs(n=800)
    fit = baum_welch(obs, 4, seed=1, n_iter=30)
    tr = np.asarray(fit.loglik_trace)
    assert tr.size >= 2
    assert np.all(np.diff(tr) >= -1e-6)


def test_baum_welch_deterministic() -> None:
    obs = _planted_obs(n=400)
    a = baum_welch(obs, 4, seed=5, n_iter=10)
    b = baum_welch(obs, 4, seed=5, n_iter=10)
    np.testing.assert_array_equal(a.A, b.A)
    np.testing.assert_array_equal(a.B, b.B)
    assert a.loglik_trace == b.loglik_trace


def test_baum_welch_validation() -> None:
    with pytest.raises(ValueError, match=">=10"):
        baum_welch(np.asarray([0, 1, 0]), 4)
    with pytest.raises(ValueError, match="symbol range"):
        baum_welch(np.zeros(50, dtype=np.int64) + 9, 4)


def _flow(seed: int) -> MarkovRegimeFlow:
    return MarkovRegimeFlow(
        states=[
            RegimeState(name="calm", intensity_mult=1.0, p_buy=0.5),
            RegimeState(name="trend", intensity_mult=1.6, p_buy=0.8),
        ],
        stay_probs=[0.97, 0.94],
        seed=seed,
    )


def test_discretize_deterministic_and_binary_grid() -> None:
    cfg = ZILobConfig(seed=11, init_depth=8, band=8)
    a = discretize_tape(config=cfg, horizon=300.0, flow=_flow(11))
    b = discretize_tape(config=cfg, horizon=300.0, flow=_flow(11))
    np.testing.assert_array_equal(a, b)
    assert set(np.unique(a)) <= {0, 1, 2, 3}


def test_bench_smoke_and_schema() -> None:
    out = hmm_learn_bench(n_seeds=2, horizon=400.0, window_s=20.0)
    assert out["schema"] == HMM_LEARN_SCHEMA
    assert "likelihood_monotone" in out
    assert out["planted"]["p_buy_trend"] == 0.8
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["payload_sha256"]) == 64
