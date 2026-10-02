"""IMM/PDA tracking tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.bayesian_tracking import (
    bench_tracking,
    imm_filter,
    pda_update,
)


def _cv_scenario(seed: int = 0, n: int = 50):
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 2))
    vel = np.array([5.0, 2.0])
    for k in range(1, n):
        pos[k] = pos[k - 1] + vel
    meas = [pos[k] + rng.normal(0, 1.5, 2)[None, :] for k in range(n)]
    return pos, meas


def test_pda_gates_clutter():
    xp = np.zeros(4)
    Pp = np.eye(4) * 10
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    R = np.eye(2)
    zs = np.array([[0.5, 0.3], [500.0, 500.0]])  # far clutter gated out
    res = pda_update(xp, Pp, zs, H, R)
    assert np.asarray(res["betas"]).size == 1


def test_pda_missed_detection():
    xp = np.zeros(4)
    Pp = np.eye(4)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    res = pda_update(xp, Pp, np.zeros((0, 2)), H, np.eye(2))
    assert res["missed_prob"] == 1.0


def test_imm_cv_tracking():
    pos, meas = _cv_scenario()
    out = imm_filter(meas)
    est = out["track"][:, :2]
    rms = float(np.sqrt(((est - pos[1:]) ** 2).sum(1).mean()))
    assert rms < 3.0


def test_imm_mode_probs_sum():
    _, meas = _cv_scenario()
    out = imm_filter(meas)
    assert np.allclose(out["mode_probs"].sum(1), 1.0)


def test_bench_tracking():
    out = bench_tracking()
    assert out["synthetic_rms_pos"] < 8.0
