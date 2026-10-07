"""Probes for _queue_synth."""

import numpy as np
import pytest

from quant_fund.models._queue_synth import GAMMA, MU, jackson_lam, mm1_sim


def test_mm1_sim_stable_deterministic():
    a = mm1_sim(0, lam=1.0, mu=2.0, horizon=3000.0)
    b = mm1_sim(0, lam=1.0, mu=2.0, horizon=3000.0)
    assert a == b
    # M/M/1 theory: L = rho/(1-rho) = 1.0 for rho=0.5
    assert a == pytest.approx(1.0, abs=0.35)


def test_mm1_sim_unstable_queue_grows():
    # rho > 1 → mean queue length exceeds stable case (still finite over horizon)
    stable = mm1_sim(0, lam=1.0, mu=4.0, horizon=2000.0)
    heavy = mm1_sim(0, lam=3.0, mu=4.0, horizon=2000.0)
    assert heavy > stable


@pytest.mark.parametrize(
    "lam,mu,horizon",
    [
        (0.0, 1.0, 10.0),  # lam=0 → div-by-zero / degenerate arrivals
        (-1.0, 1.0, 10.0),
        (np.nan, 1.0, 10.0),
        (1.0, 0.0, 10.0),
        (1.0, -2.0, 10.0),
        (1.0, 1.0, 0.0),  # zero horizon → 0/0 NaN laundering
        (1.0, 1.0, -5.0),
        (1.0, 1.0, np.inf),
    ],
)
def test_mm1_sim_hostile(lam, mu, horizon):
    with pytest.raises(ValueError):
        mm1_sim(0, lam, mu, horizon)


def test_jackson_lam_fixed_point():
    lam = jackson_lam()
    assert lam.shape == (3,)
    # converged: lam = gamma + P^T lam
    from quant_fund.models._queue_synth import P

    np.testing.assert_allclose(lam, GAMMA + P.T @ lam, atol=1e-10)
    assert (lam <= MU).all()  # stable routing must not exceed service
