"""Wave-272 control-theory-2 module tests."""

import numpy as np

from quant_fund.models.backstepping import backstep
from quant_fund.models.pid_antiwindup import pid_run
from quant_fund.models.repetitive_ctrl import rep_run
from quant_fund.models.sliding_mode import smc_run
from quant_fund.models.smith_predictor import smith_run


def test_pid_tracks_setpoint() -> None:
    out = pid_run(-0.5, 1.0, 400, 5.0, 8.0, 10.0, antiwindup=True)
    assert abs(out[-1] - 1.0) < 0.1


def test_pid_no_overshoot_without_sat() -> None:
    out = pid_run(-0.5, 1.0, 200, 0.5, 0.1, 100.0, antiwindup=True)
    assert out.max() < 1.2


def test_smc_reaches_zero() -> None:
    h = smc_run(np.array([1.0, 0.5]), 1.0, 8.0, 600)
    assert np.linalg.norm(h[-1]) < 0.5


def test_smc_finite() -> None:
    h = smc_run(np.array([-1.0, 1.0]), 1.0, 8.0, 100)
    assert np.isfinite(h).all()


def test_smith_finite_and_tracks() -> None:
    out = smith_run(8, 400, use_smith=True)
    assert np.isfinite(out).all()
    assert abs(out[-1] - 1.0) < 0.3


def test_backstep_converges() -> None:
    h = backstep(np.array([1.0, -0.5]), 800)
    assert h[-1] < 0.5


def test_repetitive_learns() -> None:
    h = rep_run(40, 400)
    assert h[-40:].mean() < h[:40].mean()


def test_repetitive_bounded() -> None:
    h = rep_run(20, 200)
    assert np.isfinite(h).all()
