import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.combination.member_health import (
    leave_one_out_contribution,
    member_health_report,
    member_loss_drift,
    member_pinball,
)

pytestmark = pytest.mark.synthetic

TAUS = np.linspace(0.1, 0.9, 9)


def _members(seed: int, n: int = 2500, drift_at: int | None = None):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    z = norm.ppf(TAUS)
    good = y[:, None] + 0.2 * rng.standard_normal((n, len(TAUS))) + z[None, :] * 0.5
    ok = y[:, None] + 0.7 * rng.standard_normal((n, len(TAUS))) + z[None, :] * 0.5
    degrading = y[:, None] + 0.2 * rng.standard_normal((n, len(TAUS))) + z[None, :] * 0.5
    if drift_at is not None:
        degrading[drift_at:] += 1.5 * rng.standard_normal((n - drift_at, len(TAUS)))
    return np.stack([good, ok, degrading], axis=1), y


def test_member_pinball_orders_quality() -> None:
    q, y = _members(90)
    pin = member_pinball(q, y, TAUS)
    assert pin.shape == (3,)
    assert pin[0] < pin[1]


def test_loo_contribution_flags_noisy_member() -> None:
    q, y = _members(91)
    # member 2: same signal, 3x noise — it hurts the ensemble
    rng = np.random.default_rng(91)
    q[:, 2, :] = (
        y[:, None] + 3.0 * rng.standard_normal((len(y), len(TAUS))) + norm.ppf(TAUS)[None, :] * 0.5
    )
    out = leave_one_out_contribution(q, y, TAUS)
    c = np.asarray(out["contribution"])
    assert c.shape == (3,)
    assert c[2] < 0.0  # dropping it lowers ensemble loss
    assert c[0] > 0.0  # the best member contributes positively


def test_loo_contribution_deterministic_hand_check() -> None:
    # hand-computable: equal-weight ensemble of two members
    rng = np.random.default_rng(94)
    n = 2000
    y = rng.standard_normal(n)
    z = norm.ppf(TAUS)
    good = y[:, None] + 0.1 * rng.standard_normal((n, len(TAUS))) + z[None, :] * 0.3
    bad = y[:, None] + 2.5 * rng.standard_normal((n, len(TAUS))) + z[None, :] * 0.3
    q = np.stack([good, bad], axis=1)
    out = leave_one_out_contribution(q, y, TAUS)
    c = np.asarray(out["contribution"])
    # dropping the bad member must lower loss → negative contribution
    assert c[1] < 0.0
    assert c[0] > 0.0


def test_member_loss_drift_flags_degrading() -> None:
    q, y = _members(92, drift_at=1500)
    out = member_loss_drift(q, y, TAUS, pre_period=200, threshold=5.5, slack=1.0)
    alarm = np.asarray(out["alarm_time"])
    assert alarm[2] > 0  # degrading member alarms
    assert alarm[0] < 0  # stable member stays quiet
    assert alarm[1] < 0  # noisier-but-stable member also stays quiet


def test_health_report_bundle() -> None:
    q, y = _members(93, drift_at=2000)
    out = member_health_report(q, y, TAUS, drift_threshold=5.5, pre_period=200)
    for key in ("pinball", "contribution", "drift_alarm", "healthy"):
        assert key in out
    h = np.asarray(out["healthy"])
    assert h.shape == (3,)
    assert h[0] == 1.0


def test_validation() -> None:
    with pytest.raises(ValueError):
        member_pinball(np.zeros((10, 2, 3)), np.zeros(9), TAUS)
    with pytest.raises(ValueError):
        leave_one_out_contribution(np.zeros((10, 2, 3)), np.zeros(10), np.array([0.5, 0.5]))
