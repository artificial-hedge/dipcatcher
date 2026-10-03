"""Unit tests for wave-286 security-defensive canon modules."""

import numpy as np

from quant_fund.models.beacon_detect import _cv
from quant_fund.models.cred_stuffing import stuffing_score
from quant_fund.models.entropy_dns import label_entropy
from quant_fund.models.exfil_zscore import _robust_z
from quant_fund.models.impossible_travel import _hav, flag
from quant_fund.models.sig_score import score


def test_beacon_cv_periodic():
    t = np.cumsum(np.full(500, 60.0))
    assert _cv(t) < 1e-9


def test_dns_entropy_levels():
    assert label_entropy("mail") < label_entropy("x7k2p9q4z8w1")


def test_stuffing_score_zero_when_no_fails():
    assert stuffing_score(np.zeros(4), np.zeros(4, dtype=int), np.ones(4)) == 0.0


def test_haversine_known():
    assert abs(_hav(51.5, -0.12, 48.85, 2.35) - 340) < 20  # london->paris ~340km


def test_flag_clear():
    assert flag([(0.0, 0.0, 0.0), (0.01, 0.01, 1.0)]) == []


def test_exfil_robust_z():
    x = np.array([1.0] * 10 + [50.0])
    assert _robust_z(x)[-1] > 10


def test_score_threshold():
    s, hit = score({"a": True, "b": False}, {"a": 2.0, "b": 1.0}, 1.5)
    assert s == 2.0 and hit
