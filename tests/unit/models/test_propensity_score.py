"""Adversarial probes for propensity_score."""

import numpy as np
import pytest

from quant_fund.models import propensity_score as psm


def _data(n: int = 400, seed: int = 0):
    d = psm.synth_propensity(n=n, seed=seed)
    return d["y"], d["d"], d["x"]


def test_ps_match_rejects_nan_in_d():
    y, d, x = _data()
    d = d.copy()
    d[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        psm.ps_match(y, d, x)


def test_ipw_rejects_nonbinary_d():
    y, d, _ = _data()
    d = d.copy()
    d[0] = 2.0
    ps = np.full(y.size, 0.5)
    with pytest.raises(ValueError, match="binary"):
        psm.ipw_ate(y, d, ps)


def test_ipw_rejects_d_length_mismatch():
    y, d, _ = _data()
    ps = np.full(y.size, 0.5)
    with pytest.raises(ValueError, match="match"):
        psm.ipw_ate(y, np.concatenate([d, [1.0]]), ps)


def test_ipw_rejects_negative_trim():
    y, d, _ = _data()
    ps = np.full(y.size, 0.5)
    with pytest.raises(ValueError, match="trim"):
        psm.ipw_ate(y, d, ps, trim=-0.1)


def test_overlap_rejects_nan_ps():
    y, d, _ = _data()
    ps = np.full(y.size, 0.5)
    ps[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        psm.overlap_ate(y, d, ps)


def test_overlap_fails_closed_on_degenerate_ps():
    """All treated have e=1 -> treated overlap weight is 0 -> raise."""
    y, d, _ = _data()
    ps = np.where(d == 1, 1.0, 0.5)
    with pytest.raises(ValueError, match="overlap"):
        psm.overlap_ate(y, d, ps)


def test_overlap_rejects_ps_out_of_range():
    y, d, _ = _data()
    ps = np.full(y.size, 1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        psm.overlap_ate(y, d, ps)


def test_bench_smoke():
    out = psm.bench_propensity_score()
    assert out["synthetic_determinism"] == 1.0
    assert out["synthetic_ipw_beats_raw"] == 1.0
