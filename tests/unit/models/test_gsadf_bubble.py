"""Unit tests for quant_fund.models.gsadf_bubble."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.gsadf_bubble import (
    adf_tstat,
    bench_gsadf,
    date_stamp,
    gsadf,
    gsadf_critical_values,
    synth_gsadf,
)


def test_adf_negative_on_stationary() -> None:
    rng = np.random.default_rng(1)
    y = np.empty(80)
    y[0] = 0.0
    for i in range(1, 80):
        y[i] = 0.6 * y[i - 1] + rng.normal(0, 1)
    stat = adf_tstat(y)
    assert stat < 0.0


def test_adf_positive_on_explosive() -> None:
    rng = np.random.default_rng(2)
    y = np.empty(80)
    y[0] = 50.0
    for i in range(1, 80):
        y[i] = 1.05 * y[i - 1] + rng.normal(0, 1)
    assert adf_tstat(y) > 2.0


def test_gsadf_exceeds_sadf() -> None:
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.standard_normal(120))
    out = gsadf(y)
    assert out["gsadf"] >= out["sadf"] - 1e-9
    bs = np.asarray(out["bsadf"])
    assert np.all(np.isnan(bs[: int(out["min_window"])]))


def test_date_stamp_merges() -> None:
    bs = np.array([np.nan, np.nan, 3.0, 3.1, 0.0, 3.2, 3.4, 0.0, 3.5])
    flags = date_stamp(bs, cv=2.5, min_len=2)
    assert flags[2] == 1.0 and flags[3] == 1.0
    assert flags[5] == 1.0 and flags[6] == 1.0
    assert flags[8] == 0.0  # singleton dropped


def test_critical_values_ordered() -> None:
    cvs = gsadf_critical_values(120, n_rep=60, seed=7)
    assert cvs["gsadf_q90"] < cvs["gsadf_q95"] < cvs["gsadf_q99"]
    assert cvs["gsadf_q95"] > 0.0


def test_critical_values_deterministic() -> None:
    a = gsadf_critical_values(100, n_rep=50, seed=11)
    b = gsadf_critical_values(100, n_rep=50, seed=11)
    assert a == b


def test_synth_episode_detected() -> None:
    d = synth_gsadf(seed=8)
    out = gsadf(np.asarray(d["y"]))
    cvs = gsadf_critical_values(400, n_rep=80, seed=9)
    flags = date_stamp(np.asarray(out["bsadf"]), cvs["gsadf_q95"], min_len=3)
    episode = np.asarray(d["episode"])
    assert float(np.sum(flags * episode)) >= 15.0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        gsadf(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        gsadf(np.full(60, np.nan))
    with pytest.raises(ValueError):
        adf_tstat(np.ones(30))
    with pytest.raises(ValueError):
        gsadf(np.cumsum(np.ones(60)), r0=0.99)


def test_bench_score() -> None:
    out = bench_gsadf()
    assert out["synthetic_score"] == 1.0
    assert set(out) == {
        "synthetic_gsadf",
        "synthetic_cv95",
        "synthetic_overlap",
        "synthetic_flagged",
        "synthetic_tranquil_fp",
        "synthetic_score",
    }
