"""Tests for pesaran_timmermann — market-timing direction tests."""

import numpy as np
import pytest

from quant_fund.models.pesaran_timmermann import (
    bench_pesaran_timmermann,
    hm_test,
    pt_test,
    synth_pt,
)


def test_accepts_independent() -> None:
    y, f = synth_pt(seed=1, rho=0.0)
    r = pt_test(y, f)
    assert r["p_value"] > 0.05


def test_rejects_skillful() -> None:
    y, f = synth_pt(seed=2, rho=0.45)
    r = pt_test(y, f)
    assert r["p_value"] < 0.01
    assert r["s_stat"] > 2.0
    assert r["p_hat"] > r["p_star"]


def test_rejects_reverse_skill() -> None:
    y, f = synth_pt(seed=3, rho=0.45)
    r = pt_test(y, -f)
    assert r["p_value"] > 0.5
    assert r["p_hat"] < r["p_star"]


def test_hm_exact_tail() -> None:
    y, f = synth_pt(seed=4, rho=0.45)
    r = hm_test(y, f)
    assert r["hm_p"] < 0.01
    y0, f0 = synth_pt(seed=5, rho=0.0)
    r0 = hm_test(y0, f0)
    assert r0["hm_p"] > r["hm_p"]


def test_fail_closed() -> None:
    y, f = synth_pt(seed=6)
    with pytest.raises(ValueError):
        pt_test(y[:20], f[:20])
    with pytest.raises(ValueError):
        pt_test(y, f[:100])
    with pytest.raises(ValueError):
        pt_test(np.full_like(y, np.nan), f)
    with pytest.raises(ValueError):
        pt_test(np.ones(500), f)  # degenerate sign split
    with pytest.raises(ValueError):
        hm_test(y[:20], f[:20])


def test_determinism() -> None:
    y, f = synth_pt(seed=7)
    a = pt_test(y, f)
    b = pt_test(y, f)
    assert a == b


def test_bench_schema_and_score() -> None:
    r = bench_pesaran_timmermann()
    for k in ("p_null", "p_alt", "hm_p", "p_hat_alt", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
