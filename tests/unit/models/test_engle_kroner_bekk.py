"""Tests for engle_kroner_bekk — diagonal BEKK(1,1)."""

import numpy as np
import pytest

from quant_fund.models.engle_kroner_bekk import (
    _h_path,
    bench_engle_kroner_bekk,
    fit_bekk,
    synth_bekk,
)


def test_h_path_positive_definite() -> None:
    d = synth_bekk(seed=1, t=400)
    e = np.asarray(d["e"])
    h = _h_path(e, np.asarray(d["C_true"]), np.array([0.25, 0.3]), np.array([0.9, 0.88]))
    for i in range(50, 400, 50):
        assert np.all(np.linalg.eigvalsh(h[i]) > 0.0)


def test_filter_reproduces_true_path() -> None:
    d = synth_bekk(seed=2)
    h = _h_path(
        np.asarray(d["e"]),
        np.asarray(d["C_true"]),
        np.array([0.25, 0.3]),
        np.array([0.9, 0.88]),
    )
    ht = np.asarray(d["H_true"])
    assert np.max(np.abs(h[50:] - ht[50:])) < 1e-4


def test_fit_recovers_persistence() -> None:
    d = synth_bekk(seed=3)
    r = fit_bekk(np.asarray(d["e"]))
    assert 0.7 < r["persistence"] < 1.0
    assert np.all(np.asarray(r["a"]) != 0.0)


def test_corr_last_near_truth() -> None:
    d = synth_bekk(seed=4)
    r = fit_bekk(np.asarray(d["e"]))
    ht = np.asarray(d["H_true"])
    rho_t = ht[-1, 0, 1] / np.sqrt(ht[-1, 0, 0] * ht[-1, 1, 1])
    assert abs(float(np.asarray(r["corr_last"])[0, 1]) - rho_t) < 0.2


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        fit_bekk(np.ones((50, 2)))
    with pytest.raises(ValueError):
        fit_bekk(np.ones((200, 1)))
    with pytest.raises(ValueError):
        fit_bekk(np.full((200, 2), np.nan))


def test_determinism() -> None:
    d = synth_bekk(seed=6)
    e = np.asarray(d["e"])
    a = fit_bekk(e)
    b = fit_bekk(e)
    np.testing.assert_array_equal(a["a"], b["a"])
    np.testing.assert_array_equal(a["H"][-1], b["H"][-1])


def test_bench_schema_and_score() -> None:
    r = bench_engle_kroner_bekk()
    for k in ("var_rel_rmse", "rho_err", "persistence", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
