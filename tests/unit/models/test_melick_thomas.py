"""Tests for melick_thomas — implied PDF recovery."""

import numpy as np
import pytest

from quant_fund.models.melick_thomas import (
    bench_melick_thomas,
    implied_pdf,
    synth_smile,
)


def test_recovers_disaster_mass() -> None:
    d = synth_smile(seed=1)
    r = implied_pdf(
        np.asarray(d["strikes"]),
        np.asarray(d["calls"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    assert abs(float(r["tail_mass"]) - float(d["tail_true"])) < 0.05
    assert float(r["rmse_rel"]) < 0.01


def test_recovers_implied_variance() -> None:
    d = synth_smile(seed=2, disaster_w=0.25)
    r = implied_pdf(
        np.asarray(d["strikes"]),
        np.asarray(d["calls"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    assert abs(float(r["var_ln"]) - float(d["var_true"])) / float(d["var_true"]) < 0.25


def test_weights_simplex_and_positive_sds() -> None:
    d = synth_smile(seed=3)
    r = implied_pdf(
        np.asarray(d["strikes"]),
        np.asarray(d["calls"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    w = np.asarray(r["weights"])
    s = np.asarray(r["sds"])
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0.0)
    assert np.all(s > 0.0)


def test_no_disaster_low_tail() -> None:
    d = synth_smile(seed=4, disaster_w=0.0)
    r = implied_pdf(
        np.asarray(d["strikes"]),
        np.asarray(d["calls"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    assert float(r["tail_mass"]) < 0.1


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        implied_pdf(np.ones(3), np.ones(3), 100.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        implied_pdf(-np.ones(10), np.ones(10), 100.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        implied_pdf(np.ones(10), np.ones(10), -100.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        implied_pdf(np.ones(10), np.ones(10), 100.0, 0.0, -0.5)


def test_determinism() -> None:
    d = synth_smile(seed=6)
    a = implied_pdf(
        np.asarray(d["strikes"]),
        np.asarray(d["calls"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    b = implied_pdf(
        np.asarray(d["strikes"]),
        np.asarray(d["calls"]),
        float(d["spot"]),
        float(d["rate"]),
        float(d["tau"]),
    )
    np.testing.assert_array_equal(a["weights"], b["weights"])
    assert a["tail_mass"] == b["tail_mass"]


def test_bench_schema_and_score() -> None:
    r = bench_melick_thomas()
    for k in ("tail_hat", "tail_true", "tail_err", "var_err_rel", "rmse_rel", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
