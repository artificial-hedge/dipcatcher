"""Tests for recurrence quantification analysis (metrics/rqa.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.rqa import (
    bench_rqa,
    embed_series,
    mutual_information_delay,
    recurrence_plot,
    rqa_measures,
    synth_ar,
    synth_iid,
    synth_lorenz,
    synth_periodic,
)


def test_embed_series_shape():
    x = np.arange(50, dtype=float)
    emb = embed_series(x, m=3, tau=2)
    assert emb.shape == (50 - 4, 3)
    assert np.allclose(emb[0], [0.0, 2.0, 4.0])
    assert np.allclose(emb[5], [5.0, 7.0, 9.0])


def test_embed_validation():
    with pytest.raises(ValueError):
        embed_series(np.zeros(5), m=4, tau=2)
    with pytest.raises(ValueError):
        embed_series(np.arange(30.0), m=0)
    with pytest.raises(ValueError):
        embed_series(np.concatenate([np.zeros(30), [np.nan]]))


def test_recurrence_plot_self_similar():
    x = np.sin(np.linspace(0, 20, 200))
    rp = recurrence_plot(x, m=1, tau=1, eps=0.5)
    assert rp.shape == (200, 200)
    # identical series is fully self-recurrent on the diagonal
    assert np.diag(rp).mean() == 1.0
    assert rp.mean() > 0.3


def test_recurrence_plot_disjoined():
    x = np.concatenate([np.zeros(100), np.ones(100) * 100.0])
    rp = recurrence_plot(x, m=1, tau=1, eps=1.0)
    # cross-cluster block should be ~empty
    assert rp[:100, 100:].mean() < 0.01


def test_rqa_periodic_vs_iid():
    per = rqa_measures(synth_periodic(400, seed=1), m=3, tau=3)
    iid = rqa_measures(synth_iid(400, seed=1), m=3, tau=3)
    assert per["det"] > 0.8
    assert iid["det"] < 0.5
    assert per["det"] > iid["det"]


def test_rqa_lorenz_high_det():
    lor = rqa_measures(synth_lorenz(600, seed=2), m=3, tau=5)
    assert lor["det"] > 0.9
    assert lor["lmax"] > 10


def test_rqa_output_keys():
    out = rqa_measures(synth_periodic(200, seed=0))
    for k in ("rr", "det", "lam", "lmax", "entr", "tt", "n_recur"):
        assert k in out
        assert math.isfinite(out[k])
    assert 0.0 <= out["det"] <= 1.0
    assert 0.0 <= out["lam"] <= 1.0
    assert 0.0 < out["rr"] < 1.0


def test_mi_delay_periodic():
    tau = mutual_information_delay(synth_periodic(500, freq=0.2, seed=0), max_tau=30)
    assert 1 <= tau <= 30


def test_rqa_determinism():
    a = rqa_measures(synth_ar(300, seed=11), m=3, tau=1)
    b = rqa_measures(synth_ar(300, seed=11), m=3, tau=1)
    assert all(a[k] == b[k] for k in a)


def test_synth_series_shapes():
    assert synth_periodic(200).shape == (200,)
    assert synth_iid(200).shape == (200,)
    assert synth_lorenz(300).shape == (300,)
    assert synth_ar(200).shape == (200,)
    assert np.all(np.isfinite(synth_lorenz(300)))


def test_bench_keys():
    out = bench_rqa()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_det_periodic"] > 0.7
    assert out["synthetic_det_iid"] < 0.5
    assert out["synthetic_det_lorenz"] > 0.9
    assert out["synthetic_det_margin_order"] > 0.5
    assert out["synthetic_determinism"] == 1.0
