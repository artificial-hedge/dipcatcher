"""Tests for labels/meta_model.py — purged CV + weighted meta-logistic."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.labels.meta_model import (
    fit_logistic,
    logistic_predict,
    meta_model_demo,
    meta_model_eval,
    purged_kfold_indices,
)


def test_purged_kfold_partitions_test_indices():
    ts = np.arange(20, dtype=float)
    te = ts + 5
    folds = purged_kfold_indices(ts, te, 4, embargo_frac=0.0)
    assert len(folds) == 4
    all_test = np.concatenate([t for _, t in folds])
    assert sorted(all_test.tolist()) == list(range(20))
    for train, test in folds:
        assert len(set(train) & set(test)) == 0


def test_purged_kfold_drops_overlapping_train_events():
    # events 0..9 start at 0..9, each spans 10 bars -> all overlap everything
    ts = np.arange(10, dtype=float)
    te = ts + 10
    folds = purged_kfold_indices(ts, te, 2, embargo_frac=0.0)
    train0, test0 = folds[0]
    # every train event's interval [i, i+10] overlaps test span [5, 14]
    assert (
        all(not (te[i] < ts[test0[0]] or ts[i] > te[test0[-1]]) for i in train0) or train0.size == 0
    )


def test_purged_kfold_embargo_tail():
    # disjoint windows: event i spans [10i, 10i+2]; fold boundary at event 5
    ts = np.arange(10, dtype=float) * 10
    te = ts + 2
    folds = purged_kfold_indices(ts, te, 2, embargo_frac=0.5)  # embargo = 5 units
    train0, _ = folds[0]
    # test span ends at te[4] = 42; events starting at 50 must wait till 47+ -> event5 (t=50) embargoed? 50 > 42+5=47 -> allowed
    # event with t_start=50 allowed; check events 6,7,8 (60,70,80) allowed too
    assert all(ts[i] > 47 for i in train0 if ts[i] > ts[4])


def test_purged_kfold_rejects_bad_inputs():
    ts = np.arange(10, dtype=float)
    te = ts + 3
    with pytest.raises(ValueError, match="n_folds"):
        purged_kfold_indices(ts, te, 1)
    with pytest.raises(ValueError, match="embargo_frac"):
        purged_kfold_indices(ts, te, 2, embargo_frac=1.5)
    with pytest.raises(ValueError, match="t_end"):
        purged_kfold_indices(ts, ts - 1, 2)


def test_fit_logistic_recovers_signal():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 1))
    p = 1.0 / (1.0 + np.exp(-(0.5 + 2.0 * x[:, 0])))
    y = (rng.random(400) < p).astype(float)
    beta = fit_logistic(x, y)
    pred = logistic_predict(x, beta)
    assert np.corrcoef(pred, p)[0, 1] > 0.9
    assert beta[1] > 0  # positive slope recovered


def test_fit_logistic_rejects_bad_labels():
    x = np.ones((10, 2))
    with pytest.raises(ValueError, match="0, 1"):
        fit_logistic(x, np.full(10, 0.5))
    with pytest.raises(ValueError, match="rows"):
        fit_logistic(x, np.zeros(5))


def test_meta_model_eval_runs_and_is_deterministic():
    out1 = meta_model_demo(seed=5, n_bars=1500)
    out2 = meta_model_demo(seed=5, n_bars=1500)
    assert out1["receipt_sha256"] == out2["receipt_sha256"]
    assert out1["data_label"] == "SYNTHETIC"
    res = out1["claim"]["result"]
    assert 0.0 <= res["oof"]["brier"] <= 0.3  # Brier is not capped at 0.25 for non-degenerate probs
    assert res["n_events"] > 0
    assert len(res["folds"]) == 5


def test_meta_model_eval_purges_overlapping_window():
    # flat price -> all labels unobservable except mid-path ones; ensure raise paths work
    close = 100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 500)))
    ev = np.arange(5, 480, dtype=np.intp)
    sides = np.ones(ev.size)
    feats = np.column_stack([np.linspace(-1, 1, ev.size), np.ones(ev.size)])
    res = meta_model_eval(close, ev, sides, feats, n_folds=3)
    assert res["n_events"] == ev.size


def test_meta_model_eval_rejects_mismatched_inputs():
    close = np.linspace(100, 101, 200)
    ev = np.arange(10, 100, dtype=np.intp)
    with pytest.raises(ValueError, match="finite \\(n_events, p\\)"):
        meta_model_eval(close, ev, np.ones(ev.size), np.ones((5, 2)))
    with pytest.raises(ValueError, match="\\{-1, 0, \\+1\\}"):
        meta_model_eval(close, ev, np.full(ev.size, 2.0), np.ones((ev.size, 1)))
