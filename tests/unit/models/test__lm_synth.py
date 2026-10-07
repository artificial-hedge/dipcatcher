"""Adversarial probes for quant_fund.models._lm_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._lm_synth import recall_batch, regime_task


def test_recall_batch_deterministic() -> None:
    x1, y1 = recall_batch(11, B=8, T=12)
    x2, y2 = recall_batch(11, B=8, T=12)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_recall_batch_semantics() -> None:
    x, y = recall_batch(3, B=16, T=12)
    for b in range(16):
        row = x[b]
        # planted structure: key at i1, val at i1+1, key again at i2 >= i1+2.
        # Existence check: some i has row[i] repeated strictly after i+1 and
        # row[i+1] == y[b].
        ok = any(
            int(row[i + 1]) == int(y[b]) and (row[i + 2 :] == row[i]).any()
            for i in range(len(row) - 2)
        )
        assert ok, f"row {b} lacks the key->val->key parse"


def test_recall_batch_hostile_params() -> None:
    with pytest.raises(ValueError):
        recall_batch(0, B=0)
    with pytest.raises(ValueError):
        recall_batch(0, B=4, T=4)


def test_regime_task_deterministic_and_shapes() -> None:
    x1, y1 = regime_task(5, n=64, k=3, d=6)
    x2, y2 = regime_task(5, n=64, k=3, d=6)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert x1.shape == (64, 6)
    assert set(np.unique(y1)) <= {0.0, 1.0}


def test_regime_task_hostile_params() -> None:
    with pytest.raises(ValueError):
        regime_task(0, n=0)
    with pytest.raises(ValueError):
        regime_task(0, k=0)
    with pytest.raises(ValueError):
        regime_task(0, d=0)


def test_attn_baseline_iters_guard() -> None:
    pytest.importorskip("torch")
    from quant_fund.models._lm_synth import attn_baseline

    with pytest.raises(ValueError):
        attn_baseline(0, iters=0)
