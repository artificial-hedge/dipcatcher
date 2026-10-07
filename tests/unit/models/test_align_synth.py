"""Unit tests for quant_fund.models._align_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._align_synth import (
    N_ACT,
    action_embeddings,
    best_action_rate,
    contexts,
    pref_pairs,
    true_reward,
)


def test_embeddings_and_contexts_unit_norm() -> None:
    rng = np.random.default_rng(0)
    emb = action_embeddings(rng)
    x = contexts(50, rng)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)


def test_pref_pairs_deterministic_and_labels_honest() -> None:
    rng = np.random.default_rng(1)
    emb = action_embeddings(np.random.default_rng(2))
    x, aw, al = pref_pairs(200, emb, rng, noise=0.0)
    r = true_reward(x, emb)
    # with zero label noise every winner must beat its loser
    assert np.all(r[np.arange(200), aw] >= r[np.arange(200), al])
    rng2 = np.random.default_rng(1)
    x2, aw2, al2 = pref_pairs(200, emb, rng2, noise=0.0)
    assert np.array_equal(aw, aw2) and np.array_equal(al, al2)


def test_pref_pairs_rejects_bad_noise() -> None:
    rng = np.random.default_rng(0)
    emb = action_embeddings(rng)
    with pytest.raises(ValueError, match="noise"):
        pref_pairs(10, emb, rng, noise=-0.5)
    with pytest.raises(ValueError, match="noise"):
        pref_pairs(10, emb, rng, noise=1.5)


def test_best_action_rate_bounds() -> None:
    rng = np.random.default_rng(0)
    emb = action_embeddings(rng)
    x = contexts(100, rng)
    logits = np.zeros((100, N_ACT))
    rate = best_action_rate(logits, x, emb)
    assert 0.0 <= rate <= 1.0
