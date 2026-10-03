"""Tests for Dawid-Skene EM and GLAD label aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.dawid_skene import (
    bench_dawid_skene,
    dawid_skene,
    glad,
    majority_vote,
)


def _votes(seed: int = 0, n_items: int = 60, n_workers: int = 6):
    rng = np.random.default_rng(seed)
    k = 3
    true = rng.integers(0, k, size=n_items)
    votes = np.full((n_items, n_workers), -1)
    for w in range(n_workers):
        for j in range(n_items):
            if rng.random() < 0.7:
                votes[j, w] = true[j] if rng.random() < 0.8 else int(rng.integers(0, k))
    empty = (votes < 0).all(axis=1)
    votes[empty, 0] = true[empty]
    return true, votes, k


def test_majority_vote():
    true, votes, k = _votes()
    mv = majority_vote(votes, k)
    assert float((mv == true).mean()) > 0.7


def test_dawid_skene_shapes_and_posterior():
    true, votes, k = _votes()
    r = dawid_skene(votes, k, n_iter=20)
    assert r["posterior"].shape == (60, k)
    assert np.allclose(r["posterior"].sum(axis=1), 1.0, atol=1e-6)
    assert r["confusion"].shape == (6, k, k)
    # confusion rows are distributions
    assert np.allclose(r["confusion"].sum(axis=2), 1.0, atol=1e-4)
    assert float((r["labels"] == true).mean()) > 0.7


def test_dawid_skene_recovers_skilled_worker():
    rng = np.random.default_rng(3)
    n_items, k = 80, 2
    true = rng.integers(0, k, size=n_items)
    # one expert (95%) + five random guessers (35%)
    skills = np.array([0.95, 0.35, 0.35, 0.35, 0.35, 0.35])
    votes = np.zeros((n_items, 6), dtype=int)
    for w, s in enumerate(skills):
        for j in range(n_items):
            votes[j, w] = true[j] if rng.random() < s else int(rng.integers(0, k))
    r = dawid_skene(votes, k, n_iter=30)
    # expert row should have the largest diagonal mass
    diag = r["confusion"][:, 0, 0] + r["confusion"][:, 1, 1]
    assert int(diag.argmax()) == 0
    assert float((r["labels"] == true).mean()) > 0.8


def test_glad_binary():
    rng = np.random.default_rng(4)
    n_items = 60
    true = rng.integers(0, 2, size=n_items)
    votes = np.zeros((n_items, 5), dtype=int)
    for w in range(5):
        for j in range(n_items):
            votes[j, w] = true[j] if rng.random() < 0.8 else 1 - true[j]
    r = glad(votes, 2, n_iter=10)
    assert r["posterior"].shape == (n_items, 2)
    assert float((r["labels"] == true).mean()) > 0.8
    with pytest.raises(ValueError):
        glad(votes, 3)


def test_fail_closed():
    with pytest.raises(ValueError):
        majority_vote(np.zeros((2, 2), dtype=int), 0)
    with pytest.raises(ValueError):
        dawid_skene(np.array([[5, 5], [5, 5]]), 3)
    with pytest.raises(ValueError):
        dawid_skene(np.array([[-1, -1], [0, 1]]), 2)


def test_bench():
    res = bench_dawid_skene()
    assert res["synthetic_score"] == 1.0
    assert res["synthetic_ds_gain"] > 0.0
