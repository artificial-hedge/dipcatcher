"""Tests for models/banzhaf.py — must sum marginal contributions."""

from __future__ import annotations

import numpy as np


def test_additive_game_proportional() -> None:
    """On a non-0/1 (additive) game each player's marginal contribution
    is its weight — the Banzhaf value is proportional to weight. The old
    code counted nonzero swings and returned a uniform index."""
    from quant_fund.models.banzhaf import banzhaf_index
    from quant_fund.models.nucleolus import CoopGame

    w = np.array([1.0, 2.0, 3.0])
    vals = np.array([sum(w[i] for i in range(3) if s & (1 << i)) for s in range(8)])
    g = CoopGame(3, vals)
    idx = banzhaf_index(g)
    assert np.allclose(idx, w / w.sum())


def test_voting_game_unchanged() -> None:
    from quant_fund.models.banzhaf import banzhaf_index
    from quant_fund.models.nucleolus import voting_game

    idx = banzhaf_index(voting_game(np.ones(3), 2.0))
    assert np.allclose(idx, 1.0 / 3)
