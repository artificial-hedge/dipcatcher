"""Unit tests for quant_fund.models._compress_synth."""

from __future__ import annotations

import numpy as np

from quant_fund.models._compress_synth import split


def test_split_halves_and_deterministic() -> None:
    a = split(3, n=200)
    b = split(3, n=200)
    xtr, ytr, xte, yte = a
    assert xtr.shape[0] == ytr.shape[0] == 100
    assert xte.shape[0] == yte.shape[0] == 100
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))


def test_split_feature_dim_matches_mlp() -> None:
    xtr, _ytr, xte, _yte = split(0, n=100)
    # make_mlp is 8 -> 24 -> 2: the dataset must feed it 8 features
    assert xtr.shape[1] == xte.shape[1] == 8
