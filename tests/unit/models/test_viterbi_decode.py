"""Tests for viterbi_decode."""

from __future__ import annotations

import numpy as np

from quant_fund.models.viterbi_decode import bench_viterbi_decode


def test_viterbi_clean_decode():
    from quant_fund.models.viterbi_decode import conv_encode, viterbi_hard

    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, 60).astype(np.float64)
    assert np.all(viterbi_hard(conv_encode(bits)) == bits)


def test_viterbi_corrects_single_error():
    from quant_fund.models.viterbi_decode import conv_encode, viterbi_hard

    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, 60).astype(np.float64)
    enc = conv_encode(bits)
    enc[14] = 1.0 - enc[14]
    assert np.all(viterbi_hard(enc) == bits)


def test_viterbi_soft_beats_hard():
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, 150).astype(np.float64)
    from quant_fund.models.viterbi_decode import (
        conv_encode,
        viterbi_hard,
        viterbi_soft,
    )

    sym = 2 * conv_encode(bits) - 1
    noisy = sym + rng.normal(0, 1.0, len(sym))
    hb = np.mean(viterbi_hard((noisy > 0).astype(float)) != bits)
    sb = np.mean(viterbi_soft(noisy) != bits)
    assert sb <= hb


def test_bench_viterbi():
    out = bench_viterbi_decode(seed=20261231)
    assert out["synthetic_viterbi_clean_ber"] == 0.0
    assert out["synthetic_viterbi_soft_ber"] <= out["synthetic_viterbi_hard_ber"]
    assert all(np.isfinite(v) for v in out.values())
