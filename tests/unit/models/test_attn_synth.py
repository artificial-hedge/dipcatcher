"""Unit tests for quant_fund.models._attn_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._attn_synth import attn_dot_cost, synth_retrieval


def test_synth_retrieval_deterministic_and_labeled() -> None:
    rng = np.random.default_rng(0)
    x1, y1 = synth_retrieval(8, 5, 4, rng)
    x2, y2 = synth_retrieval(8, 5, 4, np.random.default_rng(0))
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert x1.shape == (8, 11, 64 + 4 + 8)
    assert set(np.unique(y1)) <= set(range(4))


def test_full_attention_cost_is_one() -> None:
    assert attn_dot_cost(128, "full", 0) == 1.0


def test_blocked_cost_counts_tail_tokens() -> None:
    # 10 tokens in 3 sinkhorn blocks: the 10th token cannot be dropped,
    # so the honest cost uses ceil blocks of size 4.
    c = attn_dot_cost(10, "sinkhorn", 3)
    assert c == pytest.approx(3 * 4 * 4 * 2 / 100.0)
    c2 = attn_dot_cost(10, "lsh", 3)
    assert c2 == pytest.approx(3 * 4 * 4 / 100.0)


def test_dot_cost_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown attention method"):
        attn_dot_cost(64, "bogus", 4)


def test_dot_cost_rejects_bad_param() -> None:
    with pytest.raises(ValueError, match="param"):
        attn_dot_cost(64, "sinkhorn", 0)
    with pytest.raises(ValueError, match="param"):
        attn_dot_cost(64, "lsh", -2)
    with pytest.raises(ValueError, match="n_tokens"):
        attn_dot_cost(0, "linear", 4)


def test_linear_cost_scales_with_param() -> None:
    c4 = attn_dot_cost(128, "linformer", 4)
    c8 = attn_dot_cost(128, "linformer", 8)
    assert c8 == pytest.approx(2 * c4)
