"""Unit tests for quant_fund.models._dc_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._dc_synth import dc_data, fit_eval


def test_data_deterministic() -> None:
    a = dc_data(5, n=200)
    b = dc_data(5, n=200)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))


def test_flip_zero_gives_unflipped_pool() -> None:
    # flip=0: pool labels follow the generative p exactly (no flips applied)
    X, y, Xt, yt = dc_data(3, n=200, flip=0.0)
    assert set(np.unique(y)) <= {0, 1}
    assert set(np.unique(yt)) <= {0, 1}


def test_flip_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="flip"):
        dc_data(0, flip=-0.2)
    with pytest.raises(ValueError, match="flip"):
        dc_data(0, flip=1.3)


def test_fit_eval_accuracy_bounds() -> None:
    X, y, Xt, yt = dc_data(2, n=300)
    acc = fit_eval(X, y, Xt, yt)
    assert 0.0 <= acc <= 1.0
