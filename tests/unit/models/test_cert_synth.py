"""Unit tests for quant_fund.models._cert_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._cert_synth import synth_cls_2d, synth_subset


def test_cls_margin_kwarg_is_live() -> None:
    # a bigger margin must drop strictly more boundary samples
    x_big, y_big = synth_cls_2d(400, np.random.default_rng(0), margin=1.0)
    x_def, y_def = synth_cls_2d(400, np.random.default_rng(0))
    assert len(x_big) < len(x_def)
    # everything kept respects the margin
    assert np.all(np.abs(x_big[:, 0] + 0.5 * x_big[:, 1]) > 1.0)
    assert np.all(np.abs(x_def[:, 0] + 0.5 * x_def[:, 1]) > 0.15)


def test_cls_rejects_negative_margin() -> None:
    with pytest.raises(ValueError, match="margin"):
        synth_cls_2d(100, np.random.default_rng(0), margin=-0.2)


def test_cls_deterministic_and_labels_correct() -> None:
    x1, y1 = synth_cls_2d(200, np.random.default_rng(1))
    x2, y2 = synth_cls_2d(200, np.random.default_rng(1))
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    assert np.array_equal(y1, (x1[:, 0] + 0.5 * x1[:, 1] > 0).astype(np.int64))


def test_subset_mask_and_determinism() -> None:
    x1, y1, m1 = synth_subset(100, 6, 2, np.random.default_rng(0))
    x2, y2, m2 = synth_subset(100, 6, 2, np.random.default_rng(0))
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2) and np.array_equal(m1, m2)
    assert m1.tolist() == [True, True, False, False, False, False]
