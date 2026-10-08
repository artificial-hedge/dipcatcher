"""Unit tests for quant_fund.models.data_cartography."""

from __future__ import annotations

import numpy as np
import pytest


def test_region_accuracy_measures_real_verdicts() -> None:
    """The ambiguous-region accuracy key was a constant-0.5 stub
    (np.where(cond, 0.5, 0.5)); it must reflect the model's actual
    per-region verdicts."""
    from quant_fund.models.data_cartography import region_accuracy

    prob = np.array([0.9, 0.1, 0.8, 0.7])
    y = np.array([1, 0, 0, 1])
    mask = np.array([True, True, True, True])
    assert region_accuracy(prob, y, mask) == pytest.approx(0.75)
    # restricted to a sub-region
    mask2 = np.array([False, False, True, True])
    assert region_accuracy(prob, y, mask2) == pytest.approx(0.5)
    # empty region is an honest zero, not a fabricated 0.5
    assert region_accuracy(prob, y, np.zeros(4, dtype=bool)) == 0.0
