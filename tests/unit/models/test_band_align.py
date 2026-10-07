"""Tests for models/band_align.py — unreachable targets must fail closed."""

from __future__ import annotations

import pytest


def test_unreachable_target_raises() -> None:
    """With |m-n| > w the banded DP can never reach (m,n); the old code
    returned the -1e9 sentinel as a score. It must raise instead."""
    from quant_fund.models.band_align import _nw_band

    with pytest.raises(ValueError):
        _nw_band("AAAA", "AAAAAA", 1)


def test_reachable_target_scores() -> None:
    from quant_fund.models.band_align import _nw, _nw_band

    assert _nw_band("ACGTACGT", "ACGTTCGT", 3) == _nw("ACGTACGT", "ACGTTCGT")
