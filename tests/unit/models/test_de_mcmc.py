"""Unit tests for quant_fund.models.de_mcmc."""

from __future__ import annotations

import pytest


def test_de_gamma_scale() -> None:
    """ter Braak (2006): gamma = 2.38/sqrt(2d). For d=2 that is
    1.19 — the code had 2.38/sqrt(2)=1.68 baked in, over-sized
    jumps that depress acceptance."""
    from quant_fund.models.de_mcmc import _de_gamma

    assert _de_gamma(2) == pytest.approx(2.38 / 2.0)
    assert _de_gamma(5) == pytest.approx(2.38 / (2 * 5) ** 0.5)
