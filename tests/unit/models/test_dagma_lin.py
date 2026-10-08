"""Unit tests for quant_fund.models.dagma_lin."""

from __future__ import annotations

import numpy as np
import pytest


def test_h_logdet_rejects_indefinite_barrier() -> None:
    """sI - W∘W not positive definite: h is a log-barrier and must
    fail, not return a finite-but-meaningless (possibly negative)
    value from log|det|."""
    from quant_fund.models.dagma_lin import _h_logdet

    w = np.full((4, 4), 3.0)  # I - W∘W deeply indefinite
    with pytest.raises(np.linalg.LinAlgError):
        _h_logdet(w)


def test_h_logdet_valid_pd() -> None:
    from quant_fund.models.dagma_lin import _h_logdet

    h, g = _h_logdet(np.zeros((3, 3)))
    assert h == pytest.approx(0.0)
    assert np.isfinite(g).all()
