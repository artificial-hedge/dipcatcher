"""Adversarial probe: dml_irm must reject non-binary treatment that only
LOOKS binary after int truncation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.double_ml import dml_irm, synth_irm


def test_fractional_dose_rejected() -> None:
    di = synth_irm(n=200, seed=3)
    y, x = np.asarray(di["y"]), np.asarray(di["x"])
    # d in {0.3, 1.7}: astype(int) would see {0, 1} and pass the old gate.
    d = np.where(np.asarray(di["d"]) > 0.5, 1.7, 0.3)
    with pytest.raises(ValueError, match="binary"):
        dml_irm(y, d, x, n_folds=2, seed=3)


def test_dose_response_rejected() -> None:
    di = synth_irm(n=200, seed=4)
    y, x = np.asarray(di["y"]), np.asarray(di["x"])
    d = np.where(np.asarray(di["d"]) > 0.5, 1.0, 0.4)  # {0.4, 1.0} -> int {0, 1}
    with pytest.raises(ValueError, match="binary"):
        dml_irm(y, d, x, n_folds=2, seed=4)


def test_true_binary_still_accepted() -> None:
    di = synth_irm(n=200, seed=5)
    out = dml_irm(
        np.asarray(di["y"]),
        np.asarray(di["d"]),
        np.asarray(di["x"]),
        n_folds=2,
        learner="ridge",
        seed=5,
    )
    assert np.isfinite(out["ate"])
