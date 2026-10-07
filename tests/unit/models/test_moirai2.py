"""Adversarial probes for moirai2 (fail-closed lazy-dep adapter)."""

import numpy as np
import pytest

from quant_fund.models import moirai2 as mo


def test_tau_grid_validation():
    with pytest.raises(ValueError, match="taus"):
        mo.Moirai2Distribution(taus=[])
    with pytest.raises(ValueError, match="taus"):
        mo.Moirai2Distribution(taus=[0.9, 0.5])
    with pytest.raises(ValueError, match="taus"):
        mo.Moirai2Distribution(taus=[0.0, 0.5])
    with pytest.raises(ValueError, match="taus"):
        mo.Moirai2Distribution(taus=[0.5, np.nan])


def test_lookback_floor():
    with pytest.raises(ValueError, match="lookback"):
        mo.Moirai2Distribution(lookback=4)


def test_unfitted_predict_refuses():
    m = mo.Moirai2Distribution()
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict(np.zeros((3, 2)))
    with pytest.raises(RuntimeError, match="not been fitted"):
        m.predict_from_history(np.zeros(400))


def test_fit_fails_closed_without_dep():
    if mo.availability():
        pytest.skip("uni2ts installed in this env")
    m = mo.Moirai2Distribution()
    y = np.random.default_rng(0).standard_normal(600)
    with pytest.raises(RuntimeError, match="uni2ts is not installed"):
        m.fit(np.zeros((600, 1)), y)
    # half-fit object still cannot predict
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((2, 1)))


def test_fit_rejects_short_and_nonfinite():
    m = mo.Moirai2Distribution(lookback=64)
    with pytest.raises(ValueError, match="observations"):
        m.fit(np.zeros((40, 1)), np.zeros(40))
    y = np.full(200, 1.0)
    y[7] = np.inf
    with pytest.raises(ValueError, match="all-finite"):
        m.fit(np.zeros((200, 1)), y)


def test_quantile_row_fail_closed():
    m = mo.Moirai2Distribution(taus=(0.25, 0.5, 0.75))
    # wrong-length vector
    with pytest.raises(ValueError, match="quantiles"):
        mo._quantile_row(np.array([1.0, 2.0]), m.taus)
    # non-finite vector
    with pytest.raises(ValueError, match="quantiles"):
        mo._quantile_row(np.array([1.0, np.nan, 3.0]), m.taus)
    # on-grid vector accepted verbatim
    out = mo._quantile_row(np.array([0.5, 1.0, 2.0]), m.taus)
    np.testing.assert_allclose(out, [0.5, 1.0, 2.0])


def test_quantile_row_from_samples():
    m = mo.Moirai2Distribution(taus=(0.1, 0.5, 0.9))

    class _Pred:
        samples = np.linspace(-2.0, 2.0, 101)

    out = mo._quantile_row(_Pred(), m.taus)
    np.testing.assert_allclose(out, [-1.6, 0.0, 1.6], atol=1e-12)


def test_metadata_discloses_stub():
    m = mo.Moirai2Distribution(lookback=32)
    md = m.metadata()
    assert md.extra["framework"] == "uni2ts"
    assert md.extra["pretrained"] is True
    assert "not vendored" in md.extra["weights_note"]
    assert md.extra["lookback"] == 32
