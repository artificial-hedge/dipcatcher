import numpy as np
import pytest

from quant_fund.decay.regime_ic import (
    regime_ic_difference,
    regime_ic_dispersion,
    regime_ic_summary,
)

pytestmark = pytest.mark.synthetic


def _regime_ic(seed: int, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    n = 800
    ic = 0.02 + 0.1 * rng.standard_normal(n)
    regime = (rng.random(n) < 0.5).astype(np.int64)
    ic[regime == 1] += shift
    return ic, regime


def test_regime_summary_counts_and_means() -> None:
    ic, regime = _regime_ic(70, shift=0.1)
    out = regime_ic_summary(ic, regime)
    assert set(out.keys()) == {"0", "1"}
    assert out["1"]["n"] > 200
    assert out["1"]["mean"] > out["0"]["mean"]


def test_regime_difference_detects_shift() -> None:
    ic, regime = _regime_ic(71, shift=0.12)
    out = regime_ic_difference(ic, regime, 1, 0, n_perm=500, seed=0)
    assert out["diff"] > 0.05
    assert out["p"] < 0.1


def test_regime_difference_accepts_no_shift() -> None:
    ic, regime = _regime_ic(72, shift=0.0)
    out = regime_ic_difference(ic, regime, 1, 0, n_perm=500, seed=0)
    assert out["p"] > 0.05


def test_dispersion_zero_without_regime_structure() -> None:
    ic, regime = _regime_ic(73, shift=0.0)
    d = regime_ic_dispersion(ic, regime)
    assert d >= 0.0
    assert d < 0.5


def test_validation() -> None:
    with pytest.raises(ValueError):
        regime_ic_summary(np.zeros(5), np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError):
        regime_ic_dispersion(np.zeros(10), np.zeros(10, dtype=np.int64))
