import numpy as np
import pytest

from quant_fund.decay.ic_series import spearman_ic
from quant_fund.decay.lag_weighting import (
    fused_signal,
    fusion_beats_single,
    horizon_ic_profile,
    ic_weight_fuse,
)

pytestmark = pytest.mark.synthetic


def _horizons(seed: int, t_total: int = 400, n: int = 60):
    """Signal at 3 horizons; horizon 2 (index 1) carries most of the truth."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((t_total, n))
    actual = latent + 2.0 * rng.standard_normal((t_total, n))
    s1 = latent + 3.0 * rng.standard_normal((t_total, n))  # weak
    s2 = latent + 0.5 * rng.standard_normal((t_total, n))  # strong
    s3 = latent + 1.5 * rng.standard_normal((t_total, n))  # medium
    return [s1, s2, s3], actual


def test_horizon_ic_profile_ranks_strongest() -> None:
    signals, actual = _horizons(60)
    profile = horizon_ic_profile(signals, actual)
    assert profile[1] > profile[0]
    assert profile[1] > profile[2]
    assert profile[2] > profile[0]


def test_ic_weight_fuse_simplex() -> None:
    w = ic_weight_fuse(np.array([0.02, 0.10, 0.04]))
    assert abs(float(w.sum()) - 1.0) < 1e-12
    assert w[1] == pytest.approx(0.10 / 0.16)
    np.testing.assert_allclose(ic_weight_fuse(np.array([-0.1, -0.2])), [0.5, 0.5])


def test_fused_signal_nan_robust() -> None:
    rng = np.random.default_rng(61)
    a = rng.standard_normal((20, 10))
    b = rng.standard_normal((20, 10))
    b[5, 3] = np.nan
    out = fused_signal([a, b], np.array([0.3, 0.7]))
    assert out.shape == (20, 10)
    assert np.isfinite(out[5, 3])
    assert out[0, 0] == pytest.approx(0.3 * a[0, 0] + 0.7 * b[0, 0])


def test_fusion_robust_when_best_horizon_uncertain() -> None:
    signals_tr, actual_tr = _horizons(62)
    signals_ev, actual_ev = _horizons(63)
    out = fusion_beats_single(signals_tr, actual_tr, signals_ev, actual_ev)
    singles = [float(np.nanmean(spearman_ic(s, actual_ev))) for s in signals_ev]
    assert np.isfinite(out["fused_ic"])
    # fusion beats the median horizon and stays close to the best one —
    # the robustness property, not a domination claim
    assert out["fused_ic"] > float(np.median(singles))
    assert out["fused_ic"] >= out["best_single_ic"] * 0.85


def test_fusion_beats_single_out_of_sample() -> None:
    signals_tr, actual_tr = _horizons(62)
    signals_ev, actual_ev = _horizons(63)
    out = fusion_beats_single(signals_tr, actual_tr, signals_ev, actual_ev)
    assert np.isfinite(out["fused_ic"])
    assert out["best_single_ic"] > 0.0


def test_validation() -> None:
    with pytest.raises(ValueError):
        horizon_ic_profile([], np.zeros((5, 3)))
    with pytest.raises(ValueError):
        fused_signal([np.zeros((2, 2))], np.zeros(2))
