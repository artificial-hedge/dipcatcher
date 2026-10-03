"""Tests for models/expert_aggregation.py — prediction with expert advice."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.expert_aggregation import (
    bench_expert_aggregation,
    expert_fit,
    fixed_share_weights,
    hedge_weights,
    regret_decomp,
    specialist_weights,
    synth_experts,
)


@pytest.fixture(scope="module")
def panel() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return synth_experts(240, 5, 4, seed=0)


@pytest.fixture(scope="module")
def losses(panel) -> np.ndarray:
    y, fc, _ = panel
    l_mat = (y[:, None] - fc) ** 2
    return l_mat / max(float(np.ptp(l_mat)), 1e-12)


class TestGuards:
    def test_bad_losses(self):
        with pytest.raises(ValueError):
            hedge_weights(np.ones(5))
        with pytest.raises(ValueError):
            hedge_weights(np.ones((10, 1)))
        with pytest.raises(ValueError):
            hedge_weights(np.full((10, 3), np.nan))
        with pytest.raises(ValueError):
            hedge_weights(-np.ones((10, 3)))
        with pytest.raises(ValueError):
            hedge_weights(np.ones((10, 3)), eta=0.0)
        with pytest.raises(ValueError):
            fixed_share_weights(np.ones((10, 3)), alpha=1.5)
        with pytest.raises(ValueError):
            specialist_weights(np.ones((10, 3)), np.ones((5, 3)))
        with pytest.raises(ValueError):
            synth_experts(5, 3, 2, 0)
        with pytest.raises(ValueError):
            expert_fit(np.ones((10, 3)), np.ones(5))
        with pytest.raises(ValueError):
            expert_fit(np.ones((10, 3)), np.ones(10), loss="mae")


class TestHedge:
    def test_weights_simplex(self, losses):
        out = hedge_weights(losses)
        w = out["weights"]
        assert w.shape == losses.shape
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
        assert (w >= 0).all()

    def test_regret_bound_respected(self, losses):
        out = hedge_weights(losses)
        assert out["cum_regret"][-1] <= out["bound"] + 1e-9

    def test_zero_loss_expert_dominates(self):
        l_mat = np.ones((50, 4))
        l_mat[:, 2] = 0.0
        out = hedge_weights(l_mat)
        assert out["weights"][-1, 2] > 0.9

    def test_eta_default_positive(self, losses):
        assert hedge_weights(losses)["eta"] > 0

    def test_uniform_losses_stay_uniform(self):
        out = hedge_weights(np.full((30, 3), 0.5))
        np.testing.assert_allclose(out["weights"][-1], np.full(3, 1 / 3), atol=1e-12)


class TestFixedShare:
    def test_weights_simplex(self, losses):
        out = fixed_share_weights(losses, alpha=0.02)
        np.testing.assert_allclose(out["weights"].sum(axis=1), 1.0, atol=1e-12)

    def test_alpha_floor(self, losses):
        out = fixed_share_weights(losses, alpha=0.1)
        # mixing guarantees every expert keeps >= alpha/K mass post-update
        assert (out["weights"][1:] >= 0.1 / losses.shape[1] - 1e-12).all()

    def test_tracks_switching(self, losses):
        # under regime switches fixed-share total loss beats static hedge
        h = hedge_weights(losses)
        fs = fixed_share_weights(losses, alpha=0.05)
        assert fs["algo_loss"].sum() < h["algo_loss"].sum()


class TestSpecialists:
    def test_only_awake_played(self, losses):
        mask = np.zeros_like(losses, dtype=bool)
        mask[:, :2] = True
        out = specialist_weights(losses, mask)
        assert (out["weights"][:, 2:] == 0).all()

    def test_all_asleep_row_fallback(self, losses):
        mask = np.ones_like(losses, dtype=bool)
        mask[5] = False
        out = specialist_weights(losses, mask)
        np.testing.assert_allclose(out["weights"][5].sum(), 1.0)


class TestExpertFit:
    def test_pinball_and_squared(self, panel):
        y, fc, _ = panel
        for loss in ("squared", "pinball"):
            out = expert_fit(fc, y, loss=loss, tau=0.8)
            assert np.isfinite(out["cum_regret"][-1])

    def test_regret_decomp(self, losses):
        out = hedge_weights(losses)
        dec = regret_decomp(losses, out["weights"])
        np.testing.assert_allclose(dec["cum_regret"], out["cum_regret"], atol=1e-12)
        assert dec["best_expert_id"].shape == (losses.shape[0],)


class TestBench:
    def test_keys_finite(self):
        blob = bench_expert_aggregation(seed=0)
        assert blob
        for k, v in blob.items():
            assert k.startswith("synthetic_")
            assert isinstance(v, float) and np.isfinite(v)

    def test_no_forbidden_tokens(self):
        forbidden = {"sharpe", "sortino", "calmar", "pnl", "nav"}
        for k in bench_expert_aggregation(seed=0):
            assert not forbidden.intersection(k.split("_"))

    def test_science(self):
        blob = bench_expert_aggregation(seed=0)
        assert blob["synthetic_hedge_regret_ratio"] < 0.5
        assert blob["synthetic_fixedshare_beats_static"] == 1.0
        assert blob["synthetic_switch_detect_lag"] < 60
        assert blob["synthetic_eta_bound_respected"] == 1.0
        assert blob["synthetic_determinism"] == 1.0
        assert blob["synthetic_bad_expert_mass"] < 0.5
