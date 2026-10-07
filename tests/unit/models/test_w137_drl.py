"""Wave-137 distributional-RL canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._drl_synth import cvar, mc_return_dist, sample_return, synth_reward_env
from quant_fund.models.bootstrapped_dqn import bench_bootstrapped_dqn
from quant_fund.models.c51_dqn import bench_c51_dqn
from quant_fund.models.iqn_dqn import bench_iqn_dqn
from quant_fund.models.noisy_net import bench_noisy_net
from quant_fund.models.prioritized_replay import bench_prioritized_replay
from quant_fund.models.qr_dqn import bench_qr_dqn


class TestFixture:
    def test_synth(self) -> None:
        s, x = synth_reward_env(8, np.random.default_rng(0))
        assert s.shape == (8,) and x.shape == (8, 4)

    def test_returns(self) -> None:
        r = sample_return(np.zeros(5), np.zeros(5), np.random.default_rng(0))
        assert r.shape == (5,)
        assert np.isfinite(cvar(mc_return_dist(0, 0, np.random.default_rng(0), 200)))


class TestC51:
    def test_bench(self) -> None:
        out = bench_c51_dqn(seed=3, n_train=100, iters=40, mc_eval=200)
        assert 0 <= out["synthetic_c51_risk_share"] <= 1


class TestQR:
    def test_bench(self) -> None:
        out = bench_qr_dqn(seed=5, n_train=100, iters=40, mc_eval=200)
        assert np.isfinite(out["synthetic_qr_cvar"])


class TestIQN:
    def test_bench(self) -> None:
        out = bench_iqn_dqn(seed=7, n_train=100, iters=40, mc_eval=200)
        assert np.isfinite(out["synthetic_iqn_cvar"])


class TestNoisy:
    def test_bench(self) -> None:
        out = bench_noisy_net(seed=9, n_train=100, iters=40, cover_steps=20)
        assert 0 <= out["synthetic_noisy_coverage"] <= 1


class TestPER:
    def test_bench(self) -> None:
        out = bench_prioritized_replay(seed=11, n_train=60, iters=30, mc_eval=200)
        assert np.isfinite(out["synthetic_per_tail_mae"])


class TestBoot:
    def test_bench(self) -> None:
        out = bench_bootstrapped_dqn(seed=13, n_train=100, iters=40, mc_eval=200)
        assert np.isfinite(out["synthetic_boot_cvar"])


class TestReturnDist:
    def test_action_means_equal(self) -> None:
        rng = np.random.default_rng(3)
        m0 = mc_return_dist(0, 0, rng, 200000).mean()
        m1 = mc_return_dist(0, 1, rng, 200000).mean()
        # Docstring contract: action 1 shares the mean, adds left-tail mass.
        # Lomax(1.5) tail has infinite variance — tol 0.08 separates the fixed
        # base (≈0 diff) from the old 0.94 base (≈0.21 diff).
        assert abs(m0 - m1) < 0.08

    def test_action1_heavier_left_tail(self) -> None:
        rng = np.random.default_rng(5)
        c0 = cvar(mc_return_dist(0, 0, rng, 20000))
        c1 = cvar(mc_return_dist(0, 1, rng, 20000))
        assert c1 < c0
