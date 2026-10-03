"""Wave-147 alignment canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._align_synth import (
    N_ACT,
    action_embeddings,
    contexts,
    pref_pairs,
    true_reward,
)
from quant_fund.models.dpo_train import bench_dpo_train
from quant_fund.models.grpo_train import bench_grpo_train
from quant_fund.models.ipo_train import bench_ipo_train
from quant_fund.models.kto_train import bench_kto_train
from quant_fund.models.reward_model import bench_reward_model
from quant_fund.models.rlhf_ppo import bench_rlhf_ppo


class TestFixture:
    def test_embeddings(self) -> None:
        e = action_embeddings(np.random.default_rng(0))
        assert e.shape == (N_ACT, 4)

    def test_pairs(self) -> None:
        emb = action_embeddings(np.random.default_rng(0))
        x, w, ll = pref_pairs(20, emb, np.random.default_rng(1), noise=0.0)
        r = true_reward(x, emb)
        assert (r[np.arange(20), w] >= r[np.arange(20), ll]).all()

    def test_contexts(self) -> None:
        x = contexts(10, np.random.default_rng(0))
        assert np.allclose(np.linalg.norm(x, axis=-1), 1.0)


class TestRM:
    def test_bench(self) -> None:
        out = bench_reward_model(seed=3, n_pairs=80, n_eval=40, iters=40)
        assert 0 <= out["synthetic_rm_pair_auc"] <= 1


class TestDPO:
    def test_bench(self) -> None:
        out = bench_dpo_train(seed=5, n_pairs=80, n_eval=40, iters=50)
        assert 0 <= out["synthetic_dpo_best_rate"] <= 1


class TestIPO:
    def test_bench(self) -> None:
        out = bench_ipo_train(seed=7, n_pairs=80, n_eval=40, iters=50)
        assert 0 <= out["synthetic_ipo_best_rate"] <= 1


class TestKTO:
    def test_bench(self) -> None:
        out = bench_kto_train(seed=9, n_lab=80, n_eval=40, iters=50)
        assert 0 <= out["synthetic_kto_best_rate"] <= 1


class TestGRPO:
    def test_bench(self) -> None:
        out = bench_grpo_train(seed=11, n_ctx=40, group=4, iters=30)
        assert np.isfinite(out["synthetic_grpo_reward"])


class TestRLHF:
    def test_bench(self) -> None:
        out = bench_rlhf_ppo(seed=13, n_pairs=60, n_ctx=40, iters_rm=40, iters_ppo=40)
        assert np.isfinite(out["synthetic_rlhf_overopt_gap"])
