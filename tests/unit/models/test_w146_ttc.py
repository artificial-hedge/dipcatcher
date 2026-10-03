"""Wave-146 test-time-compute + multi-agent canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._ttc_synth import eval_chain, op_features, synth_problems
from quant_fund.models.consistency_vote import bench_consistency_vote
from quant_fund.models.debate_multiagent import bench_debate_multiagent
from quant_fund.models.knowledge_graph_embed import bench_knowledge_graph_embed
from quant_fund.models.mcts_reason import bench_mcts_reason
from quant_fund.models.unlearn_ga import bench_unlearn_ga
from quant_fund.models.verifier_prm import bench_verifier_prm


class TestFixture:
    def test_problems(self) -> None:
        a, ops, y = synth_problems(20, np.random.default_rng(0))
        assert a.shape == (20, 3) and ops.shape == (20, 2) and y.shape == (20,)

    def test_eval_chain(self) -> None:
        a = np.array([[2, 3, 4]])
        ops = np.array([[0, 1]])  # (2+3)-4 = 1
        assert eval_chain(a, ops)[0] == 1.0

    def test_features(self) -> None:
        a = np.array([[2, 3, 4]])
        assert op_features(a, 0).shape == (1, 10)


class TestConsistency:
    def test_bench(self) -> None:
        out = bench_consistency_vote(seed=3, n_train=10, n_test=40, k=4, iters=40)
        assert 0 <= out["synthetic_sc_acc"] <= 1


class TestPRM:
    def test_bench(self) -> None:
        out = bench_verifier_prm(seed=5, n_train=10, n_test=40, k=3, iters=40)
        assert 0 <= out["synthetic_prm_acc"] <= 1


class TestMCTS:
    def test_bench(self) -> None:
        out = bench_mcts_reason(seed=7, n_train=10, n_test=20, iters=40)
        assert 0 <= out["synthetic_mcts_acc"] <= 1


class TestDebate:
    def test_bench(self) -> None:
        out = bench_debate_multiagent(seed=9, n_train=30, n_test=40, iters=40)
        assert 0 <= out["synthetic_debate_acc"] <= 1


class TestUnlearn:
    def test_bench(self) -> None:
        out = bench_unlearn_ga(seed=11, n_forget=20, n_retain=40, iters=40, unl_iters=10)
        assert 0 <= out["synthetic_unlearn_retain_post"] <= 1


class TestKGE:
    def test_bench(self) -> None:
        out = bench_knowledge_graph_embed(seed=13, n_triples=80, iters=40)
        assert 0 <= out["synthetic_kge_hits10"] <= 1
