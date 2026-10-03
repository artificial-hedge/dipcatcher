"""Wave-127 exec-summary DL SOTA-2 module tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.bnn_ensemble import bench_bnn_ensemble, synth_hetero
from quant_fund.models.contrastive_repr import bench_contrastive_repr, synth_regime_windows
from quant_fund.models.diff_policy import bench_diff_policy, exec_sim
from quant_fund.models.hypernetwork_alloc import bench_hypernetwork_alloc, synth_market
from quant_fund.models.neural_thompson import bench_neural_thompson, synth_bandit
from quant_fund.models.option_vae import bench_option_vae, synth_smiles


class TestContrastive:
    def test_windows(self) -> None:
        x, y, reg = synth_regime_windows(50, 32, np.random.default_rng(0))
        assert x.shape == (50, 32) and reg.shape == (50,)

    def test_bench(self) -> None:
        out = bench_contrastive_repr(seed=3)
        assert out["synthetic_contrastive_acc_margin"] > 0


class TestHypernet:
    def test_market(self) -> None:
        rets, state, z = synth_market(100, np.random.default_rng(0))
        assert rets.shape == (100, 3) and state.shape == (100, 2)
        assert set(np.unique(z)) <= {0, 1, 2}

    @pytest.mark.slow
    def test_bench(self) -> None:
        out = bench_hypernetwork_alloc(seed=5)
        assert out["synthetic_hypernet_margin_vs_ew"] > 0


class TestNeuralThompson:
    def test_bandit(self) -> None:
        ctx, rew, k = synth_bandit(50, np.random.default_rng(0))
        assert ctx.shape == (50, 4) and rew.shape == (50, k)

    def test_bench(self) -> None:
        out = bench_neural_thompson(seed=7)
        assert out["synthetic_nthompson_margin_vs_unif"] > 0


class TestBnnEnsemble:
    def test_synth(self) -> None:
        x, y, sig = synth_hetero(100, np.random.default_rng(0))
        assert x.shape == (100, 1) and np.all(sig > 0)

    def test_bench(self) -> None:
        out = bench_bnn_ensemble(seed=11)
        assert out["synthetic_bnn_epistemic_ood_ratio"] > 1.0
        assert out["synthetic_bnn_cover90"] > 0.5


class TestOptionVae:
    def test_smiles(self) -> None:
        iv, f = synth_smiles(50, np.random.default_rng(0))
        assert iv.shape == (50, 5) and f.shape == (50, 3)

    def test_bench(self) -> None:
        out = bench_option_vae(seed=13)
        assert out["synthetic_ovae_level_corr"] > 0.3


class TestDiffPolicy:
    def test_sim(self) -> None:
        c = exec_sim(np.full(10, 0.1), np.random.default_rng(0))
        assert np.isfinite(c)

    def test_bench(self) -> None:
        out = bench_diff_policy(seed=17)
        assert out["synthetic_diffpolicy_margin_vs_twap"] > 0
