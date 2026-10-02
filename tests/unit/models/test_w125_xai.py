"""Wave-125 exec-summary pricing/quantum/XAI module tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.adversarial_robust import Mlp, bench_adversarial_robust, fgsm, synth_signal
from quant_fund.models.causal_miner import bench_causal_miner, partial_corr_lag1, synth_zoo
from quant_fund.models.pinn_pricing import bench_pinn_pricing, bs_call
from quant_fund.models.qubo_portfolio import anneal, bench_qubo_portfolio, brute_force, build_qubo
from quant_fund.models.risk_flow import QuantileFlow, bench_risk_flow, emp_es, synth_factors
from quant_fund.models.xai_shap import (
    bench_xai_shap,
    integrated_gradients,
    kernel_shap,
    signal_model,
)


class TestPinn:
    def test_bs_shape(self) -> None:
        c = bs_call(np.array([100.0]), np.array([0.5]))
        assert c[0] > 0 and c[0] < 100

    def test_pinn_runs(self) -> None:
        out = bench_pinn_pricing(seed=5)
        assert out["synthetic_pinn_rel_mae"] < out["synthetic_pinn_untrained_mae"]


class TestQubo:
    def test_anneal_reaches_opt(self) -> None:
        rng = np.random.default_rng(0)
        mu = rng.uniform(-0.02, 0.05, 10)
        a = rng.standard_normal((10, 10))
        cov = a @ a.T / 10 + 0.01 * np.eye(10)
        q = build_qubo(mu, cov, 3)
        x, e = anneal(q, 2000, rng, 3)
        assert e <= brute_force(q, 3) + 1e-6
        assert x.sum() == 3

    def test_bench_keys(self) -> None:
        out = bench_qubo_portfolio(seed=2)
        assert out["synthetic_qubo_optimality_gap"] >= 0
        assert out["synthetic_qubo_cardinality_ok"] == 1.0


class TestXai:
    def test_shap_finite(self) -> None:
        rng = np.random.default_rng(0)
        bg = rng.standard_normal((100, 5))
        phi = kernel_shap(signal_model, rng.standard_normal(5), bg, 100, rng)
        assert phi.shape == (5,) and np.isfinite(phi).all()

    def test_bench_tau(self) -> None:
        out = bench_xai_shap(seed=8)
        assert out["synthetic_shap_rank_tau"] > 0.5
        ig = integrated_gradients(signal_model, np.ones(5), np.zeros(5), steps=32)
        assert ig.shape == (5,)


class TestAdvRobust:
    def test_fgsm_moves(self) -> None:
        rng = np.random.default_rng(0)
        x, y = synth_signal(50, rng)
        m = Mlp()
        m.fit(x, y, epochs=5)
        xa = fgsm(m, x, y, 0.1)
        assert not np.allclose(xa, x)

    def test_bench_margin(self) -> None:
        out = bench_adversarial_robust(seed=9)
        assert out["synthetic_adv_robustness_margin"] > 0.0
        assert out["synthetic_adv_plain_attacked_acc"] < out["synthetic_adv_plain_clean_acc"]


class TestRiskFlow:
    def test_flow_sample_shape(self) -> None:
        rng = np.random.default_rng(0)
        f = QuantileFlow()
        f.fit(synth_factors(500, rng))
        d = f.sample(100, rng)
        assert d.shape == (100, 2)

    def test_flow_beats_gauss(self) -> None:
        out = bench_risk_flow(seed=1)
        assert out["synthetic_flow_es_err"] < out["synthetic_flow_gauss_es_err"]
        assert emp_es(np.array([1.0, -1.0, -2.0, 0.5]), 0.75) > 0


class TestCausalMiner:
    def test_recovery(self) -> None:
        out = bench_causal_miner(seed=4)
        assert out["synthetic_pcmci_recall"] == 1.0
        assert out["synthetic_pcmci_precision"] > 0.5

    def test_partial_corr(self) -> None:
        rng = np.random.default_rng(0)
        f, y, _ = synth_zoo(400, rng)
        r = partial_corr_lag1(f, y, 0)
        assert -1 <= r <= 1
