"""Wave-148 interpretability canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._interp_synth import feature_directions, synth_activations
from quant_fund.models.activation_steering import bench_activation_steering
from quant_fund.models.circuit_ablation import bench_circuit_ablation
from quant_fund.models.logit_lens import bench_logit_lens
from quant_fund.models.patch_activation import bench_patch_activation
from quant_fund.models.probe_linear import bench_probe_linear
from quant_fund.models.sae_feature import bench_sae_feature


class TestFixture:
    def test_dirs(self) -> None:
        d = feature_directions(np.random.default_rng(0))
        assert np.allclose(np.linalg.norm(d, axis=-1), 1.0)

    def test_acts(self) -> None:
        d = feature_directions(np.random.default_rng(0))
        a, s = synth_activations(20, d, np.random.default_rng(1))
        assert a.shape == (20, 16) and s.shape == (20, 8)


class TestSAE:
    def test_bench(self) -> None:
        out = bench_sae_feature(seed=3, n=300, hidden=16, iters=40)
        assert 0 <= out["synthetic_sae_match"] <= 1


class TestSteer:
    def test_bench(self) -> None:
        out = bench_activation_steering(seed=5, n=300)
        assert 0 <= out["synthetic_steer_flip_rate"] <= 1


class TestProbe:
    def test_bench(self) -> None:
        out = bench_probe_linear(seed=7, n=300)
        assert out["synthetic_probe_acc"] > out["synthetic_probe_null_acc"]


class TestLens:
    def test_bench(self) -> None:
        out = bench_logit_lens(seed=9, n=300)
        assert 0 <= out["synthetic_lens_late_acc"] <= 1


class TestPatch:
    def test_bench(self) -> None:
        out = bench_patch_activation(seed=11, n=300)
        assert 0 <= out["synthetic_patch_flip"] <= 1


class TestAbl:
    def test_bench(self) -> None:
        out = bench_circuit_ablation(seed=13, n=300)
        assert out["synthetic_abl_target_drop"] >= 0
