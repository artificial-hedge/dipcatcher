"""Wave-186 energy-based-model canon tests."""

from __future__ import annotations

from quant_fund.models.adversarial_ebm import bench_adversarial_ebm
from quant_fund.models.contrastive_divergence import bench_contrastive_divergence
from quant_fund.models.denoising_sm import bench_denoising_sm
from quant_fund.models.noise_contrastive import bench_noise_contrastive
from quant_fund.models.persistent_cd import bench_persistent_cd
from quant_fund.models.score_matching import bench_score_matching


class TestSM:
    def test_bench(self) -> None:
        out = bench_score_matching(seed=3, iters=40)
        assert out["synthetic_sm_mmd"] >= 0.0


class TestDSM:
    def test_bench(self) -> None:
        out = bench_denoising_sm(seed=5, iters=40)
        assert out["synthetic_dsm_mmd"] >= 0.0


class TestNCE:
    def test_bench(self) -> None:
        out = bench_noise_contrastive(seed=7, iters=40)
        assert out["synthetic_nce_mmd"] >= 0.0


class TestCD:
    def test_bench(self) -> None:
        out = bench_contrastive_divergence(seed=9, iters=30, k=5)
        assert out["synthetic_cd_mmd"] >= 0.0


class TestPCD:
    def test_bench(self) -> None:
        out = bench_persistent_cd(seed=11, iters=30, k=4)
        assert out["synthetic_pcd_mmd"] >= 0.0


class TestAEBM:
    def test_bench(self) -> None:
        out = bench_adversarial_ebm(seed=13, iters=40)
        assert out["synthetic_aebm_mmd"] >= 0.0


class TestLangevinRNG:
    def test_preserves_global_rng(self) -> None:
        import torch

        from quant_fund.models._eb_synth import langevin, make_energy

        net = make_energy(torch)
        torch.manual_seed(123)
        before = torch.rand(3)
        torch.manual_seed(123)
        langevin(torch, net, 8, steps=2, seed=7)
        assert torch.equal(before, torch.rand(3))
