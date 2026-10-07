"""Wave-154 vision/multimodal canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._vision_synth import patches, synth_images
from quant_fund.models.attention_rollout import bench_attention_rollout
from quant_fund.models.clip_align import bench_clip_align
from quant_fund.models.convnet_baseline import bench_convnet_baseline
from quant_fund.models.diffusion_ddim import bench_diffusion_ddim
from quant_fund.models.simclr_views import bench_simclr_views
from quant_fund.models.vit_classifier import bench_vit_classifier


class TestFixture:
    def test_images(self) -> None:
        x, y = synth_images(0, 40)
        assert x.shape == (40, 6, 6) and set(np.unique(y)) == {0, 1}

    def test_patches(self) -> None:
        x, _y = synth_images(0, 10)
        assert patches(x).shape == (10, 9, 4)


class TestCNN:
    def test_bench(self) -> None:
        out = bench_convnet_baseline(seed=3, n=80, iters=15)
        assert 0 <= out["synthetic_cnn_acc"] <= 1


class TestViT:
    def test_bench(self) -> None:
        out = bench_vit_classifier(seed=5, n=80, iters=15)
        assert 0 <= out["synthetic_vit_acc"] <= 1


class TestCLIP:
    def test_bench(self) -> None:
        out = bench_clip_align(seed=7, n=80, iters=20)
        assert 0 <= out["synthetic_clip_acc"] <= 1


class TestSimCLR:
    def test_bench(self) -> None:
        out = bench_simclr_views(seed=9, n=80, iters=20)
        assert 0 <= out["synthetic_simclr_probe"] <= 1


class TestDDIM:
    def test_bench(self) -> None:
        out = bench_diffusion_ddim(seed=11, n=100, steps=5)
        assert out["synthetic_ddim_w1"] >= 0


class TestRollout:
    def test_bench(self) -> None:
        out = bench_attention_rollout(seed=13)
        assert out["synthetic_rollout_relevance"] > 0


class TestPatches:
    def test_other_patch_sizes(self) -> None:
        x = np.ones((2, 6, 6))
        out3 = patches(x, p=3)
        assert out3.shape == (2, 4, 9)
        assert (out3 != 0).all()  # no silent zero-padding

    def test_indivisible_patch_size(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            patches(np.zeros((2, 6, 6)), p=5)
