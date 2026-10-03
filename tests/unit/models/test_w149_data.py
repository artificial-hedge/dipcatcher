"""Wave-149 data-dynamics canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._data_synth import synth_dataset, true_margin
from quant_fund.models.coreset_herding import bench_coreset_herding
from quant_fund.models.curriculum_magnitude import bench_curriculum_magnitude
from quant_fund.models.dataset_distillation import bench_dataset_distillation
from quant_fund.models.label_smoothing import bench_label_smoothing
from quant_fund.models.mixup_cutmix import bench_mixup_cutmix
from quant_fund.models.sharpness_sam import bench_sharpness_sam


class TestFixture:
    def test_dataset(self) -> None:
        x, y, h = synth_dataset(50, np.random.default_rng(0))
        assert x.shape == (50, 8) and h.shape == (50,)

    def test_margin(self) -> None:
        x, y, _h = synth_dataset(50, np.random.default_rng(0))
        m = true_margin(x, y)
        assert (m > 0).mean() > 0.5


class TestDD:
    def test_bench(self) -> None:
        out = bench_dataset_distillation(seed=3, n=120, k=8, iters=30, inner_steps=2)
        assert 0 <= out["synthetic_dd_acc"] <= 1


class TestHerd:
    def test_bench(self) -> None:
        out = bench_coreset_herding(seed=5, n=120, k=10)
        assert 0 <= out["synthetic_herd_acc"] <= 1


class TestCurr:
    def test_bench(self) -> None:
        out = bench_curriculum_magnitude(seed=7, n=160, stages=2)
        assert 0 <= out["synthetic_curr_acc"] <= 1


class TestLS:
    def test_bench(self) -> None:
        out = bench_label_smoothing(seed=9, n=160, iters=40)
        assert out["synthetic_ls_ece"] >= 0


class TestMixup:
    def test_bench(self) -> None:
        out = bench_mixup_cutmix(seed=11, n=160, iters=40)
        assert 0 <= out["synthetic_mixup_acc"] <= 1


class TestSAM:
    def test_bench(self) -> None:
        out = bench_sharpness_sam(seed=13, n=160, iters=40)
        assert 0 <= out["synthetic_sam_test_acc"] <= 1
