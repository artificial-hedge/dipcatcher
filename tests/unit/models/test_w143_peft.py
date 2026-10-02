"""Wave-143 PEFT canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._peft_synth import synth_peft_base, synth_peft_shift
from quant_fund.models.dora_weight import bench_dora_weight
from quant_fund.models.lora_ft import bench_lora_ft
from quant_fund.models.prefix_tuning import bench_prefix_tuning
from quant_fund.models.prompt_tuning import bench_prompt_tuning
from quant_fund.models.qlora_nf4 import _nf4_levels, bench_qlora_nf4
from quant_fund.models.task_vector_merge import bench_task_vector_merge


class TestFixture:
    def test_base(self) -> None:
        x, y = synth_peft_base(30, np.random.default_rng(0))
        assert x.shape == (30, 4)

    def test_shift(self) -> None:
        x, y = synth_peft_shift(30, np.random.default_rng(0))
        assert y.max() <= 1

    def test_nf4(self) -> None:
        lv = _nf4_levels()
        assert len(lv) == 16 and abs(lv.min() + 1) < 1e-9


class TestLoRA:
    def test_bench(self) -> None:
        out = bench_lora_ft(seed=3, n_train=80, n_shift=40, iters_base=40, iters_adapt=40)
        assert 0 <= out["synthetic_lora_acc_shift"] <= 1


class TestQLoRA:
    def test_bench(self) -> None:
        out = bench_qlora_nf4(seed=5, n_train=80, n_shift=40, iters_base=40, iters_adapt=40)
        assert out["synthetic_qlora_bits_ratio"] < 1


class TestDoRA:
    def test_bench(self) -> None:
        out = bench_dora_weight(seed=7, n_train=80, n_shift=40, iters_base=40, iters_adapt=40)
        assert 0 <= out["synthetic_dora_acc_shift"] <= 1


class TestPrompt:
    def test_bench(self) -> None:
        out = bench_prompt_tuning(
            seed=9, n_train=120, n_shift=60, m_pairs=8, iters_base=40, iters_adapt=40, n_prompt=4
        )
        assert 0 <= out["synthetic_prompt_acc_shift"] <= 1


class TestPrefix:
    def test_bench(self) -> None:
        out = bench_prefix_tuning(
            seed=11, n_train=120, n_shift=60, m_pairs=8, iters_base=40, iters_adapt=40, n_prefix=4
        )
        assert 0 <= out["synthetic_prefix_acc_shift"] <= 1


class TestMerge:
    def test_bench(self) -> None:
        out = bench_task_vector_merge(
            seed=13, n_train=80, n_shift=40, iters_base=40, iters_adapt=40
        )
        assert 0 <= out["synthetic_merge_mean_acc"] <= 1
