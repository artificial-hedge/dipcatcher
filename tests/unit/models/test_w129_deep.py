"""Wave-129 exec-summary DL SOTA-4 module tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.crossformer import bench_crossformer, synth_patch_coupled
from quant_fund.models.ft_transformer import bench_ft_transformer, synth_interaction
from quant_fund.models.itransformer import bench_itransformer, synth_variate_panel
from quant_fund.models.mambats import bench_mambats, synth_selective_memory
from quant_fund.models.nbeats_deep import bench_nbeats_deep, synth_multicomponent
from quant_fund.models.tcn_forecaster import bench_tcn_forecaster, synth_long_lag


class TestITransformer:
    def test_panel(self) -> None:
        x, y = synth_variate_panel(40, 24, np.random.default_rng(0))
        assert x.shape == (40, 24, 4)

    def test_bench(self) -> None:
        out = bench_itransformer(seed=3, n=120, iters=200)
        assert out["synthetic_itransformer_margin_vs_perchan"] > -1


class TestTcn:
    def test_signal(self) -> None:
        x, y = synth_long_lag(40, 48, np.random.default_rng(0))
        assert x.shape == (40, 48)

    def test_bench(self) -> None:
        out = bench_tcn_forecaster(seed=5, n=120, iters=200)
        assert out["synthetic_tcn_margin_vs_ar"] > -1


class TestFtTransformer:
    def test_table(self) -> None:
        x, y = synth_interaction(40, 10, np.random.default_rng(0))
        assert x.shape == (40, 10)

    def test_bench(self) -> None:
        out = bench_ft_transformer(seed=7, n=120, iters=200)
        assert out["synthetic_ft_margin_vs_logistic"] > -1


class TestNbeats:
    def test_series(self) -> None:
        x, y = synth_multicomponent(40, 40, 8, np.random.default_rng(0))
        assert x.shape == (40, 40) and y.shape == (40, 8)

    def test_bench(self) -> None:
        out = bench_nbeats_deep(seed=11, n=120, iters=200)
        assert out["synthetic_nbeats_margin_vs_naive"] > 0


class TestMambats:
    def test_memory(self) -> None:
        x, y = synth_selective_memory(40, 32, np.random.default_rng(0))
        assert x.shape == (40, 32, 2)

    def test_bench(self) -> None:
        out = bench_mambats(seed=13, n=120, iters=200)
        assert np.isfinite(out["synthetic_mamba_mae"])


class TestCrossformer:
    def test_panel(self) -> None:
        x, y = synth_patch_coupled(40, 48, 4, np.random.default_rng(0))
        assert x.shape == (40, 48, 4)

    def test_bench(self) -> None:
        out = bench_crossformer(seed=17, n=120, iters=200)
        assert np.isfinite(out["synthetic_crossformer_mae"])
