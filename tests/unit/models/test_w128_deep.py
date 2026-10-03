"""Wave-128 exec-summary DL SOTA-3 module tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.cnn_alpha import bench_cnn_alpha, synth_charts
from quant_fund.models.graph_temporal import bench_graph_temporal, synth_graph_panel
from quant_fund.models.informer_attn import bench_informer_attn, synth_sparse_signal
from quant_fund.models.kan_forecaster import bench_kan_forecaster, synth_nonlinear
from quant_fund.models.mask_autoencoder import bench_mask_autoencoder, synth_mae_windows
from quant_fund.models.ts_mixer import bench_ts_mixer, synth_mixer_panel


class TestKan:
    def test_synth(self) -> None:
        xs, y = synth_nonlinear(50, 8, np.random.default_rng(0))
        assert xs.shape == (50, 8)

    def test_bench(self) -> None:
        out = bench_kan_forecaster(seed=3)
        assert out["synthetic_kan_margin_vs_mlp"] > 0


class TestTsMixer:
    def test_panel(self) -> None:
        x = synth_mixer_panel(100, np.random.default_rng(0))
        assert x.shape == (100, 4)

    def test_bench(self) -> None:
        out = bench_ts_mixer(seed=5)
        assert out["synthetic_tsmixer_margin_vs_ar"] > 0


class TestInformer:
    def test_signal(self) -> None:
        xs, y = synth_sparse_signal(50, 128, np.random.default_rng(0))
        assert xs.shape == (50, 128)

    def test_bench(self) -> None:
        out = bench_informer_attn(seed=7)
        assert np.isfinite(out["synthetic_informer_mae"])
        assert out["synthetic_informer_query_share"] < 1.0


class TestCnnAlpha:
    def test_charts(self) -> None:
        x, y = synth_charts(30, 24, np.random.default_rng(0))
        assert x.shape == (30, 24, 4)

    def test_bench(self) -> None:
        out = bench_cnn_alpha(seed=11)
        assert out["synthetic_cnnalpha_margin"] > 0


class TestMaskAutoencoder:
    def test_windows(self) -> None:
        x, reg = synth_mae_windows(50, 40, np.random.default_rng(0))
        assert x.shape == (50, 40) and reg.shape == (50,)

    def test_bench(self) -> None:
        out = bench_mask_autoencoder(seed=13)
        assert out["synthetic_mae_margin_vs_raw"] > 0


class TestGraphTemporal:
    def test_panel(self) -> None:
        x, adj = synth_graph_panel(50, np.random.default_rng(0))
        assert x.shape == (50, 6) and adj.shape == (6, 6)

    def test_bench(self) -> None:
        out = bench_graph_temporal(seed=17)
        assert np.isfinite(out["synthetic_graphtemporal_mae"])
