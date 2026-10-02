"""Wave-126 exec-summary deep-learning module tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.lob_transformer import bench_lob_transformer, synth_lob
from quant_fund.models.neural_ode import bench_neural_ode, synth_process
from quant_fund.models.patchtst import _patchify, bench_patchtst
from quant_fund.models.set_transformer import bench_set_transformer, synth_order_sets
from quant_fund.models.tft_forecaster import bench_tft_forecaster, synth_series
from quant_fund.models.world_model import bench_world_model, collect, exec_sim_step


class TestTft:
    def test_series_shapes(self) -> None:
        x, y = synth_series(120, np.random.default_rng(0))
        assert x.shape == (120, 3) and y.shape == (120,)

    @pytest.mark.slow
    def test_bench(self) -> None:
        out = bench_tft_forecaster(seed=3)
        assert out["synthetic_tft_margin_vs_ridge"] > 0
        assert out["torch_available"] == 1.0


class TestPatchTst:
    def test_patchify(self) -> None:
        p = _patchify(np.arange(40.0), 16, 8)
        assert p.shape == (4, 16)

    def test_bench(self) -> None:
        out = bench_patchtst(seed=5)
        assert np.isfinite(out["synthetic_patchtst_mae"])
        assert out["torch_available"] == 1.0


class TestLobTransformer:
    def test_lob_shapes(self) -> None:
        x, y = synth_lob(50, 40, np.random.default_rng(0))
        assert x.shape == (50, 40, 4) and set(np.unique(y)) <= {0.0, 1.0}

    def test_bench(self) -> None:
        out = bench_lob_transformer(seed=7)
        assert out["synthetic_lobdl_margin"] > 0


class TestSetTransformer:
    def test_sets(self) -> None:
        xs, ys = synth_order_sets(30, np.random.default_rng(0))
        assert len(xs) == 30 and xs[0].shape[1] == 3

    def test_bench(self) -> None:
        out = bench_set_transformer(seed=11)
        assert out["synthetic_setformer_r2_margin"] > 0
        assert out["synthetic_setformer_perm_invariance_err"] < 1e-3


class TestNeuralOde:
    def test_process(self) -> None:
        y = synth_process(100, np.random.default_rng(0))
        assert y.shape == (100,)

    def test_bench(self) -> None:
        out = bench_neural_ode(seed=13)
        assert np.isfinite(out["synthetic_node_1step_mae"])


class TestWorldModel:
    def test_sim(self) -> None:
        rng = np.random.default_rng(0)
        s = np.array([1.0, 0.0, 0.0])
        s2, r = exec_sim_step(s, 0.5, rng)
        assert s2.shape == (3,) and np.isfinite(r)

    def test_bench(self) -> None:
        eps = collect(5, np.random.default_rng(0))
        assert len(eps) == 5 and eps[0]["s"].shape == (10, 3)
        out = bench_world_model(seed=17)
        assert np.isfinite(out["synthetic_wm_margin_vs_twap"])
