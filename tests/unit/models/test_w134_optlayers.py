"""Wave-134 differentiable-optimization canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.cvxpy_layer import bench_cvxpy_layer
from quant_fund.models.deep_declarative import bench_deep_declarative, synth_expert_positions
from quant_fund.models.diff_mpc import bench_diff_mpc
from quant_fund.models.input_convex import bench_input_convex
from quant_fund.models.optnet_qp import bench_optnet_qp, synth_decision_data
from quant_fund.models.spd_net import bench_spd_net, synth_spd_data


class TestOptNet:
    def test_synth(self) -> None:
        x, mu, r = synth_decision_data(20, 12, np.random.default_rng(0))
        assert x.shape == (20, 12) and mu.shape == (20, 6)

    def test_bench(self) -> None:
        out = bench_optnet_qp(seed=3, n_train=300, n_test=60, iters=40)
        assert np.isfinite(out["synthetic_optnet_ret_e2e"])


class TestCvxLayer:
    def test_bench(self) -> None:
        out = bench_cvxpy_layer(seed=5, n_train=300, n_test=60, iters=40)
        assert np.isfinite(out["synthetic_cvx_ret_e2e"])


class TestICNN:
    def test_bench(self) -> None:
        out = bench_input_convex(seed=7, n_train=400, n_test=80, iters=60)
        assert 0 <= out["synthetic_icnn_jensen_viol"] <= 1


class TestDeclarative:
    def test_synth(self) -> None:
        x, y = synth_expert_positions(20, 10, np.random.default_rng(0))
        assert x.shape == (20, 10) and y.shape == (20,)

    def test_bench(self) -> None:
        out = bench_deep_declarative(seed=11, n_train=300, n_test=60, iters=40)
        assert out["synthetic_decl_mse"] >= 0


class TestSpdNet:
    def test_synth(self) -> None:
        c, y = synth_spd_data(20, np.random.default_rng(0))
        assert c.shape == (20, 6, 6) and y.shape == (20,)

    def test_bench(self) -> None:
        out = bench_spd_net(seed=13, n_train=120, n_test=40, iters=40)
        assert 0 <= out["synthetic_spd_acc"] <= 1


class TestDiffMPC:
    def test_bench(self) -> None:
        out = bench_diff_mpc(seed=17, n_train=8, n_test=4, iters=15, t_final=10)
        assert np.isfinite(out["synthetic_mpc_cl_err"])
