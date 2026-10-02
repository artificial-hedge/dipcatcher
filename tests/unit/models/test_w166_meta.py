"""Wave-166 meta-learning canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._meta_synth import sine_task
from quant_fund.models.anil_meta import bench_anil_meta
from quant_fund.models.matching_net import bench_matching_net
from quant_fund.models.meta_sgd import bench_meta_sgd
from quant_fund.models.protonet import bench_protonet
from quant_fund.models.r2d2_meta import bench_r2d2_meta
from quant_fund.models.reptile import bench_reptile


class TestMetaSynth:
    def test_task(self) -> None:
        rng = np.random.default_rng(3)
        xs, ys, xq, yq = sine_task(rng, K=3, M=5)
        assert xs.shape == (3,) and yq.shape == (5,)


class TestReptile:
    def test_bench(self) -> None:
        out = bench_reptile(seed=5, n_tasks=4)
        assert np.isfinite(out["synthetic_rep_query_mse"])


class TestProto:
    def test_bench(self) -> None:
        out = bench_protonet(seed=7, n_tasks=4)
        assert np.isfinite(out["synthetic_pn_query_mse"])


class TestMatching:
    def test_bench(self) -> None:
        out = bench_matching_net(seed=9, n_tasks=4)
        assert np.isfinite(out["synthetic_mn_query_mse"])


class TestANIL:
    def test_bench(self) -> None:
        out = bench_anil_meta(seed=11, n_tasks=4)
        assert np.isfinite(out["synthetic_anil_query_mse"])


class TestMetaSGD:
    def test_bench(self) -> None:
        out = bench_meta_sgd(seed=13, n_tasks=4)
        assert np.isfinite(out["synthetic_msgd_query_mse"])


class TestR2D2:
    def test_bench(self) -> None:
        out = bench_r2d2_meta(seed=15, n_tasks=4)
        assert np.isfinite(out["synthetic_r2d2_query_mse"])
