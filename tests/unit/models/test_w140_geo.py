"""Wave-140 geometric/structured nets canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._geo_synth import (
    synth_field,
    synth_hierarchy,
    synth_monotonic,
    synth_partwhole,
    synth_rot_cloud,
    synth_sort,
)
from quant_fund.models.capsule_dynamic import bench_capsule_dynamic
from quant_fund.models.equivar_gnn import bench_equivar_gnn
from quant_fund.models.hyperbolic_nn import bench_hyperbolic_nn
from quant_fund.models.monotonic_net import bench_monotonic_net
from quant_fund.models.siren_inr import bench_siren_inr
from quant_fund.models.sort_net import bench_sort_net


class TestFixture:
    def test_hierarchy(self) -> None:
        v, i, d = synth_hierarchy(3, np.random.default_rng(0))
        assert v.shape == (8, 2) and d.shape == (8, 8)

    def test_partwhole(self) -> None:
        x, y = synth_partwhole(8, np.random.default_rng(0))
        assert x.shape == (8, 4, 4) and y.shape == (8,)

    def test_field(self) -> None:
        p, f, g = synth_field(8, np.random.default_rng(0))
        assert p.shape == (8, 2) and g.shape == (8, 2)

    def test_rot(self) -> None:
        x, y = synth_rot_cloud(8, 6, np.random.default_rng(0))
        assert x.shape == (8, 6, 3)

    def test_mono(self) -> None:
        x, y = synth_monotonic(8, np.random.default_rng(0))
        assert x.shape == (8, 1)

    def test_sort(self) -> None:
        x, r = synth_sort(8, 5, np.random.default_rng(0))
        assert x.shape == (8, 5) and r.max() == 4


class TestHyp:
    def test_bench(self) -> None:
        out = bench_hyperbolic_nn(seed=3, depth=3, iters=60)
        assert out["synthetic_hyp_distortion"] >= 0


class TestCapsule:
    def test_bench(self) -> None:
        out = bench_capsule_dynamic(seed=5, n_train=80, n_test=40, iters=40)
        assert 0 <= out["synthetic_capsule_acc"] <= 1


class TestSiren:
    def test_bench(self) -> None:
        out = bench_siren_inr(seed=7, n_train=100, iters=60)
        assert out["synthetic_siren_mse"] >= 0


class TestEquivar:
    def test_bench(self) -> None:
        out = bench_equivar_gnn(seed=9, n_train=80, n_test=40, m_pts=6, iters=40)
        assert 0 <= out["synthetic_equivar_acc_rot"] <= 1


class TestMono:
    def test_bench(self) -> None:
        out = bench_monotonic_net(seed=11, n_train=80, n_probe=40, iters=40, n_terms=6)
        assert 0 <= out["synthetic_mono_viol"] <= 1


class TestSort:
    def test_bench(self) -> None:
        out = bench_sort_net(seed=13, n_train=80, n_test=40, m=5, iters=40)
        assert -1 <= out["synthetic_sortnet_spearman"] <= 1
