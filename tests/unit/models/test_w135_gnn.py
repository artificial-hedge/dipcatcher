"""Wave-135 GNN canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._graph_synth import normalize_adj, synth_sbm_graph
from quant_fund.models.apnp_prop import bench_apnp_prop
from quant_fund.models.chebnet import bench_chebnet
from quant_fund.models.gin_gnn import bench_gin_gnn
from quant_fund.models.graph_unet import bench_graph_unet
from quant_fund.models.graphsage import bench_graphsage
from quant_fund.models.jk_net import bench_jk_net


class TestFixture:
    def test_synth(self) -> None:
        a, x, y = synth_sbm_graph(n=40, seed=0)
        assert a.shape == (40, 40) and x.shape[0] == 40
        an = normalize_adj(a)
        assert np.allclose(np.asarray(an), np.asarray(an).T)


class TestChebNet:
    def test_bench(self) -> None:
        out = bench_chebnet(seed=3, iters=60)
        assert 0 <= out["synthetic_cheb_acc"] <= 1


class TestGraphSage:
    def test_bench(self) -> None:
        out = bench_graphsage(seed=5, iters=60)
        assert 0 <= out["synthetic_sage_acc"] <= 1


class TestGin:
    def test_bench(self) -> None:
        out = bench_gin_gnn(seed=7, iters=60)
        assert 0 <= out["synthetic_gin_acc"] <= 1


class TestGraphUNet:
    def test_bench(self) -> None:
        out = bench_graph_unet(seed=9, iters=60)
        assert 0 <= out["synthetic_gunet_acc"] <= 1


class TestAPPNP:
    def test_bench(self) -> None:
        out = bench_apnp_prop(seed=11, iters=60)
        assert 0 <= out["synthetic_apnp_acc"] <= 1


class TestJKNet:
    def test_bench(self) -> None:
        out = bench_jk_net(seed=13, iters=60)
        assert 0 <= out["synthetic_jk_acc"] <= 1
