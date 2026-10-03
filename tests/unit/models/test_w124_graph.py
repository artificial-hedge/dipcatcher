"""Wave-124 exec-summary graph/meta/continual module tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.asset_gnn import bench_asset_gnn, build_graph, synth_market
from quant_fund.models.continual_learning import (
    EwcLinear,
    bench_continual_learning,
    regime_panel,
)
from quant_fund.models.counterparty_gnn import (
    auc_score,
    bench_counterparty_gnn,
    synth_system,
)
from quant_fund.models.fed_avg import bench_fed_avg, local_train, synth_shards
from quant_fund.models.insider_anomaly import (
    LinearAe,
    bench_insider_anomaly,
    robust_z,
    synth_flow,
)
from quant_fund.models.maml_portfolio import Maml, alloc_score, bench_maml_portfolio, make_task


class TestAssetGnn:
    def test_graph_normalized(self) -> None:
        rng = np.random.default_rng(0)
        rets, sectors, _ = synth_market(12, 200, rng)
        a = build_graph(rets, sectors)
        assert a.shape == (12, 12)
        assert np.allclose(a, a.T)

    def test_bench_keys(self) -> None:
        out = bench_asset_gnn(seed=3)
        for k in ("synthetic_gnn_oos_r2", "synthetic_gnn_sign_accuracy", "synthetic_gnn_r2_margin"):
            assert k in out
        assert np.isfinite(out["synthetic_gnn_oos_r2"])


class TestCounterpartyGnn:
    def test_system_shapes(self) -> None:
        rng = np.random.default_rng(0)
        feats, adj, dist = synth_system(20, 50, rng)
        assert feats.shape == (20, 4)
        assert adj.shape == (20, 20)
        assert dist.shape == (50, 20)

    def test_gnn_beats_node_only(self) -> None:
        out = bench_counterparty_gnn(seed=5)
        assert out["synthetic_cgnn_auc_margin"] > 0.0

    def test_auc_bounds(self) -> None:
        s = auc_score(np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.1, 0.9, 0.2, 0.8]))
        assert s == 1.0


class TestMaml:
    def test_adapt_moves_theta(self) -> None:
        rng = np.random.default_rng(0)
        m = Maml()
        sup = make_task("trend", 40, rng)[:8]
        w = m.adapt(sup, 3)
        assert np.linalg.norm(w) > 0

    def test_bench_margin_positive(self) -> None:
        out = bench_maml_portfolio(seed=4)
        assert out["synthetic_maml_margin_vs_scratch"] > 0.0
        assert np.isfinite(
            alloc_score(np.ones(6) / 6, make_task("trend", 30, np.random.default_rng(1)))
        )


class TestContinual:
    def test_ewc_remembers(self) -> None:
        out = bench_continual_learning(seed=6)
        assert out["synthetic_ewc_first_task_mse"] < out["synthetic_sgd_first_task_mse"]

    def test_regime_panels(self) -> None:
        rng = np.random.default_rng(0)
        x, y = regime_panel("momentum", 30, rng)
        assert x.shape == (30, 4) and y.shape == (30,)
        e = EwcLinear()
        e.train_task(x, y, epochs=2)
        assert len(e.anchors) == 1


class TestFedAvg:
    def test_close_to_central(self) -> None:
        out = bench_fed_avg(seed=2)
        assert out["synthetic_fedavg_gap_to_central"] < 0.05
        assert out["synthetic_fedavg_margin_vs_local"] > 0.0

    def test_local_train_runs(self) -> None:
        rng = np.random.default_rng(0)
        shards = synth_shards(30, rng)
        w = local_train(shards[0][0], shards[0][1], np.zeros(4), 0.05, 2)
        assert w.shape == (4,) and np.isfinite(w).all()


class TestInsiderAnomaly:
    def test_flow_shapes(self) -> None:
        rng = np.random.default_rng(0)
        x, y = synth_flow(200, rng)
        assert x.shape == (200, 5) and y.shape == (200,)
        assert 0 < y.mean() < 0.3

    def test_ae_scores(self) -> None:
        rng = np.random.default_rng(0)
        x, y = synth_flow(200, rng)
        ae = LinearAe(k=2)
        ae.fit(x[y < 0.5])
        s = robust_z(ae.score(x))
        out = bench_insider_anomaly(seed=7)
        assert out["synthetic_insider_auc"] > 0.8
        assert np.isfinite(s).all()
