"""SYNTHETIC algorithmic/PIT checks for the trainable asset GCN lane.

No fixture result is market evidence. The graph signal fixture deliberately
plants a neighboring-node target; the node-only ablation verifies that the
network actually learns message passing rather than a graph-named MLP.
"""

from __future__ import annotations

import builtins
import importlib.util
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quant_fund.models.asset_graph import (
    AssetGraph,
    AssetGraphForecaster,
    GraphForecast,
    build_graph,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="asset graph fitting needs optional nn extra"
)
ASSETS = ("A", "B", "C", "D")
FEATURES = ("signal", "context")


def _times(n: int, start: int = 0) -> tuple[datetime, ...]:
    return tuple(
        datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(start, start + n)
    )


def _graph(n: int = 80) -> AssetGraph:
    times = _times(n)
    rng = np.random.default_rng(11)
    common = rng.normal(size=n)
    history = np.column_stack(
        [common, common + 0.01 * rng.normal(size=n), -common, rng.normal(size=n)]
    )
    return build_graph(
        history,
        asset_ids=ASSETS,
        timestamps=times,
        available_times=times,
        as_of=times[-1],
        correlation_threshold=0.5,
    )


def _panel(n: int = 80) -> dict:
    times = _times(n)
    rng = np.random.default_rng(40)
    features = rng.normal(size=(n, 4, 2))
    graph = _graph(n)
    s = graph.normalized_adjacency
    mean = (s @ (s @ features))[..., 0]
    targets = mean + 0.05 * rng.normal(size=mean.shape)
    return {
        "features": features,
        "targets": targets,
        "graph": graph,
        "asset_ids": ASSETS,
        "timestamps": times,
        "feature_available_times": times,
        "target_available_times": tuple(t + timedelta(hours=12) for t in times),
        "feature_names": FEATURES,
        "fit_as_of": times[-1] + timedelta(hours=18),
    }


def _prediction_kwargs(graph: AssetGraph, n: int = 40) -> dict:
    times = tuple(graph.as_of + timedelta(days=i + 1) for i in range(n))
    return {
        "graph": graph,
        "asset_ids": graph.asset_ids,
        "timestamps": times,
        "feature_available_times": times,
        "feature_names": FEATURES,
    }


def test_graph_suffix_and_unpublished_rows_cannot_change_adjacency() -> None:
    rng = np.random.default_rng(3)
    history = rng.normal(size=(30, 4))
    times = _times(30)
    cutoff = times[19]
    original = build_graph(
        history[:20],
        asset_ids=ASSETS,
        timestamps=times[:20],
        available_times=times[:20],
        as_of=cutoff,
    )
    history[20:] = 1e6 * rng.normal(size=(10, 4))
    suffix = build_graph(
        history, asset_ids=ASSETS, timestamps=times, available_times=times, as_of=cutoff
    )
    np.testing.assert_array_equal(original.adjacency, suffix.adjacency)
    assert original.observation_times == suffix.observation_times == times[:20]
    availability = list(times)
    availability[19] += timedelta(days=50)
    lagged = build_graph(
        history, asset_ids=ASSETS, timestamps=times, available_times=availability, as_of=cutoff
    )
    assert lagged.observation_times == times[:19]


def test_graph_constant_nodes_sector_vintages_and_hybrid() -> None:
    times = _times(8)
    history = np.column_stack([np.arange(8), -np.arange(8), np.ones(8), np.arange(8) ** 2])
    common = dict(asset_ids=ASSETS, timestamps=times, available_times=times, as_of=times[-1])
    corr = build_graph(history, **common, correlation_threshold=0.9, top_k=1)
    assert corr.adjacency[0, 1] == pytest.approx(1)
    assert not corr.adjacency[2].any()
    sectors = ("tech", "finance", "tech", "finance")
    sector = build_graph(
        history, **common, method="sector", sectors=sectors, sector_available_time=times[0]
    )
    assert sector.adjacency[0, 2] == 1
    assert sector.adjacency[0, 1] == 0
    hybrid = build_graph(
        history, **common, method="hybrid", sectors=sectors, sector_available_time=times[0]
    )
    assert hybrid.adjacency[0, 1] == pytest.approx(1)
    assert hybrid.adjacency[0, 2] == 1
    with pytest.raises(ValueError, match="published"):
        build_graph(
            history,
            **common,
            method="sector",
            sectors=sectors,
            sector_available_time=times[-1] + timedelta(days=1),
        )


def test_induced_subgraph_retains_explicit_order_and_recomputes_normalization() -> None:
    graph = _graph()
    subgraph = graph.subgraph(("C", "A", "D"))
    np.testing.assert_array_equal(subgraph.adjacency, graph.adjacency[np.ix_([2, 0, 3], [2, 0, 3])])
    assert subgraph.asset_ids == ("C", "A", "D")
    assert subgraph.normalized_adjacency.shape == (3, 3)
    assert np.isfinite(subgraph.normalized_adjacency).all()
    with pytest.raises(ValueError, match="unknown"):
        graph.subgraph(("UNKNOWN",))
    with pytest.raises(ValueError):
        graph.subgraph(())


@pytest.mark.parametrize(
    "adjacency",
    [
        np.empty((0, 0)),
        np.array([[0, 1], [0, 0]]),
        np.array([[0, np.nan], [np.nan, 0]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, -1], [-1, 0]]),
    ],
)
def test_invalid_adjacency_fails_closed(adjacency: np.ndarray) -> None:
    with pytest.raises(ValueError):
        AssetGraph(("A", "B"), adjacency, _times(4)[-1], _times(4))


@pytest.mark.parametrize(
    "change",
    ["empty", "nonfinite", "duplicate_time", "naive_time", "asset_order", "future_availability"],
)
def test_graph_input_contract_failures(change: str) -> None:
    times = _times(6)
    data = np.arange(24, dtype=float).reshape(6, 4)
    kwargs = dict(asset_ids=ASSETS, timestamps=times, available_times=times, as_of=times[-1])
    if change == "empty":
        data = np.empty((0, 4))
    elif change == "nonfinite":
        data[1, 2] = np.inf
    elif change == "duplicate_time":
        kwargs["timestamps"] = (*times[:5], times[4])
    elif change == "naive_time":
        kwargs["as_of"] = datetime(2020, 1, 1)
    elif change == "asset_order":
        kwargs["asset_ids"] = ASSETS[:3]
    elif change == "future_availability":
        kwargs["available_times"] = tuple(t + timedelta(days=100) for t in times)
    with pytest.raises(ValueError):
        build_graph(data, **kwargs)


def test_gaussian_proper_scores_and_honesty() -> None:
    forecast = GraphForecast(np.zeros((1, 1)), np.ones((1, 1)), ("A",), _times(1))
    truth = np.zeros((1, 1))
    assert forecast.log_score(truth)[0, 0] == pytest.approx(0.5 * math.log(2 * math.pi))
    assert forecast.crps(truth)[0, 0] == pytest.approx((math.sqrt(2) - 1) / math.sqrt(math.pi))
    assert forecast.research_only and not forecast.live_pnl_claim
    rng = np.random.default_rng(20)
    values = rng.normal(size=(6000, 1))
    times = _times(6000)
    correct = GraphForecast(np.zeros_like(values), np.ones_like(values), ("A",), times)
    wrong = GraphForecast(np.ones_like(values), 2 * np.ones_like(values), ("A",), times)
    assert correct.log_score(values).mean() < wrong.log_score(values).mean()
    assert correct.crps(values).mean() < wrong.crps(values).mean()
    with pytest.raises(ValueError):
        forecast.crps(np.zeros((2, 1)))
    with pytest.raises(ValueError):
        GraphForecast(np.zeros((1, 1)), np.zeros((1, 1)), ("A",), _times(1))
    with pytest.raises(ValueError, match="research-only"):
        GraphForecast(np.zeros((1, 1)), np.ones((1, 1)), ("A",), _times(1), live_pnl_claim=True)


@requires_torch
def test_trainable_message_passing_beats_node_only_on_planted_graph_target() -> None:
    panel = _panel(110)
    gcn = AssetGraphForecaster(hidden_dim=16, epochs=240, seed=5).fit(**panel)
    node = AssetGraphForecaster(hidden_dim=16, epochs=240, seed=5, node_only=True).fit(**panel)
    assert gcn.loss_history[-1] < gcn.loss_history[0] - 0.5
    assert gcn._weights is not None
    rng = np.random.default_rng(101)
    features = rng.normal(size=(50, 4, 2))
    s = panel["graph"].normalized_adjacency
    targets = (s @ (s @ features))[..., 0] + 0.05 * rng.normal(size=(50, 4))
    kwargs = _prediction_kwargs(panel["graph"], n=50)
    gcn_forecast = gcn.predict_from_graph(features, **kwargs)
    node_forecast = node.predict_from_graph(features, **kwargs)
    # SYNTHETIC planted mechanism validation, not a real-market advantage claim.
    assert gcn_forecast.crps(targets).mean() < 0.7 * node_forecast.crps(targets).mean()
    assert gcn_forecast.log_score(targets).mean() < node_forecast.log_score(targets).mean()


@requires_torch
def test_fitting_suffix_invariance_and_label_availability() -> None:
    panel = _panel(25)
    fitted = AssetGraphForecaster(epochs=20, seed=4).fit(**panel)
    suffix = dict(panel)
    suffix["features"] = np.concatenate([panel["features"], np.full((3, 4, 2), 1000.0)])
    suffix["targets"] = np.concatenate([panel["targets"], np.full((3, 4), 1e6)])
    suffix["timestamps"] = _times(28)
    suffix["feature_available_times"] = _times(28)
    suffix["target_available_times"] = tuple(t + timedelta(hours=12) for t in _times(28))
    extended = AssetGraphForecaster(epochs=20, seed=4).fit(**suffix)
    assert fitted.training_times == extended.training_times
    assert fitted.loss_history == extended.loss_history
    for a, b in zip(fitted._weights, extended._weights, strict=True):
        np.testing.assert_array_equal(a, b)
    poisoned = dict(panel)
    poisoned["targets"] = panel["targets"].copy()
    poisoned["targets"][-1] = 1e5
    late = list(panel["target_available_times"])
    late[-1] += timedelta(days=10)
    poisoned["target_available_times"] = late
    unpublished = AssetGraphForecaster(epochs=20, seed=4).fit(**poisoned)
    assert unpublished.training_times == panel["timestamps"][:-1]


def test_torch_absence_is_explicit_and_core_stays_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def no_torch(name: str, *args, **kwargs):
        if name == "torch":
            raise ImportError("deliberately unavailable in this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    graph = _graph(8)
    assert graph.normalized_adjacency.shape == (4, 4)
    with pytest.raises(ImportError, match="optional nn extra"):
        AssetGraphForecaster(epochs=1).fit(**_panel(8))


@requires_torch
def test_exported_numpy_inference_needs_no_torch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _panel(8)
    model = AssetGraphForecaster(epochs=2).fit(**panel)
    original_import = builtins.__import__

    def no_torch(name: str, *args, **kwargs):
        if name == "torch":
            raise ImportError("deliberately unavailable in this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    forecast = model.predict_from_graph(np.ones((2, 4, 2)), **_prediction_kwargs(panel["graph"], 2))
    assert np.isfinite(forecast.mean).all() and (forecast.scale > 0).all()


@requires_torch
def test_asset_permutation_subgraph_and_unseen_asset_predictions() -> None:
    panel = _panel(30)
    model = AssetGraphForecaster(epochs=30, seed=2).fit(**panel)
    kwargs = _prediction_kwargs(panel["graph"], n=3)
    data = np.random.default_rng(4).normal(size=(3, 4, 2))
    original = model.predict_from_graph(data, **kwargs)
    perm = [2, 0, 3, 1]
    ids = tuple(ASSETS[i] for i in perm)
    perm_kwargs = {**kwargs, "graph": panel["graph"].subgraph(ids), "asset_ids": ids}
    reordered = model.predict_from_graph(data[:, perm], **perm_kwargs)
    np.testing.assert_allclose(original.mean[:, perm], reordered.mean, atol=1e-12)
    np.testing.assert_allclose(original.scale[:, perm], reordered.scale, atol=1e-12)
    subset = panel["graph"].subgraph(("A", "C"))
    subset_out = model.predict_from_graph(
        data[:, [0, 2]], **{**kwargs, "graph": subset, "asset_ids": subset.asset_ids}
    )
    assert subset_out.mean.shape == (3, 2)
    # IDs have never been seen during fitting; no node identity parameters are used.
    heldout = AssetGraph(
        ("HELDOUT1", "HELDOUT2"),
        np.array([[0, 0.7], [0.7, 0]]),
        panel["graph"].as_of,
        panel["graph"].observation_times,
    )
    out = model.predict_from_graph(
        data[:, :2], **{**kwargs, "graph": heldout, "asset_ids": heldout.asset_ids}
    )
    assert out.asset_ids == heldout.asset_ids
    assert out.mean.shape == (3, 2) and np.isfinite(out.scale).all()


@pytest.mark.parametrize(
    "change",
    [
        "nonfinite",
        "misordered_assets",
        "misordered_dates",
        "future_feature",
        "contemporaneous_label",
        "future_graph",
        "empty",
        "shape",
    ],
)
def test_fit_rejects_misaligned_or_noncausal_panels_before_importing_torch(change: str) -> None:
    panel = _panel(8)
    if change == "nonfinite":
        panel["features"][0, 0, 0] = np.nan
    elif change == "misordered_assets":
        panel["asset_ids"] = ASSETS[::-1]
    elif change == "misordered_dates":
        panel["timestamps"] = panel["timestamps"][::-1]
    elif change == "future_feature":
        panel["feature_available_times"] = tuple(
            t + timedelta(seconds=1) for t in panel["timestamps"]
        )
    elif change == "contemporaneous_label":
        panel["target_available_times"] = panel["timestamps"]
    elif change == "future_graph":
        panel["fit_as_of"] = panel["graph"].as_of - timedelta(days=1)
    elif change == "empty":
        panel["features"] = np.empty((0, 4, 2))
    elif change == "shape":
        panel["targets"] = panel["targets"][:, :3]
    with pytest.raises(ValueError):
        AssetGraphForecaster(epochs=1).fit(**panel)


@requires_torch
@pytest.mark.parametrize(
    "change",
    ["feature_order", "asset_order", "feature_unavailable", "past_decision", "future_graph"],
)
def test_prediction_alignment_and_frozen_graph_fail_closed(change: str) -> None:
    panel = _panel(8)
    model = AssetGraphForecaster(epochs=2).fit(**panel)
    kwargs = _prediction_kwargs(panel["graph"], n=2)
    data = np.ones((2, 4, 2))
    if change == "feature_order":
        kwargs["feature_names"] = FEATURES[::-1]
    elif change == "asset_order":
        kwargs["asset_ids"] = ASSETS[::-1]
    elif change == "feature_unavailable":
        kwargs["feature_available_times"] = tuple(
            t + timedelta(seconds=1) for t in kwargs["timestamps"]
        )
    elif change == "past_decision":
        kwargs["timestamps"] = _times(2)
        kwargs["feature_available_times"] = _times(2)
    elif change == "future_graph":
        kwargs["graph"] = AssetGraph(
            ASSETS, panel["graph"].adjacency, _times(105)[-1], panel["graph"].observation_times
        )
    with pytest.raises(ValueError):
        model.predict_from_graph(data, **kwargs)


def test_unfitted_and_invalid_configuration_fail_closed() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        AssetGraphForecaster().predict_from_graph(
            np.ones((2, 4, 2)), **_prediction_kwargs(_graph(), 2)
        )
    for kwargs in (
        {"epochs": 0},
        {"hidden_dim": True},
        {"learning_rate": math.nan},
        {"min_scale": 0},
        {"seed": -1},
        {"node_only": 1},
    ):
        with pytest.raises(ValueError):
            AssetGraphForecaster(**kwargs)
