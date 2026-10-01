"""Trainable graph Gaussian forecasts for timestamped multi-asset research.

Two graph convolutions use the Kipf-Welling renormalization
``S = D^-1/2 (A + I) D^-1/2`` (ICLR 2017, arXiv:1609.02907), with shared
node weights and a Gaussian mean/scale head fitted by proper log score.
This is an adaptation to panel regression, not a reproduction of that
paper's classification results. ``node_only=True`` replaces S by I while
retaining the same parameter count and optimizer for a useful ablation.

Graphs use only observations published by an explicit training cutoff;
features and labels carry separate availability clocks. Prediction requires
decision clocks after the fit cutoff and a graph frozen by that cutoff.
Global training-only feature statistics and shared weights permit unseen
assets and induced subgraphs without identity embeddings. Asset and feature
order must match explicitly; nothing silently sorts or imputes a panel.

Torch is optional and imported only by fit. Exported float64 weights make
inference and Gaussian CRPS numpy/scipy-only. CPU fitting is seeded; GPU
determinism is not claimed. Results are research-only: synthetic tests are
correctness evidence, not market evidence or a SOTA/performance claim.
The lane does not procure PIT sector vintages or establish a dynamic graph,
spatiotemporal architecture, cross-asset joint density, or trading policy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

Array = NDArray[np.float64]

__all__ = ["AssetGraph", "AssetGraphForecaster", "GraphForecast", "build_graph"]


def _ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if (
        not result
        or any(not isinstance(v, str) or not v.strip() for v in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain unique nonempty strings")
    return result


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware datetimes")
    return value.astimezone(UTC)


def _times(values: Sequence[datetime], n: int) -> tuple[datetime, ...]:
    result = tuple(_utc(v, "timestamps") for v in values)
    if (
        len(result) != n
        or not result
        or any(a >= b for a, b in zip(result, result[1:], strict=False))
    ):
        raise ValueError("timestamps must align with rows and be strictly increasing")
    return result


def _availability(values: Any, shape: tuple[int, int], name: str) -> NDArray[Any]:
    arr = np.asarray(values, dtype=object)
    if arr.shape == (shape[0],):
        arr = np.broadcast_to(arr[:, None], shape)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape} or ({shape[0]},)")
    return np.asarray([_utc(v, name) for v in arr.flat], dtype=object).reshape(shape)


def _array(value: Any, ndim: int, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or any(s == 0 for s in result.shape) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}-dimensional array")
    return result


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class AssetGraph:
    """Undirected nonnegative graph in explicit asset order, without self edges.

    ``observation_times`` records the history visible at ``as_of``. For a
    sector graph ``sector_available_time`` must independently prove that its
    membership was published by ``as_of``. Arrays are copied and read-only.
    """

    asset_ids: tuple[str, ...]
    adjacency: Array
    as_of: datetime
    observation_times: tuple[datetime, ...]
    method: str = "explicit"

    def __post_init__(self) -> None:
        ids = _ids(self.asset_ids, "asset_ids")
        adjacency = _array(self.adjacency, 2, "adjacency").copy()
        if adjacency.shape != (len(ids), len(ids)):
            raise ValueError("adjacency shape must align with asset_ids")
        if (
            (adjacency < 0).any()
            or not np.allclose(adjacency, adjacency.T, rtol=0, atol=1e-12)
            or not np.allclose(np.diag(adjacency), 0, rtol=0, atol=1e-12)
        ):
            raise ValueError("adjacency must be symmetric, nonnegative, and have a zero diagonal")
        cutoff = _utc(self.as_of, "as_of")
        history = _times(self.observation_times, len(self.observation_times))
        if history[-1] > cutoff:
            raise ValueError("graph observations after as_of are forbidden")
        adjacency.setflags(write=False)
        object.__setattr__(self, "asset_ids", ids)
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(self, "as_of", cutoff)
        object.__setattr__(self, "observation_times", history)

    @property
    def normalized_adjacency(self) -> Array:
        """Self-loop augmented symmetric degree normalization, including isolates."""
        weights = self.adjacency + np.eye(len(self.asset_ids))
        inv_degree = 1.0 / np.sqrt(weights.sum(axis=1))
        return np.asarray(weights * inv_degree[:, None] * inv_degree[None, :])

    def subgraph(self, asset_ids: Sequence[str]) -> AssetGraph:
        """Induced subgraph in requested order; renormalize its own degrees."""
        ids = _ids(asset_ids, "asset_ids")
        if not set(ids).issubset(self.asset_ids):
            raise ValueError("subgraph contains unknown asset ids")
        positions = [self.asset_ids.index(v) for v in ids]
        return AssetGraph(
            ids,
            self.adjacency[np.ix_(positions, positions)],
            self.as_of,
            self.observation_times,
            self.method,
        )


def build_graph(
    observations: Array,
    *,
    asset_ids: Sequence[str],
    timestamps: Sequence[datetime],
    available_times: Any,
    as_of: datetime,
    method: str = "correlation",
    correlation_threshold: float = 0.3,
    top_k: int | None = None,
    sectors: Sequence[str] | None = None,
    sector_available_time: datetime | None = None,
) -> AssetGraph:
    """Build a frozen graph from jointly PIT-visible training observations.

    Correlation weights are absolute Pearson correlations above a threshold;
    constant series are isolated. ``top_k`` sparsifies each row then takes the
    symmetric union, so a node may have more than k incoming neighbors.
    Sector edges have weight one; hybrid takes the maximum of both graphs.
    ``sectors`` must be aligned PIT memberships, not today's classification
    retroactively applied. Future rows and unpublished rows are excluded.
    """
    data = _array(observations, 2, "observations")
    ids = _ids(asset_ids, "asset_ids")
    if data.shape[1] != len(ids):
        raise ValueError("observations must align with asset_ids")
    times = _times(timestamps, data.shape[0])
    available = _availability(available_times, data.shape, "available_times")
    event = np.asarray(times, dtype=object)[:, None]
    if (available < event).any():
        raise ValueError("graph available_times cannot precede observation timestamps")
    cutoff = _utc(as_of, "as_of")
    visible = (event[:, 0] <= cutoff) & np.all(available <= cutoff, axis=1)
    if np.count_nonzero(visible) < 4:
        raise ValueError("graph needs at least four jointly visible training observations")
    if method not in {"correlation", "sector", "hybrid"}:
        raise ValueError("method must be correlation, sector, or hybrid")
    if not math.isfinite(correlation_threshold) or not 0 <= correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in [0, 1]")
    if top_k is not None:
        top_k = _count(top_k, "top_k")
    weights = np.zeros((len(ids), len(ids)))
    if method in {"correlation", "hybrid"}:
        train = data[visible]
        centered = train - train.mean(axis=0)
        norm = np.sqrt(np.sum(centered**2, axis=0))
        denominator = norm[:, None] * norm[None, :]
        corr = np.divide(
            centered.T @ centered, denominator, out=np.zeros_like(weights), where=denominator > 0
        )
        weights = np.clip(np.abs(corr), 0, 1)
        weights[weights < correlation_threshold] = 0
        np.fill_diagonal(weights, 0)
        if top_k is not None:
            keep = np.argsort(-weights, axis=1, kind="stable")[:, :top_k]
            sparse = np.zeros_like(weights)
            np.put_along_axis(sparse, keep, np.take_along_axis(weights, keep, axis=1), axis=1)
            weights = np.maximum(sparse, sparse.T)
    if method in {"sector", "hybrid"}:
        if sectors is None or len(sectors) != len(ids):
            raise ValueError("sector memberships must align with asset_ids")
        if any(not isinstance(v, str) or not v.strip() for v in sectors):
            raise ValueError("sector memberships must be nonempty strings")
        if (
            sector_available_time is None
            or _utc(sector_available_time, "sector_available_time") > cutoff
        ):
            raise ValueError("sector membership must be published by graph as_of")
        membership = np.asarray(sectors)
        sector_graph = (membership[:, None] == membership[None, :]).astype(float)
        np.fill_diagonal(sector_graph, 0)
        weights = np.maximum(weights, sector_graph)
    return AssetGraph(
        ids,
        weights,
        cutoff,
        tuple(t for t, keep in zip(times, visible, strict=True) if keep),
        method,
    )


@dataclass(frozen=True)
class GraphForecast:
    """Marginal Gaussian forecasts in declared decision-time and asset order."""

    mean: Array
    scale: Array
    asset_ids: tuple[str, ...]
    timestamps: tuple[datetime, ...]
    research_only: bool = True
    live_pnl_claim: bool = False

    def __post_init__(self) -> None:
        mean = _array(self.mean, 2, "mean").copy()
        scale = _array(self.scale, 2, "scale").copy()
        ids = _ids(self.asset_ids, "asset_ids")
        times = _times(self.timestamps, mean.shape[0])
        if mean.shape != scale.shape or mean.shape[1] != len(ids) or (scale <= 0).any():
            raise ValueError("forecast mean/scale must align and scales must be positive")
        if not self.research_only or self.live_pnl_claim:
            raise ValueError("asset graph forecasts are research-only")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "asset_ids", ids)
        object.__setattr__(self, "timestamps", times)

    def _targets(self, targets: Array) -> Array:
        values = _array(targets, 2, "targets")
        if values.shape != self.mean.shape:
            raise ValueError("targets must align with forecast shape")
        return values

    def log_score(self, targets: Array) -> Array:
        """Negative log density (lower is better), one proper score per node/date."""
        z = (self._targets(targets) - self.mean) / self.scale
        return np.asarray(0.5 * math.log(2 * math.pi) + np.log(self.scale) + 0.5 * z**2)

    def crps(self, targets: Array) -> Array:
        """Closed-form Gaussian CRPS, including the original target scale."""
        z = (self._targets(targets) - self.mean) / self.scale
        phi = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
        return np.asarray(self.scale * (z * (2 * ndtr(z) - 1) + 2 * phi - 1 / math.sqrt(math.pi)))


class AssetGraphForecaster:
    """Two trainable GCN layers and a shared Gaussian distributional head.

    Fit uses a static training graph, not a fitted identity embedding. Shared
    normalization across nodes allows induction onto unseen IDs with the
    same feature contract. ``predict_from_graph`` also accepts a frozen
    induced subgraph, but changing neighborhoods can change predictions.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 24,
        epochs: int = 200,
        learning_rate: float = 0.01,
        min_scale: float = 0.02,
        seed: int = 0,
        node_only: bool = False,
    ) -> None:
        self.hidden_dim = _count(hidden_dim, "hidden_dim")
        self.epochs = _count(epochs, "epochs")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be positive and finite")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(node_only, bool):
            raise ValueError("node_only must be a boolean")
        self.learning_rate = learning_rate
        self.min_scale = min_scale
        self.seed = int(seed)
        self.node_only = node_only
        self.fit_as_of: datetime | None = None
        self.feature_names: tuple[str, ...] = ()
        self.training_graph: AssetGraph | None = None
        self.training_times: tuple[datetime, ...] = ()
        self.loss_history: tuple[float, ...] = ()
        self._weights: tuple[Array, Array, Array, Array] | None = None
        self._x_mean = np.empty(0)
        self._x_scale = np.empty(0)
        self._y_mean = 0.0
        self._y_scale = 1.0

    def _panel(
        self,
        features: Array,
        graph: AssetGraph,
        asset_ids: Sequence[str],
        timestamps: Sequence[datetime],
        feature_available_times: Any,
        feature_names: Sequence[str],
    ) -> tuple[Array, tuple[datetime, ...], tuple[str, ...]]:
        data = _array(features, 3, "features")
        if not isinstance(graph, AssetGraph):
            raise ValueError("graph must be an AssetGraph")
        ids = _ids(asset_ids, "asset_ids")
        names = _ids(feature_names, "feature_names")
        if ids != graph.asset_ids or data.shape[1:] != (len(ids), len(names)):
            raise ValueError(
                "feature node order/shape must match graph asset_ids and feature_names"
            )
        times = _times(timestamps, data.shape[0])
        available = _availability(
            feature_available_times, data.shape[:2], "feature_available_times"
        )
        if (available > np.asarray(times, dtype=object)[:, None]).any():
            raise ValueError("features unavailable at decision time are forbidden")
        return data, times, names

    def fit(
        self,
        features: Array,
        targets: Array,
        *,
        graph: AssetGraph,
        asset_ids: Sequence[str],
        timestamps: Sequence[datetime],
        feature_available_times: Any,
        target_available_times: Any,
        feature_names: Sequence[str],
        fit_as_of: datetime,
    ) -> AssetGraphForecaster:
        """Fit only rows whose forward labels are published by fit_as_of.

        Labels must become available strictly after their decision timestamp;
        contemporaneous labels cannot masquerade as forward forecasting.
        Unpublished suffixes do not affect weights, statistics, or history.
        Graph history must have been frozen no later than the fit cutoff.
        """
        data, times, names = self._panel(
            features, graph, asset_ids, timestamps, feature_available_times, feature_names
        )
        values = _array(targets, 2, "targets")
        if values.shape != data.shape[:2]:
            raise ValueError("targets must align with feature dates and assets")
        cutoff = _utc(fit_as_of, "fit_as_of")
        if graph.as_of > cutoff:
            raise ValueError("graph was constructed after fit_as_of")
        published = _availability(target_available_times, values.shape, "target_available_times")
        event = np.asarray(times, dtype=object)[:, None]
        if (published <= event).any():
            raise ValueError("forward targets must be published after their decision timestamps")
        keep = (event[:, 0] <= cutoff) & np.all(published <= cutoff, axis=1)
        if np.count_nonzero(keep) < 4:
            raise ValueError("fit needs at least four chronologically eligible label rows")
        x_train = data[keep]
        y_train = values[keep]
        x_mean = x_train.mean(axis=(0, 1))
        x_scale = x_train.std(axis=(0, 1))
        x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
        y_mean = float(y_train.mean())
        y_scale = float(y_train.std())
        if not (
            np.isfinite(x_mean).all()
            and np.isfinite(x_scale).all()
            and math.isfinite(y_mean)
            and math.isfinite(y_scale)
        ):
            raise ValueError("training normalization statistics became nonfinite")
        if y_scale <= 1e-12:
            raise ValueError("targets must have nonzero training variance")
        try:
            import torch
        except ImportError as exc:
            raise ImportError("asset graph fit requires the optional nn extra (torch)") from exc
        adjacency = np.eye(len(graph.asset_ids)) if self.node_only else graph.normalized_adjacency
        losses: list[float] = []
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                first = torch.nn.Linear(len(names), self.hidden_dim, dtype=torch.float64)
                second = torch.nn.Linear(self.hidden_dim, 2, dtype=torch.float64)
                parameters = [*first.parameters(), *second.parameters()]
                optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
                x_tensor = torch.tensor((x_train - x_mean) / x_scale, dtype=torch.float64)
                y_tensor = torch.tensor((y_train - y_mean) / y_scale, dtype=torch.float64)
                s_tensor = torch.tensor(adjacency, dtype=torch.float64)
                for _ in range(self.epochs):
                    optimizer.zero_grad()
                    hidden = torch.relu(first(s_tensor @ x_tensor))
                    output = second(s_tensor @ hidden)
                    mean = output[..., 0]
                    scale = torch.nn.functional.softplus(output[..., 1]) + self.min_scale
                    loss = (
                        0.5 * math.log(2 * math.pi)
                        + torch.log(scale)
                        + 0.5 * ((y_tensor - mean) / scale) ** 2
                    ).mean() + math.log(y_scale)
                    measured = float(loss.detach())
                    if not math.isfinite(measured):
                        raise ValueError("graph Gaussian training loss became nonfinite")
                    losses.append(measured)
                    loss.backward()
                    optimizer.step()
                weights = tuple(
                    p.detach().numpy().copy()
                    for p in (first.weight, first.bias, second.weight, second.bias)
                )
        finally:
            torch.set_num_threads(old_threads)
        if not all(np.isfinite(w).all() for w in weights):
            raise ValueError("graph fitted weights became nonfinite")
        self._weights = (weights[0], weights[1], weights[2], weights[3])
        self._x_mean, self._x_scale = x_mean, x_scale
        self._y_mean, self._y_scale = y_mean, y_scale
        self.fit_as_of = cutoff
        self.feature_names = names
        self.training_graph = graph
        self.training_times = tuple(t for t, eligible in zip(times, keep, strict=True) if eligible)
        self.loss_history = tuple(losses)
        return self

    def predict_from_graph(
        self,
        features: Array,
        *,
        graph: AssetGraph,
        asset_ids: Sequence[str],
        timestamps: Sequence[datetime],
        feature_available_times: Any,
        feature_names: Sequence[str],
    ) -> GraphForecast:
        """Predict on future rows/unseen assets with the same feature contract."""
        if self._weights is None or self.fit_as_of is None:
            raise ValueError("asset graph forecaster is not fitted")
        data, times, names = self._panel(
            features, graph, asset_ids, timestamps, feature_available_times, feature_names
        )
        if names != self.feature_names:
            raise ValueError("prediction feature order differs from fitted feature_names")
        if times[0] <= self.fit_as_of or graph.as_of > self.fit_as_of:
            raise ValueError(
                "prediction must be after fit_as_of using a training-only frozen graph"
            )
        s = np.eye(len(graph.asset_ids)) if self.node_only else graph.normalized_adjacency
        w1, b1, w2, b2 = self._weights
        hidden = np.maximum((s @ ((data - self._x_mean) / self._x_scale)) @ w1.T + b1, 0)
        output = (s @ hidden) @ w2.T + b2
        mean = output[..., 0] * self._y_scale + self._y_mean
        scale = (np.logaddexp(0, output[..., 1]) + self.min_scale) * self._y_scale
        return GraphForecast(mean, scale, graph.asset_ids, times)
