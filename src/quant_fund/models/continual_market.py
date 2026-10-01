"""Causal continual Gaussian regression with delayed labels and bounded replay.

The learned linear mean and homoscedastic variance minimize Gaussian NLL
using analytic gradients and momentum SGD. Welford statistics incorporate
each newly published training example once; replay never recounts it.
Changing feature coordinates rebases both mean parameters and momentum.

The replay reservoir implements Algorithm R from Vitter (1985),
https://www.cs.umd.edu/~samir/498/vitter.pdf . Experience replay is motivated
by Rolnick et al. (2019), https://arxiv.org/abs/1811.11682 ; this numeric
regression pilot does not reproduce CLEAR's RL/behavioral-cloning method.
Drift means a configurable increase between two windows of already known
prequential NLLs, not a calibrated regime test. Its only adaptation is a
bounded learning-rate multiplier on the current published-label update.

All forecasts precede their targets. Snapshots bind pending forecasts,
preprocessing, optimizer, reservoir, RNG, clocks and provenance. This is
research-only infrastructure; synthetic tests are not empirical evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
_SCHEMA = "continual_market_v1"
_MAX_SNAPSHOT_BYTES = 16_000_000
_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _json(value: Any) -> str:
    def encode(item: Any) -> Any:
        if isinstance(item, datetime):
            return item.isoformat()
        raise TypeError("unsupported snapshot value")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=encode)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _clock(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty bounded string")
    return value


def _digest(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("a lowercase SHA256 evidence digest is required")
    return value


def _number(value: float, name: str, bound: float = 1e6) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
        or abs(value) > bound
    ):
        raise ValueError(f"{name} must be finite and bounded")
    return float(value)


def _names(values: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("feature names require an explicit ordered sequence")
    names = tuple(values)
    if not 1 <= len(names) <= 64 or len(set(names)) != len(names):
        raise ValueError("one to64 unique feature names required")
    for name in names:
        _text(name, "feature name")
    return names


@dataclass(frozen=True)
class MarketObservation:
    record_id: str
    asset_id: str
    feature_names: tuple[str, ...]
    features: tuple[float, ...]
    event_time: datetime
    available_time: datetime
    target_event_time: datetime
    data_source: str
    evidence_sha256: str
    synthetic: bool
    schema_version: str = field(default=_SCHEMA, init=False)

    def __post_init__(self) -> None:
        for name in ("record_id", "asset_id", "data_source"):
            _text(getattr(self, name), name)
        _digest(self.evidence_sha256)
        names = _names(self.feature_names)
        values = tuple(_number(v, "feature") for v in self.features)
        if len(values) != len(names) or type(self.synthetic) is not bool:
            raise ValueError("feature axes and explicit synthetic flag required")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "features", values)
        for name in ("event_time", "available_time", "target_event_time"):
            object.__setattr__(self, name, _clock(getattr(self, name), name))
        if self.available_time < self.event_time or self.target_event_time <= self.event_time:
            raise ValueError("publication must follow feature event; target must be later")


@dataclass(frozen=True)
class MarketTarget:
    record_id: str
    value: float
    event_time: datetime
    available_time: datetime
    data_source: str
    evidence_sha256: str
    synthetic: bool
    schema_version: str = field(default=_SCHEMA, init=False)

    def __post_init__(self) -> None:
        _text(self.record_id, "record_id")
        _text(self.data_source, "data_source")
        _digest(self.evidence_sha256)
        object.__setattr__(self, "value", _number(self.value, "target"))
        for name in ("event_time", "available_time"):
            object.__setattr__(self, name, _clock(getattr(self, name), name))
        if self.available_time < self.event_time or type(self.synthetic) is not bool:
            raise ValueError(
                "target publication cannot precede event; explicit synthetic flag required"
            )


@dataclass(frozen=True)
class ContinualConfig:
    learning_rate: float = 0.02
    momentum: float = 0.8
    updates_per_label: int = 4
    reservoir_capacity: int = 128
    replay_batch_size: int = 16
    min_labels: int = 8
    feature_scale_floor: float = 0.1
    initial_scale: float = 1.0
    min_scale: float = 0.01
    max_scale: float = 1_000.0
    gradient_clip: float = 10.0
    batch_ridge: float = 0.001
    drift_window: int = 32
    drift_threshold: float = 1.0
    drift_learning_rate_multiplier: float = 2.0
    max_pending: int = 512
    max_observations: int = 10_000
    seed: int = 7

    def __post_init__(self) -> None:
        for name, low, high in (
            ("updates_per_label", 1, 64),
            ("reservoir_capacity", 1, 512),
            ("replay_batch_size", 0, 128),
            ("min_labels", 4, 512),
            ("drift_window", 4, 512),
            ("max_pending", 1, 512),
            ("max_observations", 8, 100_000),
            ("seed", 0, 2**63 - 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} is outside the resource bound")
        for name in (
            "feature_scale_floor",
            "initial_scale",
            "min_scale",
            "max_scale",
            "gradient_clip",
            "batch_ridge",
            "drift_threshold",
        ):
            value = _number(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.learning_rate <= 0.1 or not 0 <= self.momentum < 1:
            raise ValueError("valid bounded learning rate and momentum required")
        if (
            not 1 <= self.drift_learning_rate_multiplier <= 4
            or self.learning_rate * self.drift_learning_rate_multiplier > 0.1
        ):
            raise ValueError("drift learning rate must remain bounded by0.1")
        if (
            not self.min_scale <= self.initial_scale <= self.max_scale
            or self.max_observations < self.min_labels
        ):
            raise ValueError("consistent warmup and scale bounds required")


@dataclass(frozen=True)
class ProperScores:
    gaussian_nll: float
    gaussian_crps: float


def _gaussian_scores(mean: float, scale: float, value: float) -> ProperScores:
    target = _number(value, "target")
    z = (target - mean) / scale
    nll = 0.5 * z * z + math.log(scale) + 0.5 * math.log(2 * math.pi)
    phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    crps = scale * (z * math.erf(z / math.sqrt(2)) + 2 * phi - 1 / math.sqrt(math.pi))
    if not math.isfinite(nll) or not math.isfinite(crps):
        raise FloatingPointError("non-finite proper scores")
    return ProperScores(nll, max(crps, 0.0))


@dataclass(frozen=True)
class ContinualForecast:
    record_id: str
    decision_time: datetime
    target_event_time: datetime
    status: str
    mean: float | None
    scale: float | None
    model_sha256: str | None
    feature_sha256: str
    state_sha256: str
    synthetic: bool
    prediction_sha256: str = field(init=False)
    research_only: bool = field(default=True, init=False)
    market_evidence: bool = field(default=False, init=False)
    live_pnl_claim: bool = field(default=False, init=False)
    schema_version: str = field(default=_SCHEMA, init=False)

    def __post_init__(self) -> None:
        _text(self.record_id, "record_id")
        _digest(self.feature_sha256)
        _digest(self.state_sha256)
        for name in ("decision_time", "target_event_time"):
            object.__setattr__(self, name, _clock(getattr(self, name), name))
        if self.target_event_time <= self.decision_time or type(self.synthetic) is not bool:
            raise ValueError("prediction must precede target and carry explicit provenance")
        if self.status == "cold_start":
            if any(v is not None for v in (self.mean, self.scale, self.model_sha256)):
                raise ValueError("cold start cannot fabricate a Gaussian")
        elif self.status == "forecast":
            if self.mean is None or self.scale is None or self.model_sha256 is None:
                raise ValueError("fitted Gaussian parameters and identity required")
            if not math.isfinite(self.mean) or not math.isfinite(self.scale) or self.scale <= 0:
                raise ValueError("finite Gaussian parameters required")
            _digest(self.model_sha256)
        else:
            raise ValueError("unsupported forecast status")
        values = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "prediction_sha256"
        }
        object.__setattr__(self, "prediction_sha256", _hash(values))

    def score(self, value: float) -> ProperScores:
        if self.prediction_sha256 != _hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "prediction_sha256"
            }
        ):
            raise RuntimeError("immutable prequential forecast changed")
        if self.mean is None or self.scale is None:
            raise RuntimeError("cold-start prediction has no proper score")
        return _gaussian_scores(self.mean, self.scale, value)


@dataclass(frozen=True)
class FrozenContinualGaussian:
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    scale: float
    fit_as_of: datetime
    training_source_sha256: str
    training_synthetic: bool
    model_sha256: str = field(init=False)
    schema_version: str = field(default=_SCHEMA, init=False)

    def __post_init__(self) -> None:
        names = _names(self.feature_names)
        object.__setattr__(self, "feature_names", names)
        for name in ("feature_mean", "feature_scale", "coefficients"):
            values = tuple(float(v) for v in getattr(self, name))
            if len(values) != len(names) or not all(math.isfinite(v) for v in values):
                raise ValueError("frozen model axes/parameters must be finite")
            object.__setattr__(self, name, values)
        if (
            any(v <= 0 for v in self.feature_scale)
            or not math.isfinite(self.intercept)
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("positive frozen scale and finite intercept required")
        _digest(self.training_source_sha256)
        if type(self.training_synthetic) is not bool:
            raise ValueError("explicit training synthetic flag required")
        object.__setattr__(self, "fit_as_of", _clock(self.fit_as_of, "fit_as_of"))
        object.__setattr__(
            self,
            "model_sha256",
            _hash(
                {
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "model_sha256"
                }
            ),
        )

    def predict(
        self, observation: MarketObservation, *, decision_time: datetime
    ) -> ContinualForecast:
        if self.model_sha256 != _hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "model_sha256"
            }
        ):
            raise RuntimeError("frozen parameters or identity changed")
        decision = _decision(observation, decision_time, self.feature_names)
        if decision < self.fit_as_of:
            raise ValueError("frozen model was fitted after this decision")
        z = (np.asarray(observation.features) - self.feature_mean) / self.feature_scale
        mean = float(z @ self.coefficients + self.intercept)
        return ContinualForecast(
            observation.record_id,
            decision,
            observation.target_event_time,
            "forecast",
            mean,
            self.scale,
            self.model_sha256,
            _hash(asdict(observation)),
            self.model_sha256,
            observation.synthetic or self.training_synthetic,
        )


@dataclass(frozen=True)
class ContinualSnapshot:
    state_json: str
    sha256: str
    schema_version: str = field(default="continual_market_snapshot_v1", init=False)

    def __post_init__(self) -> None:
        _digest(self.sha256)
        if (
            not isinstance(self.state_json, str)
            or len(self.state_json.encode()) > _MAX_SNAPSHOT_BYTES
        ):
            raise ValueError("snapshot exceeds bounded resource size")
        if _hash(json.loads(self.state_json)) != self.sha256:
            raise ValueError("snapshot digest mismatch")


@dataclass(frozen=True)
class PrequentialUpdate:
    record_id: str
    forecast: ContinualForecast
    scores: ProperScores | None
    target_sha256: str
    observed_as_of: datetime
    drift_alert: bool
    effective_learning_rate: float
    unique_training_count: int
    optimizer_steps: int
    state_before_sha256: str
    state_after_sha256: str
    synthetic: bool
    research_only: bool = field(default=True, init=False)
    market_evidence: bool = field(default=False, init=False)
    live_pnl_claim: bool = field(default=False, init=False)


def _decision(observation: MarketObservation, value: datetime, names: tuple[str, ...]) -> datetime:
    decision = _clock(value, "decision_time")
    if observation.feature_names != names:
        raise ValueError("explicit feature names/order must match model")
    if not observation.available_time <= decision < observation.target_event_time:
        raise ValueError("features must be published by decision, before target event")
    return decision


def _observation(payload: dict[str, Any]) -> MarketObservation:
    data = dict(payload)
    data.pop("schema_version", None)
    for name in ("event_time", "available_time", "target_event_time"):
        data[name] = datetime.fromisoformat(data[name])
    return MarketObservation(**data)


def _target(payload: dict[str, Any]) -> MarketTarget:
    data = dict(payload)
    data.pop("schema_version", None)
    for name in ("event_time", "available_time"):
        data[name] = datetime.fromisoformat(data[name])
    return MarketTarget(**data)


def _forecast(payload: dict[str, Any]) -> ContinualForecast:
    data = dict(payload)
    for name in (
        "prediction_sha256",
        "schema_version",
        "research_only",
        "market_evidence",
        "live_pnl_claim",
    ):
        data.pop(name, None)
    for name in ("decision_time", "target_event_time"):
        data[name] = datetime.fromisoformat(data[name])
    result = ContinualForecast(**data)
    if result.prediction_sha256 != payload["prediction_sha256"]:
        raise ValueError("stored prequential forecast changed")
    return result


def _frozen(payload: dict[str, Any]) -> FrozenContinualGaussian:
    data = dict(payload)
    digest = data.pop("model_sha256")
    data.pop("schema_version", None)
    data["fit_as_of"] = datetime.fromisoformat(data["fit_as_of"])
    result = FrozenContinualGaussian(**data)
    if result.model_sha256 != digest:
        raise ValueError("frozen baseline identity changed")
    return result


def _scales(state: dict[str, Any], config: ContinualConfig) -> Array:
    stats = state["preprocessing"]
    return np.asarray(
        np.maximum(
            np.sqrt(np.maximum(stats["m2"], 0) / max(stats["count"] - 1, 1)),
            config.feature_scale_floor,
        ),
        dtype=np.float64,
    )


def _new_statistics(state: dict[str, Any], x: Array, config: ContinualConfig) -> None:
    stats, model, optimizer = state["preprocessing"], state["parameters"], state["optimizer"]
    old_mean, old_scale = np.asarray(stats["mean"]), _scales(state, config)
    count = int(stats["count"]) + 1
    delta = x - old_mean
    mean = old_mean + delta / count
    stats.update(
        count=count, mean=mean.tolist(), m2=(np.asarray(stats["m2"]) + delta * (x - mean)).tolist()
    )
    new_scale = _scales(state, config)
    old_weights, velocity = np.asarray(model["coef"]), np.asarray(optimizer["velocity"])
    offset, ratio = (mean - old_mean) / old_scale, new_scale / old_scale
    model["coef"] = (old_weights * ratio).tolist()
    model["intercept"] += float(old_weights @ offset)
    velocity[-2] += float(velocity[:-2] @ offset)
    velocity[:-2] *= ratio
    optimizer["velocity"] = velocity.tolist()


def _optimize(
    state: dict[str, Any], rows: list[dict[str, Any]], config: ContinualConfig, rate: float
) -> None:
    x = np.asarray([row["observation"]["features"] for row in rows], dtype=float)
    y = np.asarray([row["target"]["value"] for row in rows], dtype=float)
    x = (x - state["preprocessing"]["mean"]) / _scales(state, config)
    model, optimizer = state["parameters"], state["optimizer"]
    weights, velocity = np.asarray(model["coef"]), np.asarray(optimizer["velocity"])
    intercept, log_variance = float(model["intercept"]), float(model["log_variance"])
    for _ in range(config.updates_per_label):
        residual = x @ weights + intercept - y
        variance = math.exp(log_variance)
        grad = np.concatenate(
            (
                np.mean(x * (residual / variance)[:, None], axis=0),
                [np.mean(residual / variance), 0.5 * np.mean(1 - residual**2 / variance)],
            )
        )
        if not np.isfinite(grad).all():
            raise FloatingPointError("non-finite Gaussian training gradients")
        norm = float(np.linalg.norm(grad))
        grad *= min(1.0, config.gradient_clip / max(norm, 1e-15))
        velocity = config.momentum * velocity + grad
        weights -= rate * velocity[:-2]
        intercept -= rate * float(velocity[-2])
        log_variance = float(
            np.clip(
                log_variance - rate * velocity[-1],
                2 * math.log(config.min_scale),
                2 * math.log(config.max_scale),
            )
        )
        optimizer["steps"] += 1
    if not np.isfinite(weights).all() or not math.isfinite(intercept):
        raise FloatingPointError("non-finite learned Gaussian parameters")
    model.update(coef=weights.tolist(), intercept=intercept, log_variance=log_variance)
    optimizer["velocity"] = velocity.tolist()


def _batch_baseline(
    state: dict[str, Any], config: ContinualConfig, as_of: datetime
) -> FrozenContinualGaussian:
    rows = state["initial_prefix"]
    x = np.asarray([row["observation"]["features"] for row in rows], dtype=float)
    y = np.asarray([row["target"]["value"] for row in rows], dtype=float)
    mean, scale = np.asarray(state["preprocessing"]["mean"]), _scales(state, config)
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    penalty = np.eye(design.shape[1]) * config.batch_ridge
    penalty[-1, -1] = 0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    sigma = float(
        np.clip(np.sqrt(np.mean((y - design @ coef) ** 2)), config.min_scale, config.max_scale)
    )
    return FrozenContinualGaussian(
        tuple(state["feature_names"]),
        tuple(mean),
        tuple(scale),
        tuple(coef[:-1]),
        float(coef[-1]),
        sigma,
        as_of,
        state["training_chain_sha256"],
        bool(state["training_synthetic"]),
    )


class ContinualMarket:
    """Single chronological delivery stream, with independent target clocks."""

    def __init__(
        self, feature_names: tuple[str, ...], config: ContinualConfig | None = None
    ) -> None:
        self.feature_names, self.config = _names(feature_names), config or ContinualConfig()
        n = len(self.feature_names)
        self._state: dict[str, Any] = {
            "schema_version": _SCHEMA,
            "implementation_sha256": _IMPLEMENTATION_SHA256,
            "numpy_version": np.__version__,
            "config": asdict(self.config),
            "feature_names": self.feature_names,
            "clock": None,
            "fit_as_of": None,
            "unique_training_count": 0,
            "seen": [],
            "delivered": [],
            "pending": {},
            "reservoir": [],
            "initial_prefix": [],
            "baseline": None,
            "preprocessing": {"count": 0, "mean": [0.0] * n, "m2": [0.0] * n},
            "parameters": {
                "coef": [0.0] * n,
                "intercept": 0.0,
                "log_variance": 2 * math.log(self.config.initial_scale),
            },
            "optimizer": {"velocity": [0.0] * (n + 2), "steps": 0},
            "rng_state": np.random.default_rng(self.config.seed).bit_generator.state,
            "losses": [],
            "last_drift": {"alert": False, "known_losses": 0, "recent_minus_previous_nll": None},
            "training_chain_sha256": _hash([]),
            "event_chain_sha256": _hash([]),
            "training_synthetic": False,
        }
        self._commit(self._state)

    def _commit(self, state: dict[str, Any]) -> None:
        image = _json(state)
        if len(image.encode()) > _MAX_SNAPSHOT_BYTES:
            raise ValueError("state exceeds snapshot resource budget")
        self._state = json.loads(image)
        self._sealed_sha256 = _hash(self._state)

    def _ensure(self) -> None:
        if (
            _hash(self._state) != self._sealed_sha256
            or self._state["config"] != asdict(self.config)
            or tuple(self._state["feature_names"]) != self.feature_names
        ):
            raise RuntimeError(
                "model/config state changed outside a validated update; rollback required"
            )

    def _now(self, value: datetime) -> datetime:
        self._ensure()
        now = _clock(value, "processing clock")
        if self._state["clock"] is not None and now < datetime.fromisoformat(self._state["clock"]):
            raise ValueError("processing clocks cannot move backwards outside rollback")
        return now

    @property
    def n_training(self) -> int:
        self._ensure()
        return int(self._state["unique_training_count"])

    @property
    def optimizer_steps(self) -> int:
        self._ensure()
        return int(self._state["optimizer"]["steps"])

    @property
    def reservoir_ids(self) -> tuple[str, ...]:
        self._ensure()
        return tuple(row["observation"]["record_id"] for row in self._state["reservoir"])

    @property
    def drift_status(self) -> dict[str, Any]:
        self._ensure()
        return dict(self._state["last_drift"])

    def snapshot(self) -> ContinualSnapshot:
        self._ensure()
        return ContinualSnapshot(_json(self._state), self._sealed_sha256)

    def freeze(self) -> FrozenContinualGaussian:
        self._ensure()
        if self.n_training < self.config.min_labels:
            raise RuntimeError("cold start: insufficient published training labels")
        model = self._state["parameters"]
        return FrozenContinualGaussian(
            self.feature_names,
            tuple(self._state["preprocessing"]["mean"]),
            tuple(_scales(self._state, self.config)),
            tuple(model["coef"]),
            float(model["intercept"]),
            math.exp(0.5 * float(model["log_variance"])),
            datetime.fromisoformat(self._state["fit_as_of"]),
            self._state["training_chain_sha256"],
            bool(self._state["training_synthetic"]),
        )

    def predict(
        self, observation: MarketObservation, *, decision_time: datetime
    ) -> ContinualForecast:
        now = self._now(decision_time)
        _decision(observation, now, self.feature_names)
        if observation.record_id in self._state["seen"]:
            raise ValueError("record already predicted; original prequential forecast is immutable")
        if (
            len(self._state["seen"]) >= self.config.max_observations
            or len(self._state["pending"]) >= self.config.max_pending
        ):
            raise ValueError("observation/pending resource bound exceeded")
        if self.n_training < self.config.min_labels:
            forecast = ContinualForecast(
                observation.record_id,
                now,
                observation.target_event_time,
                "cold_start",
                None,
                None,
                None,
                _hash(asdict(observation)),
                self._sealed_sha256,
                observation.synthetic or bool(self._state["training_synthetic"]),
            )
        else:
            forecast = replace(
                self.freeze().predict(observation, decision_time=now),
                state_sha256=self._sealed_sha256,
            )
        state = json.loads(_json(self._state))
        state["pending"][observation.record_id] = {
            "observation": asdict(observation),
            "forecast": asdict(forecast),
        }
        state["seen"].append(observation.record_id)
        state["clock"] = now.isoformat()
        state["event_chain_sha256"] = _hash(
            [state["event_chain_sha256"], "prediction", asdict(forecast)]
        )
        self._commit(state)
        return forecast

    def observe(self, target: MarketTarget, *, as_of: datetime) -> PrequentialUpdate:
        now = self._now(as_of)
        if target.available_time > now:
            raise ValueError("target is not published by processing clock")
        if target.record_id not in self._state["pending"]:
            raise ValueError("an original unconsumed prequential prediction is required")
        pending = self._state["pending"][target.record_id]
        observation, forecast = _observation(pending["observation"]), _forecast(pending["forecast"])
        if (
            target.event_time != observation.target_event_time
            or target.event_time <= forecast.decision_time
        ):
            raise ValueError("target endpoint must match the original future target")
        before = self._sealed_sha256
        scores = forecast.score(target.value) if forecast.status == "forecast" else None
        state = json.loads(_json(self._state))
        if scores is not None:
            state["losses"] = (state["losses"] + [scores.gaussian_nll])[
                -2 * self.config.drift_window :
            ]
        losses, window = state["losses"], self.config.drift_window
        gap = (
            float(np.mean(losses[-window:]) - np.mean(losses[-2 * window : -window]))
            if len(losses) >= 2 * window
            else None
        )
        alert = gap is not None and gap > self.config.drift_threshold
        state["last_drift"] = {
            "alert": alert,
            "known_losses": len(losses),
            "recent_minus_previous_nll": gap,
            "basis": "published targets scored against stored pre-label forecasts",
        }
        rate = self.config.learning_rate * (
            self.config.drift_learning_rate_multiplier if alert else 1.0
        )
        row = {"observation": asdict(observation), "target": asdict(target)}
        _new_statistics(state, np.asarray(observation.features), self.config)
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        replay_count = min(len(state["reservoir"]), self.config.replay_batch_size)
        indices = np.asarray(
            rng.choice(len(state["reservoir"]), size=replay_count, replace=False)
            if replay_count
            else [],
            dtype=np.int64,
        )
        _optimize(state, [row] + [state["reservoir"][int(i)] for i in indices], self.config, rate)
        state["unique_training_count"] += 1
        count = int(state["unique_training_count"])
        if len(state["reservoir"]) < self.config.reservoir_capacity:
            state["reservoir"].append(row)
        else:
            index = int(rng.integers(0, count))
            if index < self.config.reservoir_capacity:
                state["reservoir"][index] = row
        state["rng_state"] = rng.bit_generator.state
        state["training_synthetic"] |= observation.synthetic or target.synthetic
        state["training_chain_sha256"] = _hash([state["training_chain_sha256"], row, now])
        if len(state["initial_prefix"]) < self.config.min_labels:
            state["initial_prefix"].append(row)
        if count == self.config.min_labels:
            state["baseline"] = asdict(_batch_baseline(state, self.config, now))
        state["fit_as_of"] = state["clock"] = now.isoformat()
        del state["pending"][target.record_id]
        state["delivered"].append(target.record_id)
        state["event_chain_sha256"] = _hash(
            [state["event_chain_sha256"], "target_delivery", asdict(target), now]
        )
        self._commit(state)
        return PrequentialUpdate(
            target.record_id,
            forecast,
            scores,
            _hash(asdict(target)),
            now,
            alert,
            rate,
            count,
            self.optimizer_steps,
            before,
            self._sealed_sha256,
            forecast.synthetic or target.synthetic,
        )

    def rollback(self, snapshot: ContinualSnapshot) -> None:
        """Restore all state, including the processing clock and queued forecasts."""
        if (
            snapshot.schema_version != "continual_market_snapshot_v1"
            or len(snapshot.state_json.encode()) > _MAX_SNAPSHOT_BYTES
        ):
            raise ValueError("unsupported or oversized snapshot")
        if _hash(json.loads(snapshot.state_json)) != snapshot.sha256:
            raise ValueError("snapshot digest mismatch")
        state = json.loads(snapshot.state_json)
        if not isinstance(state, dict) or state.get("schema_version") != _SCHEMA:
            raise ValueError("unsupported snapshot schema")
        try:
            config = ContinualConfig(**state["config"])
            names = _names(tuple(state["feature_names"]))
            fresh = ContinualMarket(names, config)
            if set(state) != set(fresh._state):
                raise ValueError("snapshot state fields differ")
            if (
                state["implementation_sha256"] != _IMPLEMENTATION_SHA256
                or state["numpy_version"] != np.__version__
            ):
                raise ValueError(
                    "snapshot implementation/runtime differs; exact replay cannot be assumed"
                )
            count = int(state["unique_training_count"])
            clock = (
                _clock(datetime.fromisoformat(state["clock"]), "clock")
                if state["clock"] is not None
                else None
            )
            fit_clock = (
                _clock(datetime.fromisoformat(state["fit_as_of"]), "fit_as_of")
                if state["fit_as_of"] is not None
                else None
            )
            if (count > 0) != (fit_clock is not None) or (
                fit_clock is not None and (clock is None or fit_clock > clock)
            ):
                raise ValueError("inconsistent snapshot fit/publication clocks")
            if (
                not 0 <= count <= config.max_observations
                or count != state["preprocessing"]["count"]
                or count != len(state["delivered"])
            ):
                raise ValueError("inconsistent snapshot training count")
            if (
                len(state["seen"]) > config.max_observations
                or len(set(state["seen"])) != len(state["seen"])
                or len(set(state["delivered"])) != len(state["delivered"])
                or set(state["pending"]) & set(state["delivered"])
                or set(state["seen"]) != set(state["pending"]) | set(state["delivered"])
            ):
                raise ValueError("inconsistent snapshot identity ledger")
            if (
                len(state["pending"]) > config.max_pending
                or len(state["reservoir"]) > config.reservoir_capacity
                or len(state["initial_prefix"]) != min(count, config.min_labels)
                or len(state["losses"]) > 2 * config.drift_window
            ):
                raise ValueError("snapshot exceeds bounded buffer sizes")
            for key in ("mean", "m2"):
                value = np.asarray(state["preprocessing"][key], dtype=float)
                if value.shape != (len(names),) or not np.isfinite(value).all():
                    raise ValueError("invalid preprocessing snapshot")
                if key == "m2" and np.any(value < -1e-10):
                    raise ValueError("negative Welford variance in snapshot")
            weights, velocity = (
                np.asarray(state["parameters"]["coef"]),
                np.asarray(state["optimizer"]["velocity"]),
            )
            if (
                weights.shape != (len(names),)
                or velocity.shape != (len(names) + 2,)
                or not np.isfinite(weights).all()
                or not np.isfinite(velocity).all()
            ):
                raise ValueError("invalid parameter/optimizer snapshot")
            if (
                not math.isfinite(state["parameters"]["intercept"])
                or not 2 * math.log(config.min_scale)
                <= state["parameters"]["log_variance"]
                <= 2 * math.log(config.max_scale)
                or state["optimizer"]["steps"] != count * config.updates_per_label
            ):
                raise ValueError("invalid snapshot scale/optimizer steps")
            for record_id, item in state["pending"].items():
                observation, forecast = (
                    _observation(item["observation"]),
                    _forecast(item["forecast"]),
                )
                if (
                    observation.feature_names != names
                    or observation.record_id != record_id
                    or forecast.record_id != record_id
                    or forecast.target_event_time != observation.target_event_time
                    or forecast.feature_sha256 != _hash(asdict(observation))
                    or clock is None
                    or forecast.decision_time > clock
                ):
                    raise ValueError("pending observation/forecast identity mismatch")
                _decision(observation, forecast.decision_time, names)
            for row in state["reservoir"] + state["initial_prefix"]:
                observation, target = _observation(row["observation"]), _target(row["target"])
                if (
                    observation.record_id != target.record_id
                    or target.event_time != observation.target_event_time
                    or observation.record_id not in state["delivered"]
                    or observation.feature_names != names
                    or fit_clock is None
                    or target.available_time > fit_clock
                    or observation.available_time >= observation.target_event_time
                ):
                    raise ValueError("invalid published replay record")
            if (count >= config.min_labels) != (state["baseline"] is not None):
                raise ValueError("invalid initial baseline warmup")
            if state["baseline"] is not None:
                baseline = _frozen(state["baseline"])
                if (
                    fit_clock is None
                    or baseline.fit_as_of > fit_clock
                    or any(
                        _target(row["target"]).available_time > baseline.fit_as_of
                        for row in state["initial_prefix"]
                    )
                ):
                    raise ValueError("frozen initial baseline contains unpublished targets")
            for name in ("clock", "fit_as_of"):
                if state[name] is not None:
                    _clock(datetime.fromisoformat(state[name]), name)
            for name in ("training_chain_sha256", "event_chain_sha256"):
                _digest(state[name])
            rng = np.random.default_rng()
            rng.bit_generator.state = state["rng_state"]
            _json(state)
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("malformed snapshot state") from exc
        self.config, self.feature_names = config, names
        self._commit(state)

    def audit_forgetting(
        self, reference: tuple[tuple[MarketObservation, MarketTarget], ...], *, as_of: datetime
    ) -> dict[str, Any]:
        """Score only: never enqueue labels or change optimizer/drift/scaling.

        An external reference is not automatically an independent holdout.
        The receipt reports reuse of already delivered training identifiers.
        """
        now = self._now(as_of)
        current = self.freeze()
        baseline = _frozen(self._state["baseline"])
        if not 2 <= len(reference) <= 4096:
            raise ValueError("bounded reference size >=2 required")
        current_scores, baseline_scores = [], []
        reused, synthetic = 0, bool(self._state["training_synthetic"])
        hashes = []
        for observation, target in reference:
            if (
                observation.record_id != target.record_id
                or observation.target_event_time != target.event_time
                or target.available_time > now
                or observation.available_time > now
                or observation.feature_names != self.feature_names
            ):
                raise ValueError(
                    "reference must have matching already-published feature/target clocks"
                )

            # Retrospective evaluation deliberately bypasses forecast clock
            # restrictions, and is labeled as such; it never trains.
            def score(
                model: FrozenContinualGaussian, known_observation: MarketObservation, value: float
            ) -> ProperScores:
                x = (
                    np.asarray(known_observation.features) - model.feature_mean
                ) / model.feature_scale
                mean = float(x @ model.coefficients + model.intercept)
                return _gaussian_scores(mean, model.scale, value)

            current_scores.append(score(current, observation, target.value))
            baseline_scores.append(score(baseline, observation, target.value))
            reused += observation.record_id in self._state["delivered"]
            synthetic |= observation.synthetic or target.synthetic
            hashes.append(_hash([asdict(observation), asdict(target)]))
        report: dict[str, Any] = {
            "status": "retrospective_score_only_forgetting_audit",
            "as_of": now.isoformat(),
            "n_reference": len(reference),
            "n_reused_training_ids": reused,
            "independent_holdout_established": False,
            "baseline": "frozen_ridge_gaussian_from_initial_published_prefix",
            "current_model_sha256": current.model_sha256,
            "baseline_model_sha256": baseline.model_sha256,
            "reference_sha256": _hash(hashes),
            "state_sha256": self._sealed_sha256,
            "current_gaussian_nll": float(np.mean([v.gaussian_nll for v in current_scores])),
            "baseline_gaussian_nll": float(np.mean([v.gaussian_nll for v in baseline_scores])),
            "current_gaussian_crps": float(np.mean([v.gaussian_crps for v in current_scores])),
            "baseline_gaussian_crps": float(np.mean([v.gaussian_crps for v in baseline_scores])),
            "synthetic": synthetic,
            "research_only": True,
            "market_evidence": False,
            "live_pnl_claim": False,
        }
        report["current_minus_baseline_nll"] = (
            report["current_gaussian_nll"] - report["baseline_gaussian_nll"]
        )
        report["current_minus_baseline_crps"] = (
            report["current_gaussian_crps"] - report["baseline_gaussian_crps"]
        )
        return report
