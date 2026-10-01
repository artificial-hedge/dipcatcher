"""Offline sample-weighted FedAvg for a declared binary numeric event task.

Clients run actual minibatch logistic SGD, following the local optimization
and sample weighting in McMahan et al. (2017), Algorithm1:
https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf . This convex
numeric pilot is not a reproduction of that paper's deep-network results.

Each client scales only its published training prefix. Before aggregation
it transports parameters to raw coordinates: w_raw=w_local/scale and
b_raw=b_local-w_raw·mean. Thus w_local·((x-mean)/scale)+b_local equals
w_raw·x+b_raw. Local gradients retain different preconditioning; FedAvg
is not identical to centralized SGD with pooled normalization.

The aggregator accepts parameters/counts/clock commitments, never raw rows.
All participants share one Python process in this simulation. Parameter
updates, counts and hashes can leak information; there is no authentication,
encryption, secure aggregation, DP, membership protection, network or real
partner integration. Client metadata is attested, not independently verified.
All outputs are research-only, never privacy or market-readiness evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

Array = NDArray[np.float64]
_SCHEMA = "federated_market_binary_v1"
_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_MAX_JSON_BYTES = 16_000_000
_CLOCKS = {
    "event_time",
    "feature_available_time",
    "decision_time",
    "target_event_time",
    "target_available_time",
    "training_as_of",
    "aggregation_as_of",
    "available_as_of",
    "training_cutoff",
    "max_feature_availability",
    "max_target_availability",
    "completed_at",
}


def _json(value: Any) -> str:
    def encode(item: Any) -> Any:
        if isinstance(item, datetime):
            return item.isoformat()
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        raise TypeError("unsupported JSON value")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=encode)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _body(record: Any, digest: str) -> dict[str, Any]:
    return {name: getattr(record, name) for name in record.__dataclass_fields__ if name != digest}


def _seal(record: Any, name: str) -> None:
    object.__setattr__(record, name, _hash(_body(record, name)))


def _ensure(record: Any, name: str) -> None:
    if getattr(record, name) != _hash(_body(record, name)):
        raise ValueError("record parameters/provenance/identity changed")


def _clock(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _text(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("nonempty bounded string required")
    return value


def _sha(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("lowercase SHA256 required")
    return value


def _names(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("explicit feature sequence required")
    result = tuple(value)
    if not 1 <= len(result) <= 64 or len(set(result)) != len(result):
        raise ValueError("one to64 unique feature names required")
    for name in result:
        _text(name)
    return result


def _count(value: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError("integer resource bound exceeded")
    return value


def _finite(value: float, bound: float = 1e6) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
        or abs(value) > bound
    ):
        raise ValueError("finite bounded numeric value required")
    return float(value)


@dataclass(frozen=True)
class BinaryTask:
    task_id: str
    feature_names: tuple[str, ...]
    target_name: str
    positive_label_semantics: str
    task_sha256: str = field(init=False)
    schema_version: str = field(default=_SCHEMA, init=False)

    def __post_init__(self) -> None:
        for name in ("task_id", "target_name", "positive_label_semantics"):
            _text(getattr(self, name))
        object.__setattr__(self, "feature_names", _names(self.feature_names))
        _seal(self, "task_sha256")


@dataclass(frozen=True)
class BinaryExample:
    record_id: str
    asset_id: str
    task_sha256: str
    feature_names: tuple[str, ...]
    features: tuple[float, ...]
    label: int
    event_time: datetime
    feature_available_time: datetime
    decision_time: datetime
    target_event_time: datetime
    target_available_time: datetime
    data_source: str
    evidence_sha256: str
    split: str
    synthetic: bool
    schema_version: str = field(default=_SCHEMA, init=False)

    def __post_init__(self) -> None:
        for name in ("record_id", "asset_id", "data_source"):
            _text(getattr(self, name))
        _sha(self.task_sha256)
        _sha(self.evidence_sha256)
        object.__setattr__(self, "feature_names", _names(self.feature_names))
        values = tuple(_finite(v) for v in self.features)
        object.__setattr__(self, "features", values)
        if (
            len(values) != len(self.feature_names)
            or isinstance(self.label, bool)
            or self.label not in (0, 1)
            or not isinstance(self.label, int)
        ):
            raise ValueError("aligned feature axes and binary integer label required")
        if self.split not in ("train", "holdout") or type(self.synthetic) is not bool:
            raise ValueError("explicit split and synthetic flag required")
        for name in (
            "event_time",
            "feature_available_time",
            "decision_time",
            "target_event_time",
            "target_available_time",
        ):
            object.__setattr__(self, name, _clock(getattr(self, name), name))
        if (
            not self.event_time
            <= self.feature_available_time
            <= self.decision_time
            < self.target_event_time
            <= self.target_available_time
        ):
            raise ValueError("feature/event/decision/target publication clocks violate causality")

    def feature_payload(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"label", "target_available_time", "split"}
        }


@dataclass(frozen=True)
class ClientConfig:
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.1
    l2: float = 0.001
    scale_floor: float = 0.001
    gradient_clip: float = 10.0
    seed: int = 7
    max_examples: int = 20_000
    max_optimizer_steps: int = 20_000
    config_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name, low, high in (
            ("epochs", 1, 100),
            ("batch_size", 1, 256),
            ("seed", 0, 2**63 - 1),
            ("max_examples", 4, 20_000),
            ("max_optimizer_steps", 1, 20_000),
        ):
            _count(getattr(self, name), low, high)
        for name in ("learning_rate", "l2", "scale_floor", "gradient_clip"):
            _finite(getattr(self, name))
        if (
            not 0 < self.learning_rate <= 1
            or not 0 <= self.l2 <= 10
            or self.scale_floor <= 0
            or self.gradient_clip <= 0
        ):
            raise ValueError("valid bounded optimizer/scaling settings required")
        _seal(self, "config_sha256")


@dataclass(frozen=True)
class BinaryPrediction:
    record_id: str
    probability: float
    decision_time: datetime
    target_event_time: datetime
    model_sha256: str
    feature_sha256: str
    synthetic: bool
    prediction_sha256: str = field(init=False)
    research_only: bool = field(default=True, init=False)
    market_evidence: bool = field(default=False, init=False)
    privacy_guarantee: bool = field(default=False, init=False)
    schema_version: str = field(default=_SCHEMA, init=False)

    def __post_init__(self) -> None:
        _text(self.record_id)
        _finite(self.probability)
        _sha(self.model_sha256)
        _sha(self.feature_sha256)
        if not 0 < self.probability < 1 or type(self.synthetic) is not bool:
            raise ValueError("bounded logistic probability and explicit provenance required")
        for name in ("decision_time", "target_event_time"):
            object.__setattr__(self, name, _clock(getattr(self, name), name))
        if self.decision_time >= self.target_event_time:
            raise ValueError("prediction must precede target")
        _seal(self, "prediction_sha256")


@dataclass(frozen=True)
class FederatedModel:
    task: BinaryTask
    coefficients: tuple[float, ...]
    intercept: float
    training_cutoff: datetime
    available_as_of: datetime
    round_id: int
    cumulative_optimizer_steps: int
    training_source_sha256: str
    training_synthetic: bool
    model_sha256: str = field(init=False)
    research_only: bool = field(default=True, init=False)
    market_evidence: bool = field(default=False, init=False)
    privacy_guarantee: bool = field(default=False, init=False)
    schema_version: str = field(default=_SCHEMA, init=False)
    implementation_sha256: str = field(default=_CODE_SHA256, init=False)

    def __post_init__(self) -> None:
        _ensure(self.task, "task_sha256")
        values = tuple(_finite(v) for v in self.coefficients)
        if len(values) != len(self.task.feature_names) or type(self.training_synthetic) is not bool:
            raise ValueError("model feature axes and explicit provenance required")
        object.__setattr__(self, "coefficients", values)
        object.__setattr__(self, "intercept", _finite(self.intercept))
        for name in ("training_cutoff", "available_as_of"):
            object.__setattr__(self, name, _clock(getattr(self, name), name))
        if self.training_cutoff > self.available_as_of:
            raise ValueError("model cannot be available before training cutoff")
        _count(self.round_id, 0, 100)
        _count(self.cumulative_optimizer_steps, 0, 128_000_000)
        _sha(self.training_source_sha256)
        _seal(self, "model_sha256")

    def predict(self, row: BinaryExample) -> BinaryPrediction:
        _ensure(self, "model_sha256")
        _ensure(self.task, "task_sha256")
        if row.task_sha256 != self.task.task_sha256 or row.feature_names != self.task.feature_names:
            raise ValueError("prediction task/feature identity mismatch")
        if row.decision_time < self.available_as_of:
            raise ValueError("model not available at original decision")
        value = float(expit(np.asarray(row.features) @ self.coefficients + self.intercept))
        return BinaryPrediction(
            row.record_id,
            float(np.clip(value, 1e-12, 1 - 1e-12)),
            row.decision_time,
            row.target_event_time,
            self.model_sha256,
            _hash(row.feature_payload()),
            self.training_synthetic or row.synthetic,
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FederatedModel:
        """Restore a frozen predictor, checking all fields and claim identities."""
        return _load_model(payload)


@dataclass(frozen=True)
class RoundSpec:
    round_id: int
    task_sha256: str
    feature_names: tuple[str, ...]
    base_model_sha256: str
    client_ids: tuple[str, ...]
    training_as_of: datetime
    aggregation_as_of: datetime
    client_config: ClientConfig
    round_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _count(self.round_id, 1, 100)
        _sha(self.task_sha256)
        _sha(self.base_model_sha256)
        object.__setattr__(self, "feature_names", _names(self.feature_names))
        ids = tuple(self.client_ids)
        if not 1 <= len(ids) <= 64 or len(set(ids)) != len(ids):
            raise ValueError("bounded unique selected client IDs required")
        for value in ids:
            _text(value)
        object.__setattr__(self, "client_ids", ids)
        for name in ("training_as_of", "aggregation_as_of"):
            object.__setattr__(self, name, _clock(getattr(self, name), name))
        if self.training_as_of > self.aggregation_as_of:
            raise ValueError("aggregation cannot precede training cutoff")
        _ensure(self.client_config, "config_sha256")
        _seal(self, "round_sha256")


@dataclass(frozen=True)
class ClientUpdate:
    client_id: str
    round_id: int
    round_sha256: str
    task_sha256: str
    feature_names: tuple[str, ...]
    base_model_sha256: str
    config_sha256: str
    coefficients: tuple[float, ...]
    intercept: float
    n_examples: int
    optimizer_steps: int
    training_cutoff: datetime
    max_feature_availability: datetime
    max_target_availability: datetime
    completed_at: datetime
    training_source_sha256: str
    training_synthetic: bool
    update_sha256: str = field(init=False)
    research_only: bool = field(default=True, init=False)
    market_evidence: bool = field(default=False, init=False)
    privacy_guarantee: bool = field(default=False, init=False)
    schema_version: str = field(default=_SCHEMA, init=False)
    implementation_sha256: str = field(default=_CODE_SHA256, init=False)

    def __post_init__(self) -> None:
        _text(self.client_id)
        _count(self.round_id, 1, 100)
        for name in (
            "round_sha256",
            "task_sha256",
            "base_model_sha256",
            "config_sha256",
            "training_source_sha256",
        ):
            _sha(getattr(self, name))
        object.__setattr__(self, "feature_names", _names(self.feature_names))
        values = tuple(_finite(v) for v in self.coefficients)
        if len(values) != len(self.feature_names) or type(self.training_synthetic) is not bool:
            raise ValueError("update feature/provenance axes invalid")
        object.__setattr__(self, "coefficients", values)
        object.__setattr__(self, "intercept", _finite(self.intercept))
        _count(self.n_examples, 4, 20_000)
        _count(self.optimizer_steps, 1, 20_000)
        for name in (
            "training_cutoff",
            "max_feature_availability",
            "max_target_availability",
            "completed_at",
        ):
            object.__setattr__(self, name, _clock(getattr(self, name), name))
        if (
            max(self.max_feature_availability, self.max_target_availability) > self.training_cutoff
            or self.completed_at < self.training_cutoff
        ):
            raise ValueError(
                "client update includes unpublished labels/features or impossible completion"
            )
        _seal(self, "update_sha256")


@dataclass(frozen=True)
class ClientTrainingResult:
    update: ClientUpdate
    model: FederatedModel
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    standardized_coefficients: tuple[float, ...]
    standardized_intercept: float
    training_ids: tuple[str, ...]
    epoch_log_losses: tuple[float, ...]
    excluded_unpublished_rows: int
    config: ClientConfig
    comparison_method: str = "client_local_sgd"
    raw_data_pooled: bool = False
    result_sha256: str = field(init=False)
    research_only: bool = field(default=True, init=False)
    market_evidence: bool = field(default=False, init=False)
    privacy_guarantee: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _ensure(self.update, "update_sha256")
        _ensure(self.model, "model_sha256")
        _ensure(self.config, "config_sha256")
        if (
            self.config.config_sha256 != self.update.config_sha256
            or self.model.training_cutoff != self.update.training_cutoff
            or self.model.available_as_of != self.update.completed_at
            or self.model.round_id != self.update.round_id
        ):
            raise ValueError("local result config/model publication identity mismatch")
        if len(self.training_ids) != self.update.n_examples or len(set(self.training_ids)) != len(
            self.training_ids
        ):
            raise ValueError("training identity count mismatch")
        if len(self.epoch_log_losses) != self.config.epochs or not all(
            math.isfinite(v) and v >= 0 for v in self.epoch_log_losses
        ):
            raise ValueError("finite epoch losses required")
        for name in ("feature_mean", "feature_scale", "standardized_coefficients"):
            values = tuple(_finite(v, 1e12) for v in getattr(self, name))
            if len(values) != len(self.update.feature_names):
                raise ValueError("local preprocessing/parameter axes mismatch")
            object.__setattr__(self, name, values)
        raw, bias = transport_to_raw(
            np.asarray(self.standardized_coefficients),
            self.standardized_intercept,
            np.asarray(self.feature_mean),
            np.asarray(self.feature_scale),
        )
        if (
            tuple(raw) != self.update.coefficients
            or bias != self.update.intercept
            or self.model.coefficients != self.update.coefficients
            or self.model.intercept != self.update.intercept
            or self.model.task.task_sha256 != self.update.task_sha256
        ):
            raise ValueError("local/raw transmitted predictor transport mismatch")
        _count(self.excluded_unpublished_rows, 0, 20_000)
        _seal(self, "result_sha256")


def transport_to_raw(
    coefficients: Array, intercept: float, mean: Array, scale: Array
) -> tuple[Array, float]:
    w, mu, s = (
        np.asarray(coefficients, dtype=float),
        np.asarray(mean, dtype=float),
        np.asarray(scale, dtype=float),
    )
    if (
        w.ndim != 1
        or mu.shape != w.shape
        or s.shape != w.shape
        or not np.isfinite(w).all()
        or not np.isfinite(mu).all()
        or not np.isfinite(s).all()
        or np.any(s <= 0)
        or not math.isfinite(intercept)
    ):
        raise ValueError("finite aligned transport parameters and positive scales required")
    raw = w / s
    return raw, float(intercept - raw @ mu)


class FederatedClient:
    """Local in-process simulator; raw examples are never passed to aggregate."""

    def __init__(self, client_id: str, task: BinaryTask, examples: Sequence[BinaryExample]) -> None:
        self.client_id, self.task = _text(client_id), task
        _ensure(task, "task_sha256")
        self._examples = tuple(examples)
        if not 4 <= len(self._examples) <= 20_000 or len(
            {row.record_id for row in self._examples}
        ) != len(self._examples):
            raise ValueError("bounded unique local records required")
        if any(
            row.task_sha256 != task.task_sha256 or row.feature_names != task.feature_names
            for row in self._examples
        ):
            raise ValueError("client task/feature identities differ")
        self._source_seal = _hash([asdict(row) for row in self._examples])
        self._last_result: ClientTrainingResult | None = None
        self.train_histories: tuple[tuple[int, tuple[float, ...]], ...] = ()

    def train(
        self, spec: RoundSpec, base: FederatedModel, *, completed_at: datetime
    ) -> ClientTrainingResult:
        _ensure(spec, "round_sha256")
        _ensure(base, "model_sha256")
        _ensure(self.task, "task_sha256")
        if _hash([asdict(row) for row in self._examples]) != self._source_seal:
            raise ValueError("local input source changed outside a new client dataset")
        completed = _clock(completed_at, "completed_at")
        if (
            self.client_id not in spec.client_ids
            or self.task.task_sha256 != spec.task_sha256
            or base.task.task_sha256 != spec.task_sha256
            or base.model_sha256 != spec.base_model_sha256
            or base.round_id + 1 != spec.round_id
        ):
            raise ValueError("client round/task/base identity mismatch")
        if (
            not max(spec.training_as_of, base.available_as_of)
            <= completed
            <= spec.aggregation_as_of
        ):
            raise ValueError("client completion is outside the frozen round window")
        if (
            self._last_result is not None
            and self._last_result.update.round_sha256 == spec.round_sha256
        ):
            _ensure(self._last_result, "result_sha256")
            if self._last_result.update.completed_at != completed:
                raise ValueError("cached round completion clock differs")
            return self._last_result
        if len(self.train_histories) >= 100:
            raise ValueError("client round-history resource bound exceeded")
        config = spec.client_config
        candidates = [row for row in self._examples if row.split == "train"]
        rows = sorted(
            (row for row in candidates if row.target_available_time <= spec.training_as_of),
            key=lambda row: (row.decision_time, row.record_id),
        )
        _count(len(rows), 4, config.max_examples)
        steps = config.epochs * math.ceil(len(rows) / config.batch_size)
        if steps > config.max_optimizer_steps:
            raise ValueError("optimizer budget exceeded")
        x = np.asarray([row.features for row in rows], dtype=float)
        y = np.asarray([row.label for row in rows], dtype=float)
        mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), config.scale_floor)
        z = (x - mean) / scale
        weights = np.asarray(base.coefficients) * scale
        bias = float(base.intercept + np.asarray(base.coefficients) @ mean)
        seed = int(
            _hash([config.seed, self.client_id, spec.round_id, spec.base_model_sha256])[:16], 16
        )
        rng = np.random.default_rng(seed)
        losses = []
        for _ in range(config.epochs):
            order = rng.permutation(len(rows))
            for start in range(0, len(rows), config.batch_size):
                indices = order[start : start + config.batch_size]
                error = expit(z[indices] @ weights + bias) - y[indices]
                grad = np.concatenate(
                    (
                        z[indices].T @ error / len(indices) + config.l2 * weights,
                        [float(error.mean())],
                    )
                )
                if not np.isfinite(grad).all():
                    raise FloatingPointError("non-finite logistic gradients")
                grad *= min(1.0, config.gradient_clip / max(float(np.linalg.norm(grad)), 1e-15))
                weights -= config.learning_rate * grad[:-1]
                bias -= config.learning_rate * float(grad[-1])
            logits = z @ weights + bias
            loss = float(np.mean(np.logaddexp(0, logits) - y * logits))
            if not math.isfinite(loss):
                raise FloatingPointError("non-finite local log loss")
            losses.append(loss)
        raw, raw_bias = transport_to_raw(weights, bias, mean, scale)
        source = _hash([self.task.task_sha256, [asdict(row) for row in rows]])
        synthetic = any(row.synthetic for row in rows)
        update = ClientUpdate(
            self.client_id,
            spec.round_id,
            spec.round_sha256,
            self.task.task_sha256,
            self.task.feature_names,
            base.model_sha256,
            config.config_sha256,
            tuple(raw),
            raw_bias,
            len(rows),
            steps,
            spec.training_as_of,
            max(row.feature_available_time for row in rows),
            max(row.target_available_time for row in rows),
            completed,
            source,
            synthetic,
        )
        model = FederatedModel(
            self.task,
            tuple(raw),
            raw_bias,
            spec.training_as_of,
            completed,
            spec.round_id,
            base.cumulative_optimizer_steps + steps,
            _hash([base.training_source_sha256, source, update.update_sha256]),
            base.training_synthetic or synthetic,
        )
        result = ClientTrainingResult(
            update,
            model,
            tuple(mean),
            tuple(scale),
            tuple(weights),
            bias,
            tuple(row.record_id for row in rows),
            tuple(losses),
            len(candidates) - len(rows),
            config,
        )
        self._last_result = result
        self.train_histories += ((spec.round_id, result.epoch_log_losses),)
        return result


@dataclass(frozen=True)
class RoundReceipt:
    spec: RoundSpec
    updates: tuple[ClientUpdate, ...]
    sample_counts: tuple[int, ...]
    total_examples: int
    sample_weights: tuple[float, ...]
    base_model_sha256: str
    model: FederatedModel
    previous_receipt_sha256: str
    receipt_sha256: str = field(init=False)
    research_only: bool = field(default=True, init=False)
    market_evidence: bool = field(default=False, init=False)
    privacy_guarantee: bool = field(default=False, init=False)
    client_metadata_independently_verified: bool = field(default=False, init=False)
    schema_version: str = field(default=_SCHEMA, init=False)
    implementation_sha256: str = field(default=_CODE_SHA256, init=False)

    def __post_init__(self) -> None:
        _ensure(self.spec, "round_sha256")
        _ensure(self.model, "model_sha256")
        _sha(self.previous_receipt_sha256)
        if (
            self.sample_counts != tuple(update.n_examples for update in self.updates)
            or self.total_examples != sum(self.sample_counts)
            or self.sample_weights != tuple(n / self.total_examples for n in self.sample_counts)
        ):
            raise ValueError("receipt sample counts/weights mismatch")
        if (
            self.base_model_sha256 != self.spec.base_model_sha256
            or self.model.task.task_sha256 != self.spec.task_sha256
            or self.model.round_id != self.spec.round_id
            or self.model.training_cutoff != self.spec.training_as_of
            or self.model.available_as_of != self.spec.aggregation_as_of
            or tuple(update.client_id for update in self.updates) != self.spec.client_ids
        ):
            raise ValueError("round receipt model/participant identity mismatch")
        for update in self.updates:
            _ensure(update, "update_sha256")
            if (
                update.round_sha256 != self.spec.round_sha256
                or update.base_model_sha256 != self.base_model_sha256
            ):
                raise ValueError("round receipt update base identity mismatch")
        expected = np.asarray(self.sample_weights) @ np.asarray(
            [update.coefficients for update in self.updates]
        )
        expected_bias = float(
            np.asarray(self.sample_weights)
            @ np.asarray([update.intercept for update in self.updates])
        )
        if tuple(expected) != self.model.coefficients or expected_bias != self.model.intercept:
            raise ValueError("round receipt is not the declared sample-weighted average")
        _seal(self, "receipt_sha256")


class FederatedServer:
    """Synchronous all-selected-clients aggregation; no raw-data API."""

    def __init__(self, task: BinaryTask, *, initial_as_of: datetime) -> None:
        clock = _clock(initial_as_of, "initial_as_of")
        self._model = FederatedModel(
            task, (0.0,) * len(task.feature_names), 0.0, clock, clock, 0, 0, _hash([]), False
        )
        self._initial_model = self._model
        self._pending: RoundSpec | None = None
        self._receipts: tuple[RoundReceipt, ...] = ()

    def freeze(self) -> FederatedModel:
        _ensure(self._model, "model_sha256")
        return self._model

    @property
    def receipts(self) -> tuple[RoundReceipt, ...]:
        for receipt in self._receipts:
            _ensure(receipt, "receipt_sha256")
        return self._receipts

    def begin_round(
        self,
        client_ids: tuple[str, ...],
        *,
        training_as_of: datetime,
        aggregation_as_of: datetime,
        config: ClientConfig | None = None,
    ) -> RoundSpec:
        base = self.freeze()
        if self._pending is not None:
            raise ValueError("a round is already pending")
        train, aggregate = (
            _clock(training_as_of, "training_as_of"),
            _clock(aggregation_as_of, "aggregation_as_of"),
        )
        if train < base.training_cutoff or aggregate <= base.available_as_of:
            raise ValueError("round cutoff/publication clocks cannot regress")
        spec = RoundSpec(
            base.round_id + 1,
            base.task.task_sha256,
            base.task.feature_names,
            base.model_sha256,
            client_ids,
            train,
            aggregate,
            config or ClientConfig(),
        )
        self._pending = spec
        return spec

    def aggregate(self, updates: Sequence[ClientUpdate], *, as_of: datetime) -> RoundReceipt:
        base = self.freeze()
        spec = self._pending
        if spec is None:
            raise ValueError("a frozen pending round is required")
        _ensure(spec, "round_sha256")
        now = _clock(as_of, "as_of")
        if now != spec.aggregation_as_of:
            raise ValueError("aggregation clock differs from frozen deadline")
        if len(updates) > 64 or any(not isinstance(value, ClientUpdate) for value in updates):
            raise ValueError("bounded typed parameter updates required")
        incoming = tuple(updates)
        if (
            len(incoming) != len(spec.client_ids)
            or len({v.client_id for v in incoming}) != len(incoming)
            or {v.client_id for v in incoming} != set(spec.client_ids)
        ):
            raise ValueError("exactly one update per selected client required")
        for update in incoming:
            if not isinstance(update, ClientUpdate):
                raise ValueError("typed parameter updates required")
            _ensure(update, "update_sha256")
            if (
                update.round_id != spec.round_id
                or update.round_sha256 != spec.round_sha256
                or update.base_model_sha256 != base.model_sha256
                or update.task_sha256 != spec.task_sha256
                or update.feature_names != spec.feature_names
                or update.config_sha256 != spec.client_config.config_sha256
            ):
                raise ValueError("stale round/base/config or wrong task/feature identity")
            if (
                update.training_cutoff != spec.training_as_of
                or update.completed_at > now
                or update.completed_at < base.available_as_of
                or max(update.max_target_availability, update.max_feature_availability)
                > spec.training_as_of
            ):
                raise ValueError("future or inconsistent client clocks")
            if (
                update.n_examples > spec.client_config.max_examples
                or update.optimizer_steps
                != spec.client_config.epochs
                * math.ceil(update.n_examples / spec.client_config.batch_size)
                or update.optimizer_steps > spec.client_config.max_optimizer_steps
            ):
                raise ValueError("client sample/optimizer budget mismatch")
        by_id = {update.client_id: update for update in incoming}
        ordered = tuple(by_id[name] for name in spec.client_ids)
        counts = tuple(update.n_examples for update in ordered)
        total = sum(counts)
        weights = tuple(n / total for n in counts)
        coef = np.asarray([update.coefficients for update in ordered], dtype=float)
        bias = np.asarray([update.intercept for update in ordered], dtype=float)
        combined = np.asarray(weights) @ coef
        combined_bias = float(np.asarray(weights) @ bias)
        source = _hash(
            [
                base.training_source_sha256,
                spec.round_sha256,
                [update.update_sha256 for update in ordered],
                counts,
                weights,
            ]
        )
        model = FederatedModel(
            base.task,
            tuple(combined),
            combined_bias,
            spec.training_as_of,
            now,
            spec.round_id,
            base.cumulative_optimizer_steps + sum(update.optimizer_steps for update in ordered),
            source,
            base.training_synthetic or any(update.training_synthetic for update in ordered),
        )
        previous = self._receipts[-1].receipt_sha256 if self._receipts else _hash([])
        receipt = RoundReceipt(
            spec, ordered, counts, total, weights, base.model_sha256, model, previous
        )
        # All validation/model/receipt construction precede the sole commit.
        self._model, self._pending, self._receipts = model, None, self._receipts + (receipt,)
        return receipt

    def snapshot_payload(self) -> dict[str, Any]:
        model = self.freeze()
        if self._pending is not None:
            _ensure(self._pending, "round_sha256")
        body = {
            "schema_version": _SCHEMA,
            "implementation_sha256": _CODE_SHA256,
            "numpy_version": np.__version__,
            "initial_model": asdict(self._initial_model),
            "model": asdict(model),
            "pending": asdict(self._pending) if self._pending else None,
            "receipts": [asdict(receipt) for receipt in self.receipts],
        }
        return {**body, "snapshot_sha256": _hash(body)}

    def save_json(self, path: Path) -> None:
        value = _json(self.snapshot_payload()).encode()
        if len(value) > _MAX_JSON_BYTES:
            raise ValueError("snapshot exceeds resource bound")
        with path.open("xb") as stream:
            stream.write(value + b"\n")

    @classmethod
    def load_json(cls, path: Path) -> FederatedServer:
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise ValueError("snapshot exceeds resource bound")
        with path.open("rb") as stream:
            raw = stream.read(_MAX_JSON_BYTES + 1)
        if len(raw) > _MAX_JSON_BYTES:
            raise ValueError("snapshot exceeds resource bound")
        payload = json.loads(raw)
        digest = payload.pop("snapshot_sha256")
        if (
            _hash(payload) != digest
            or payload["schema_version"] != _SCHEMA
            or payload["implementation_sha256"] != _CODE_SHA256
            or payload["numpy_version"] != np.__version__
        ):
            raise ValueError("snapshot hash/schema/implementation/runtime mismatch")
        model = _load_model(payload["model"])
        initial = _load_model(payload["initial_model"])
        server = cls(initial.task, initial_as_of=initial.available_as_of)
        if server.freeze().model_sha256 != initial.model_sha256:
            raise ValueError("invalid initial model")
        if len(payload["receipts"]) > 100:
            raise ValueError("receipt-history resource bound exceeded")
        for value in payload["receipts"]:
            spec = _load_spec(value["spec"])
            updates = tuple(_load(ClientUpdate, item) for item in value["updates"])
            replay_spec = server.begin_round(
                spec.client_ids,
                training_as_of=spec.training_as_of,
                aggregation_as_of=spec.aggregation_as_of,
                config=spec.client_config,
            )
            if replay_spec.round_sha256 != spec.round_sha256:
                raise ValueError("round source/base identity mismatch")
            receipt = server.aggregate(updates, as_of=spec.aggregation_as_of)
            if json.loads(_json(asdict(receipt))) != value:
                raise ValueError("round receipt chain mismatch")
        if server.freeze().model_sha256 != model.model_sha256:
            raise ValueError("restored model does not match receipt history")
        pending = _load_spec(payload["pending"]) if payload["pending"] else None
        if pending is not None:
            replay_pending = server.begin_round(
                pending.client_ids,
                training_as_of=pending.training_as_of,
                aggregation_as_of=pending.aggregation_as_of,
                config=pending.client_config,
            )
            if replay_pending.round_sha256 != pending.round_sha256:
                raise ValueError("restored pending round base mismatch")
        return server


def _load[T](cls: type[T], payload: dict[str, Any]) -> T:
    data = dict(payload)
    if not is_dataclass(cls):
        raise ValueError("serialized record requires a dataclass type")
    definitions = fields(cls)
    if set(data) != {definition.name for definition in definitions}:
        raise ValueError("serialized record field mismatch")
    args = {definition.name: data[definition.name] for definition in definitions if definition.init}
    for name in _CLOCKS & set(args):
        args[name] = datetime.fromisoformat(args[name])
    result = cls(**args)
    for definition in definitions:
        if not definition.init and data[definition.name] != getattr(result, definition.name):
            raise ValueError("serialized identity/version/claim flag mismatch")
    return result


def _load_model(payload: dict[str, Any]) -> FederatedModel:
    data = dict(payload)
    data["task"] = _load(BinaryTask, data["task"])
    return _load(FederatedModel, data)


def _load_spec(payload: dict[str, Any]) -> RoundSpec:
    data = dict(payload)
    data["client_config"] = _load(ClientConfig, data["client_config"])
    return _load(RoundSpec, data)


def train_centralized(
    task: BinaryTask,
    examples: Sequence[BinaryExample],
    base: FederatedModel,
    *,
    config: ClientConfig,
    training_as_of: datetime,
    completed_at: datetime,
) -> ClientTrainingResult:
    """Explicit control: caller pools raw rows; this provides no privacy."""
    client = FederatedClient("__centralized__", task, examples)
    spec = RoundSpec(
        base.round_id + 1,
        task.task_sha256,
        task.feature_names,
        base.model_sha256,
        (client.client_id,),
        training_as_of,
        completed_at,
        config,
    )
    result = client.train(spec, base, completed_at=completed_at)
    return replace(result, comparison_method="centralized_pooled_sgd", raw_data_pooled=True)


def evaluate_binary(
    model: FederatedModel,
    examples: Sequence[BinaryExample],
    *,
    as_of: datetime,
    forbidden_training_ids: Sequence[str],
) -> dict[str, Any]:
    """Later disjoint-ID holdout scores; economic independence is unverified."""
    _ensure(model, "model_sha256")
    now = _clock(as_of, "evaluation as_of")
    rows = tuple(examples)
    if not 4 <= len(rows) <= 20_000 or len({row.record_id for row in rows}) != len(rows):
        raise ValueError("bounded unique holdout records required")
    forbidden = set(forbidden_training_ids)
    if any(
        row.split != "holdout"
        or row.record_id in forbidden
        or row.decision_time <= model.training_cutoff
        or row.target_available_time > now
        for row in rows
    ):
        raise ValueError("holdout is reused, not later, or not yet published")
    probabilities = np.asarray([model.predict(row).probability for row in rows])
    labels = np.asarray([row.label for row in rows], dtype=float)
    report = {
        "n": len(rows),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log1p(-probabilities))
        ),
        "model_sha256": model.model_sha256,
        "source_sha256": _hash([asdict(row) for row in rows]),
        "feature_sha256": _hash([row.feature_payload() for row in rows]),
        "evaluation_as_of": now.isoformat(),
        "disjoint_record_ids_checked": True,
        "economic_independence_established": False,
        "synthetic": model.training_synthetic or any(row.synthetic for row in rows),
        "research_only": True,
        "market_evidence": False,
        "privacy_guarantee": False,
    }
    report["receipt_sha256"] = _hash(report)
    return report


def compare_models(
    global_model: FederatedModel,
    locals_: Sequence[ClientTrainingResult],
    centralized: ClientTrainingResult,
    holdout: Sequence[BinaryExample],
    *,
    as_of: datetime,
) -> dict[str, Any]:
    """Offline coordinator sees identifiers/holdouts; it is not the server."""
    local = tuple(locals_)
    if (
        not 1 <= len(local) <= 64
        or len({result.update.client_id for result in local}) != len(local)
        or not centralized.raw_data_pooled
    ):
        raise ValueError("local results and an explicit pooled centralized control required")
    all_ids = [name for result in local for name in result.training_ids]
    if len(set(all_ids)) != len(all_ids) or set(all_ids) != set(centralized.training_ids):
        raise ValueError("client prefixes overlap or differ from pooled control")
    for result in (*local, centralized):
        _ensure(result, "result_sha256")
    models = {
        "global": global_model,
        "centralized": centralized.model,
        **{f"local:{result.update.client_id}": result.model for result in local},
    }
    report = {
        "scores": {
            name: evaluate_binary(model, holdout, as_of=as_of, forbidden_training_ids=all_ids)
            for name, model in models.items()
        },
        "budgets": {
            "global_cumulative_optimizer_steps": global_model.cumulative_optimizer_steps,
            "local_last_round_optimizer_steps": {
                result.update.client_id: result.update.optimizer_steps for result in local
            },
            "centralized_optimizer_steps": centralized.update.optimizer_steps,
            "local_configs": {result.update.client_id: asdict(result.config) for result in local},
            "centralized_config": asdict(centralized.config),
        },
        "compute_budgets_equal": False,
        "raw_training_rows_pooled_for_centralized_control": True,
        "comparison_coordinator_sees_training_ids_and_holdout_rows": True,
        "process_isolation": False,
        "research_only": True,
        "market_evidence": False,
        "privacy_guarantee": False,
        "limitations": [
            "One-process offline simulation; no actual partners, network or enforced client isolation.",
            "Parameter/count/hash updates can leak data; no DP, secure aggregation, authentication or membership-protection guarantee.",
            "Server trusts client sample counts, clocks and source commitments; it cannot verify raw-data rights, labels or cross-client overlap.",
            "Local normalization is transported to raw coordinates, but local conditioning and SGD paths differ from pooled centralized optimization.",
            "Global model inherits previous rounds; local results report only this round's additional steps, so compute comparisons are not matched.",
            "Later disjoint identifiers do not prove economic independence, fresh empirical holdout or market validity.",
        ],
    }
    report["receipt_sha256"] = _hash(report)
    return report
