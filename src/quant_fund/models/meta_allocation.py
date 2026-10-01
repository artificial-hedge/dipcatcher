"""Exact second-order MAML Gaussian forecasting and constrained research weights.

Finn, Abbeel & Levine, Model-Agnostic Meta-Learning (ICML 2017):
https://arxiv.org/abs/1703.03400. The outer Gaussian NLL differentiates through
every support-only SGD step using create_graph=True; this is MAML, not a bandit
or a first-order gradient approximation. This small CPU numeric network is not
a pretrained model, Meta-RL reproduction or financial SOTA claim.

Regime tasks have disjoint IDs, observation keys and chronological spans.
Support labels must be published before query decisions, with a forecast-horizon
purge plus explicit embargo. All meta-training labels are published by fit cutoff.
Numeric and target normalization use each task's support only. Held-out adaptation
accepts support only, ignores unpublished suffixes and never accepts query data.
Pooled/scratch comparisons preserve every outcome under equal held-out support,
step and learning-rate budgets. Pretraining optimizer counts are equal for MAML
and pooled, but MAML inner/second-order work is additional: compute is not matched.

Allocations minimize .5*risk_aversion*w'Cov*w - forecast_mean'w with long-only
caps, sum(w)<=budget and optional variance limit. Unallocated budget is cash.
Absent dated covariance evidence, independent predictive variances form an
explicit covariance PROXY. These are static research weights, not an execution
policy, market evidence or a promise of realized risk or investment performance.
Caller-supplied source hashes/clocks/target units still require independent audit.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from quant_fund.metrics.scoring import crps_gaussian

Array = NDArray[np.float64]
Arm = Literal["maml", "pooled", "scratch"]
TargetKind = Literal["numeric_outcome", "forward_simple_return"]
_NAMES = ("w1", "b1", "w_mean", "b_mean", "w_scale", "b_scale")


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def _sha(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA256")


def _name(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string <=256 characters")


def _clock(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _count(value: int, name: str, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


@dataclass(frozen=True, slots=True)
class FeaturePoint:
    point_id: str
    entity_id: str
    decision_time: datetime
    event_time: datetime
    available_time: datetime
    values: tuple[float, ...]
    source_id: str
    source_sha256: str
    synthetic: bool

    def __post_init__(self) -> None:
        for name in ("point_id", "entity_id", "source_id"):
            _name(getattr(self, name), name)
        event, available, decision = (
            _clock(v, n)
            for v, n in (
                (self.event_time, "feature event"),
                (self.available_time, "feature availability"),
                (self.decision_time, "decision_time"),
            )
        )
        if not event <= available <= decision:
            raise ValueError("feature clocks must satisfy event<=availability<=decision")
        if not isinstance(self.values, tuple) or not 1 <= len(self.values) <= 1024:
            raise ValueError("values must be a nonempty immutable tuple")
        for value in self.values:
            _finite(value, "feature")
        _sha(self.source_sha256, "source_sha256")
        if not isinstance(self.synthetic, bool):
            raise ValueError("synthetic must be boolean")


@dataclass(frozen=True, slots=True)
class LabelledPoint:
    point: FeaturePoint
    target: float
    target_event_time: datetime
    target_available_time: datetime
    label_source_id: str
    label_source_sha256: str
    label_synthetic: bool

    def __post_init__(self) -> None:
        if not isinstance(self.point, FeaturePoint):
            raise ValueError("point must be typed FeaturePoint")
        _finite(self.target, "target")
        event = _clock(self.target_event_time, "target event")
        available = _clock(self.target_available_time, "target availability")
        if event <= _clock(self.point.decision_time, "decision") or available < event:
            raise ValueError("target event must follow decision and precede target availability")
        _name(self.label_source_id, "label_source_id")
        _sha(self.label_source_sha256, "label_source_sha256")
        if not isinstance(self.label_synthetic, bool):
            raise ValueError("label_synthetic must be boolean")


@dataclass(frozen=True, slots=True)
class RegimeTask:
    task_id: str
    regime_id: str
    support: tuple[LabelledPoint, ...]
    query: tuple[LabelledPoint, ...]
    manifest_sha256: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        _name(self.task_id, "task_id")
        _name(self.regime_id, "regime_id")
        for name in ("support", "query"):
            rows = getattr(self, name)
            if (
                not isinstance(rows, tuple)
                or not rows
                or any(not isinstance(r, LabelledPoint) for r in rows)
            ):
                raise ValueError(f"{name} must be a nonempty immutable typed tuple")
        all_rows = self.support + self.query
        if len({r.point.point_id for r in all_rows}) != len(all_rows):
            raise ValueError("support/query point IDs must be disjoint")
        if (
            len(
                {
                    _clock(r.target_event_time, "target")
                    - _clock(r.point.decision_time, "decision")
                    for r in all_rows
                }
            )
            != 1
        ):
            raise ValueError("task targets must use one forecast horizon")
        object.__setattr__(
            self,
            "manifest_sha256",
            _hash(
                {
                    "task_id": self.task_id,
                    "regime_id": self.regime_id,
                    "support": [asdict(r) for r in self.support],
                    "query": [asdict(r) for r in self.query],
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class MAMLConfig:
    input_dim: int
    hidden_dim: int = 8
    inner_steps: int = 2
    outer_steps: int = 50
    inner_lr: float = 0.03
    outer_lr: float = 0.005
    min_sigma: float = 0.1
    horizon: timedelta = timedelta(days=1)
    embargo: timedelta = timedelta(0)
    target_kind: TargetKind = "numeric_outcome"
    seed: int = 42
    max_tasks: int = 32
    max_partition_rows: int = 512
    max_prediction_rows: int = 1024
    max_gradient_evaluations: int = 10000

    def __post_init__(self) -> None:
        for name, low, high in (
            ("input_dim", 1, 1024),
            ("hidden_dim", 1, 128),
            ("inner_steps", 1, 20),
            ("outer_steps", 1, 300),
            ("seed", 0, 2**32 - 1),
            ("max_tasks", 1, 64),
            ("max_partition_rows", 2, 2048),
            ("max_prediction_rows", 1, 10000),
            ("max_gradient_evaluations", 1, 100000),
        ):
            _count(getattr(self, name), name, low, high)
        for name in ("inner_lr", "outer_lr", "min_sigma"):
            value = getattr(self, name)
            _finite(value, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        if not isinstance(self.horizon, timedelta) or not timedelta(0) < self.horizon <= timedelta(
            days=365
        ):
            raise ValueError("horizon must be positive and <=365 days")
        if not isinstance(self.embargo, timedelta) or not timedelta(0) <= self.embargo <= timedelta(
            days=365
        ):
            raise ValueError("embargo must be a nonnegative bounded timedelta")
        if self.target_kind not in ("numeric_outcome", "forward_simple_return"):
            raise ValueError("unsupported target_kind")


def _shapes(config: MAMLConfig) -> tuple[tuple[int, ...], ...]:
    return (
        (config.input_dim, config.hidden_dim),
        (config.hidden_dim,),
        (config.hidden_dim,),
        (1,),
        (config.hidden_dim,),
        (1,),
    )


@dataclass(frozen=True, slots=True)
class ParameterState:
    shapes: tuple[tuple[int, ...], ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.shapes, tuple)
            or not isinstance(self.values, tuple)
            or len(self.shapes) != len(_NAMES)
            or len(self.values) != len(_NAMES)
        ):
            raise ValueError("parameter schema mismatch")
        for shape, values in zip(self.shapes, self.values, strict=True):
            if (
                not isinstance(shape, tuple)
                or not shape
                or any(
                    isinstance(x, bool) or not isinstance(x, int) or not 0 < x <= 1024
                    for x in shape
                )
            ):
                raise ValueError("invalid parameter shape")
            if not isinstance(values, tuple) or math.prod(shape) != len(values):
                raise ValueError("parameter values must match immutable shapes")
            for value in values:
                _finite(value, "parameter")

    @property
    def sha256(self) -> str:
        return _hash(asdict(self))

    def arrays(self) -> dict[str, Array]:
        return {
            name: np.array(values, dtype=float).reshape(shape)
            for name, shape, values in zip(_NAMES, self.shapes, self.values, strict=True)
        }


@dataclass(frozen=True, slots=True)
class SupportPreprocessing:
    x_mean: tuple[float, ...]
    x_scale: tuple[float, ...]
    y_mean: float
    y_scale: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.x_mean, tuple)
            or not isinstance(self.x_scale, tuple)
            or not self.x_mean
            or len(self.x_mean) != len(self.x_scale)
        ):
            raise ValueError("support preprocessing must use matching immutable tuples")
        for value in (*self.x_mean, *self.x_scale, self.y_mean, self.y_scale):
            _finite(value, "preprocessing")
        if any(value <= 0 for value in self.x_scale) or self.y_scale <= 0:
            raise ValueError("preprocessing scales must be positive")

    @property
    def sha256(self) -> str:
        return _hash(asdict(self))


def _preprocess(rows: tuple[LabelledPoint, ...]) -> SupportPreprocessing:
    x = np.array([r.point.values for r in rows], dtype=float)
    y = np.array([r.target for r in rows], dtype=float)
    sd = x.std(axis=0)
    ysd = float(y.std())
    return SupportPreprocessing(
        tuple(float(v) for v in x.mean(axis=0)),
        tuple(float(v) for v in np.where(sd < 1e-8, 1.0, sd)),
        float(y.mean()),
        ysd if ysd >= 1e-8 else 1.0,
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("MAML requires the optional nn extra (torch)") from exc
    return torch


def _initial(config: MAMLConfig, torch: Any) -> dict[str, Any]:
    # A local CPU generator avoids touching either the caller's CPU or accelerator RNG.
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    result = {}
    for name, shape in zip(_NAMES, _shapes(config), strict=True):
        if name == "w1":
            value = torch.randn(
                shape, dtype=torch.float64, device="cpu", generator=generator
            ) / math.sqrt(config.input_dim)
        elif name.startswith("w_"):
            value = (
                torch.randn(shape, dtype=torch.float64, device="cpu", generator=generator) * 0.05
            )
        else:
            value = torch.zeros(shape, dtype=torch.float64, device="cpu")
        result[name] = value.requires_grad_(True)
    return result


def _state(parameters: dict[str, Any]) -> ParameterState:
    return ParameterState(
        tuple(tuple(parameters[n].shape) for n in _NAMES),
        tuple(
            tuple(float(v) for v in parameters[n].detach().cpu().flatten().tolist()) for n in _NAMES
        ),
    )


def _tensors(
    state: ParameterState, config: MAMLConfig, torch: Any, *, grad: bool
) -> dict[str, Any]:
    if state.shapes != _shapes(config):
        raise ValueError("parameter shapes do not match configuration")
    return {
        name: torch.tensor(array, dtype=torch.float64, device="cpu", requires_grad=grad)
        for name, array in state.arrays().items()
    }


def _forward(parameters: dict[str, Any], x: Any, config: MAMLConfig, torch: Any) -> tuple[Any, Any]:
    hidden = torch.tanh(x @ parameters["w1"] + parameters["b1"])
    mean = hidden @ parameters["w_mean"] + parameters["b_mean"][0]
    sigma = (
        torch.nn.functional.softplus(hidden @ parameters["w_scale"] + parameters["b_scale"][0])
        + config.min_sigma
    )
    return mean, sigma


def _batch(
    rows: tuple[LabelledPoint, ...], preprocessing: SupportPreprocessing, torch: Any
) -> tuple[Any, Any]:
    x = (np.array([r.point.values for r in rows]) - preprocessing.x_mean) / preprocessing.x_scale
    y = (np.array([r.target for r in rows]) - preprocessing.y_mean) / preprocessing.y_scale
    return (
        torch.tensor(x, dtype=torch.float64, device="cpu"),
        torch.tensor(y, dtype=torch.float64, device="cpu"),
    )


def _nll(
    parameters: dict[str, Any],
    batch: tuple[Any, Any],
    preprocessing: SupportPreprocessing,
    config: MAMLConfig,
    torch: Any,
) -> Any:
    mean, sigma = _forward(parameters, batch[0], config, torch)
    return (
        torch.log(sigma)
        + 0.5 * ((batch[1] - mean) / sigma) ** 2
        + 0.5 * math.log(2 * math.pi)
        + math.log(preprocessing.y_scale)
    ).mean()


def _inner(
    parameters: dict[str, Any],
    batch: tuple[Any, Any],
    preprocessing: SupportPreprocessing,
    config: MAMLConfig,
    torch: Any,
    steps: int,
    *,
    second_order: bool,
) -> dict[str, Any]:
    current = parameters
    for _ in range(steps):
        loss = _nll(current, batch, preprocessing, config, torch)
        gradients = torch.autograd.grad(loss, tuple(current.values()), create_graph=second_order)
        current = {
            name: value - config.inner_lr * gradient
            for (name, value), gradient in zip(current.items(), gradients, strict=True)
        }
    return current


def _validate_rows(
    rows: tuple[LabelledPoint, ...], config: MAMLConfig, *, cutoff: datetime | None
) -> None:
    if not isinstance(rows, tuple) or not 1 <= len(rows) <= config.max_partition_rows:
        raise ValueError("partition must be a bounded immutable nonempty tuple")
    for row in rows:
        if not isinstance(row, LabelledPoint) or len(row.point.values) != config.input_dim:
            raise ValueError("typed labelled rows with matching input_dim required")
        if (
            _clock(row.target_event_time, "target") - _clock(row.point.decision_time, "decision")
            != config.horizon
        ):
            raise ValueError("label forecast horizon mismatch")
        if cutoff is not None and _clock(row.target_available_time, "label availability") > cutoff:
            raise ValueError("target unpublished at fit cutoff")
        if config.target_kind == "forward_simple_return" and row.target < -1:
            raise ValueError("simple-return target cannot be less than -1")
    keys = {
        (r.point.source_id, r.point.entity_id, _clock(r.point.decision_time, "decision"))
        for r in rows
    }
    if len(keys) != len(rows) or len({r.point.point_id for r in rows}) != len(rows):
        raise ValueError("duplicate support observation key or point_id")


def _validate_tasks(
    tasks: tuple[RegimeTask, ...], config: MAMLConfig, fit_cutoff: datetime
) -> None:
    if (
        not isinstance(tasks, tuple)
        or not 1 <= len(tasks) <= config.max_tasks
        or any(not isinstance(t, RegimeTask) for t in tasks)
    ):
        raise ValueError("tasks must be a bounded immutable typed tuple")
    if len({t.task_id for t in tasks}) != len(tasks) or len({t.regime_id for t in tasks}) != len(
        tasks
    ):
        raise ValueError("regime/task IDs must be disjoint")
    keys, point_ids, spans = set(), set(), []
    for task in tasks:
        _validate_rows(task.support, config, cutoff=fit_cutoff)
        _validate_rows(task.query, config, cutoff=fit_cutoff)
        if len(task.support) < 2:
            raise ValueError("task support requires at least two rows")
        support_end = max(_clock(r.target_available_time, "support label") for r in task.support)
        query_start = min(_clock(r.point.decision_time, "query decision") for r in task.query)
        if query_start < support_end + config.embargo:
            raise ValueError("query violates support-label horizon purge/embargo")
        for row in task.support + task.query:
            key = (
                row.point.source_id,
                row.point.entity_id,
                _clock(row.point.decision_time, "decision"),
            )
            if key in keys or row.point.point_id in point_ids:
                raise ValueError("tasks reuse an observation key or point_id")
            keys.add(key)
            point_ids.add(row.point.point_id)
        spans.append(
            (
                min(_clock(r.point.decision_time, "decision") for r in task.support),
                max(_clock(r.target_available_time, "label") for r in task.query),
            )
        )
    spans.sort()
    if any(a[1] + config.embargo > b[0] for a, b in zip(spans, spans[1:], strict=False)):
        raise ValueError("regime task chronological spans overlap")


@dataclass(frozen=True, slots=True)
class GradientAudit:
    objective_nll: float
    parameters: ParameterState
    gradients: ParameterState
    data_sha256: str
    second_order: bool = dataclass_field(init=False, default=True)


def _meta_objective(
    parameters: dict[str, Any], tasks: tuple[RegimeTask, ...], config: MAMLConfig, torch: Any
) -> Any:
    losses = []
    for task in tasks:
        preprocessing = _preprocess(task.support)
        adapted = _inner(
            parameters,
            _batch(task.support, preprocessing, torch),
            preprocessing,
            config,
            torch,
            config.inner_steps,
            second_order=True,
        )
        losses.append(
            _nll(adapted, _batch(task.query, preprocessing, torch), preprocessing, config, torch)
        )
    return torch.stack(losses).mean()


@dataclass(frozen=True, slots=True)
class GaussianPrediction:
    points: tuple[FeaturePoint, ...]
    mean: tuple[float, ...]
    sigma: tuple[float, ...]
    target_kind: TargetKind
    horizon: timedelta
    adapted_sha256: str
    model_synthetic: bool
    forecast_sha256: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.points, tuple) or any(
            not isinstance(p, FeaturePoint) for p in self.points
        ):
            raise ValueError("forecast points must be an immutable typed tuple")
        if not isinstance(self.mean, tuple) or not isinstance(self.sigma, tuple):
            raise ValueError("forecast values must be immutable tuples")
        if (
            not self.points
            or len(self.mean) != len(self.points)
            or len(self.sigma) != len(self.points)
        ):
            raise ValueError("forecast shape mismatch")
        for mean, sigma in zip(self.mean, self.sigma, strict=True):
            _finite(mean, "forecast mean")
            _finite(sigma, "forecast sigma")
            if sigma <= 0:
                raise ValueError("forecast sigma must be positive")
        _sha(self.adapted_sha256, "adapted_sha256")
        if not isinstance(self.model_synthetic, bool):
            raise ValueError("model_synthetic must be boolean")
        if (
            self.target_kind not in ("numeric_outcome", "forward_simple_return")
            or not isinstance(self.horizon, timedelta)
            or self.horizon <= timedelta(0)
        ):
            raise ValueError("forecast target kind/horizon must be explicit")
        object.__setattr__(
            self,
            "forecast_sha256",
            _hash(
                {
                    "points": [asdict(p) for p in self.points],
                    "mean": self.mean,
                    "sigma": self.sigma,
                    "target_kind": self.target_kind,
                    "horizon": self.horizon,
                    "adapted_sha256": self.adapted_sha256,
                    "model_synthetic": self.model_synthetic,
                }
            ),
        )

    @property
    def synthetic(self) -> bool:
        return self.model_synthetic or any(p.synthetic for p in self.points)


@dataclass(frozen=True, slots=True)
class AdaptedGaussianModel:
    task_id: str
    arm: Arm
    config: MAMLConfig
    parameters: ParameterState
    preprocessing: SupportPreprocessing
    adaptation_cutoff: datetime
    support_sha256: str
    parent_sha256: str
    support_rows: int
    steps: int
    synthetic: bool
    adapted_sha256: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        _name(self.task_id, "adapted task_id")
        if self.arm not in ("maml", "pooled", "scratch") or self.parameters.shapes != _shapes(
            self.config
        ):
            raise ValueError("adapted initialization/schema mismatch")
        if len(self.preprocessing.x_mean) != self.config.input_dim:
            raise ValueError("adapted preprocessing dimension mismatch")
        _count(self.support_rows, "support_rows", 2, self.config.max_partition_rows)
        _count(self.steps, "adaptation steps", 0, self.config.inner_steps)
        _clock(self.adaptation_cutoff, "adaptation_cutoff")
        _sha(self.support_sha256, "support_sha256")
        _sha(self.parent_sha256, "parent_sha256")
        if not isinstance(self.synthetic, bool):
            raise ValueError("adapted synthetic identity must be boolean")
        object.__setattr__(
            self,
            "adapted_sha256",
            _hash(
                {
                    "task_id": self.task_id,
                    "arm": self.arm,
                    "config": asdict(self.config),
                    "parameters": asdict(self.parameters),
                    "preprocessing": asdict(self.preprocessing),
                    "adaptation_cutoff": self.adaptation_cutoff,
                    "support_sha256": self.support_sha256,
                    "parent_sha256": self.parent_sha256,
                    "support_rows": self.support_rows,
                    "steps": self.steps,
                    "synthetic": self.synthetic,
                }
            ),
        )

    def predict(self, points: tuple[FeaturePoint, ...], *, asof: datetime) -> GaussianPrediction:
        asof = _clock(asof, "prediction asof")
        if not isinstance(points, tuple) or not 1 <= len(points) <= self.config.max_prediction_rows:
            raise ValueError("points must be a bounded immutable tuple")
        if any(
            not isinstance(p, FeaturePoint) or len(p.values) != self.config.input_dim
            for p in points
        ):
            raise ValueError("typed points with matching input_dim required")
        visible = tuple(p for p in points if _clock(p.decision_time, "decision") <= asof)
        if not visible:
            raise ValueError("no published prediction points at asof")
        if any(
            _clock(p.decision_time, "decision") < self.adaptation_cutoff + self.config.embargo
            for p in visible
        ):
            raise ValueError("prediction decision violates adaptation cutoff/embargo")
        if len({p.point_id for p in visible}) != len(visible):
            raise ValueError("duplicate visible prediction point_id")
        if len(
            {(p.source_id, p.entity_id, _clock(p.decision_time, "decision")) for p in visible}
        ) != len(visible):
            raise ValueError("duplicate visible prediction observation key")
        torch = _torch()
        x = (
            np.array([p.values for p in visible]) - self.preprocessing.x_mean
        ) / self.preprocessing.x_scale
        with torch.inference_mode():
            mean, sigma = _forward(
                _tensors(self.parameters, self.config, torch, grad=False),
                torch.tensor(x, dtype=torch.float64, device="cpu"),
                self.config,
                torch,
            )
        return GaussianPrediction(
            visible,
            tuple(
                float(v)
                for v in mean.numpy() * self.preprocessing.y_scale + self.preprocessing.y_mean
            ),
            tuple(float(v) for v in sigma.numpy() * self.preprocessing.y_scale),
            self.config.target_kind,
            self.config.horizon,
            self.adapted_sha256,
            self.synthetic,
        )

    def score(self, query: tuple[LabelledPoint, ...], *, asof: datetime) -> dict[str, float]:
        _validate_rows(query, self.config, cutoff=_clock(asof, "score asof"))
        forecast = self.predict(tuple(r.point for r in query), asof=asof)
        if len(forecast.points) != len(query):
            raise ValueError("query score cannot omit future/unpublished rows")
        y, mean, sigma = (
            np.array(values)
            for values in (
                [r.target for r in query],
                forecast.mean,
                forecast.sigma,
            )
        )
        nll = 0.5 * math.log(2 * math.pi) + np.log(sigma) + 0.5 * ((y - mean) / sigma) ** 2
        return {
            "gaussian_nll": float(np.mean(nll)),
            "crps": float(np.mean(crps_gaussian(y, mean, sigma))),
        }


class MAMLRegimeLearner:
    def __init__(self, config: MAMLConfig) -> None:
        if not isinstance(config, MAMLConfig):
            raise ValueError("typed MAMLConfig required")
        self._config = config
        self._states: dict[str, ParameterState] = {}
        self._fit_cutoff: datetime | None = None
        self._heldout_ids: tuple[str, ...] = ()
        self._model_sha256: str | None = None
        self._receipt: dict[str, Any] = {}
        self._training_ids: frozenset[str] = frozenset()

    @property
    def config(self) -> MAMLConfig:
        return self._config

    @property
    def model_sha256(self) -> str | None:
        return self._model_sha256

    def meta_gradient(
        self,
        tasks: tuple[RegimeTask, ...],
        *,
        fit_cutoff: datetime,
        parameters: ParameterState | None = None,
    ) -> GradientAudit:
        _validate_tasks(tasks, self.config, _clock(fit_cutoff, "fit_cutoff"))
        if 1 + len(tasks) * self.config.inner_steps > self.config.max_gradient_evaluations:
            raise ValueError("meta-gradient audit resource budget exceeded")
        torch = _torch()
        params = (
            _initial(self.config, torch)
            if parameters is None
            else _tensors(parameters, self.config, torch, grad=True)
        )
        objective = _meta_objective(params, tasks, self.config, torch)
        gradients = torch.autograd.grad(objective, tuple(params.values()))
        return GradientAudit(
            float(objective.detach()),
            _state(params),
            _state(dict(zip(_NAMES, gradients, strict=True))),
            _hash([t.manifest_sha256 for t in tasks]),
        )

    def fit(
        self,
        tasks: tuple[RegimeTask, ...],
        *,
        fit_cutoff: datetime,
        heldout_task_ids: tuple[str, ...],
    ) -> MAMLRegimeLearner:
        fit_cutoff = _clock(fit_cutoff, "fit_cutoff")
        _validate_tasks(tasks, self.config, fit_cutoff)
        if self._states:
            raise RuntimeError("fitted meta-initializations are frozen; use a fresh instance")
        if (
            not isinstance(heldout_task_ids, tuple)
            or not 1 <= len(heldout_task_ids) <= self.config.max_tasks
        ):
            raise ValueError("explicit bounded heldout_task_ids tuple required")
        for task_id in heldout_task_ids:
            _name(task_id, "heldout_task_id")
        if len(set(heldout_task_ids)) != len(heldout_task_ids) or set(
            heldout_task_ids
        ).intersection(t.task_id for t in tasks):
            raise ValueError("training and heldout task IDs must be disjoint")
        calls = self.config.outer_steps * (2 + len(tasks) * self.config.inner_steps)
        if calls > self.config.max_gradient_evaluations:
            raise ValueError("configured meta-gradient resource budget exceeded")
        torch = _torch()
        params = _initial(self.config, torch)
        scratch = _state(params)
        pooled = _tensors(scratch, self.config, torch, grad=True)
        meta_optimizer = torch.optim.Adam(tuple(params.values()), lr=self.config.outer_lr)
        pool_optimizer = torch.optim.Adam(tuple(pooled.values()), lr=self.config.outer_lr)
        meta_losses, pooled_losses = [], []
        pooled_batches = []
        for task in tasks:
            preprocessing = _preprocess(task.support)
            pooled_batches.append(
                (_batch(task.support + task.query, preprocessing, torch), preprocessing)
            )
        for _ in range(self.config.outer_steps):
            meta_optimizer.zero_grad()
            meta_loss = _meta_objective(params, tasks, self.config, torch)
            pool_optimizer.zero_grad()
            pool_loss = torch.stack(
                [_nll(pooled, batch, prep, self.config, torch) for batch, prep in pooled_batches]
            ).mean()
            if not bool(torch.isfinite(meta_loss)) or not bool(torch.isfinite(pool_loss)):
                raise FloatingPointError("nonfinite Gaussian training objective; fit rejected")
            meta_loss.backward()
            pool_loss.backward()
            torch.nn.utils.clip_grad_norm_(tuple(params.values()), 10.0)
            torch.nn.utils.clip_grad_norm_(tuple(pooled.values()), 10.0)
            meta_optimizer.step()
            pool_optimizer.step()
            meta_losses.append(float(meta_loss.detach()))
            pooled_losses.append(float(pool_loss.detach()))
        states = {"maml": _state(params), "pooled": _state(pooled), "scratch": scratch}
        row_synthetic = [
            r.point.synthetic or r.label_synthetic for t in tasks for r in t.support + t.query
        ]
        data_label = (
            "SYNTHETIC"
            if all(row_synthetic)
            else "MIXED"
            if any(row_synthetic)
            else "SUPPLIED_NON_SYNTHETIC"
        )
        self._receipt = {
            "schema": "maml_regime.v1",
            "reference": "https://arxiv.org/abs/1703.03400",
            "config": _config_payload(self.config),
            "fit_cutoff": fit_cutoff.isoformat(),
            "heldout_task_ids": heldout_task_ids,
            "training_task_ids": tuple(t.task_id for t in tasks),
            "training_point_ids": tuple(
                r.point.point_id for t in tasks for r in t.support + t.query
            ),
            "task_manifest_sha256": tuple(t.manifest_sha256 for t in tasks),
            "data_label": data_label,
            "second_order": True,
            "device": "cpu",
            "meta_query_nll": tuple(meta_losses),
            "pooled_training_nll": tuple(pooled_losses),
            "pretraining_optimizer_steps": {
                "maml": self.config.outer_steps,
                "pooled": self.config.outer_steps,
                "scratch": 0,
            },
            "meta_inner_gradient_evaluations": self.config.outer_steps
            * len(tasks)
            * self.config.inner_steps,
            "pretraining_compute_matched": False,
            "research_only": True,
            "market_evidence": False,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "states": {name: asdict(state) for name, state in states.items()},
        }
        self._model_sha256 = _hash(self._receipt)
        self._states = states
        self._fit_cutoff = fit_cutoff
        self._heldout_ids = heldout_task_ids
        self._training_ids = frozenset(self._receipt["training_point_ids"])
        return self

    def adapt(
        self,
        task_id: str,
        support: tuple[LabelledPoint, ...],
        *,
        asof: datetime,
        arm: Arm = "maml",
        steps: int | None = None,
    ) -> AdaptedGaussianModel:
        asof = _clock(asof, "adaptation asof")
        if self._fit_cutoff is None or self.model_sha256 is None:
            raise RuntimeError("meta-learner is not fitted")
        if task_id not in self._heldout_ids:
            raise ValueError("adaptation requires a declared held-out task_id")
        if arm not in self._states:
            raise ValueError("unknown initialization arm")
        steps = self.config.inner_steps if steps is None else steps
        _count(steps, "adaptation steps", 0, self.config.inner_steps)
        _validate_rows(support, self.config, cutoff=None)
        visible = tuple(
            r for r in support if _clock(r.target_available_time, "label availability") <= asof
        )
        if len(visible) < 2:
            raise ValueError("adaptation requires at least two published support labels")
        if any(r.point.point_id in self._training_ids for r in visible):
            raise ValueError("held-out support reuses a training point_id")
        if (
            min(_clock(r.point.decision_time, "decision") for r in visible)
            < self._fit_cutoff + self.config.embargo
        ):
            raise ValueError("held-out support violates fit cutoff/embargo")
        preprocessing = _preprocess(visible)
        torch = _torch()
        params = _tensors(self._states[arm], self.config, torch, grad=True)
        adapted = _inner(
            params,
            _batch(visible, preprocessing, torch),
            preprocessing,
            self.config,
            torch,
            steps,
            second_order=False,
        )
        return AdaptedGaussianModel(
            task_id,
            arm,
            self.config,
            _state(adapted),
            preprocessing,
            asof,
            _hash([asdict(r) for r in visible]),
            self.model_sha256,
            len(visible),
            steps,
            self._receipt["data_label"] in ("SYNTHETIC", "MIXED")
            or any(r.point.synthetic or r.label_synthetic for r in visible),
        )

    def compare_initializations(
        self, tasks: tuple[RegimeTask, ...], *, asof: datetime
    ) -> dict[str, Any]:
        _validate_tasks(tasks, self.config, _clock(asof, "comparison asof"))
        if (
            3 * len(tasks) * sum(range(self.config.inner_steps + 1))
            > self.config.max_gradient_evaluations
        ):
            raise ValueError("adaptation comparison resource budget exceeded")
        outcomes = []
        for task in tasks:
            adaptation_cutoff = max(
                _clock(r.target_available_time, "support label") for r in task.support
            )
            for steps in range(self.config.inner_steps + 1):
                for arm in ("maml", "pooled", "scratch"):
                    model = self.adapt(
                        task.task_id, task.support, asof=adaptation_cutoff, arm=arm, steps=steps
                    )
                    outcomes.append(
                        {
                            "task_id": task.task_id,
                            "arm": arm,
                            "steps": steps,
                            "support_rows": model.support_rows,
                            "query_rows": len(task.query),
                            **model.score(task.query, asof=asof),
                            "adapted_sha256": model.adapted_sha256,
                        }
                    )
        return {
            "outcomes": outcomes,
            "adaptation_budget_matched": True,
            "support_matched": True,
            "pretraining_compute_matched": False,
            "pretraining_optimizer_steps": self._receipt.get("pretraining_optimizer_steps"),
            "source_task_sha256": tuple(t.manifest_sha256 for t in tasks),
            "synthetic": self._receipt["data_label"] in ("SYNTHETIC", "MIXED")
            or any(
                r.point.synthetic or r.label_synthetic for t in tasks for r in t.support + t.query
            ),
            "research_only": True,
            "market_evidence": False,
            "benefit_asserted": False,
        }

    def metadata(self) -> dict[str, Any]:
        # JSON round-trip returns a detached copy; parameter arrays are immutable tuples internally.
        return {
            **json.loads(json.dumps(self._receipt)),
            "model_sha256": self.model_sha256,
            "pretrained": False,
            "sota_claim": False,
            "research_only": True,
            "market_evidence": False,
        }

    def initialization(self, arm: Arm = "maml") -> ParameterState:
        if arm not in self._states:
            raise ValueError("fitted initialization arm required")
        return self._states[arm]

    def save(self, path: Path) -> None:
        if self.model_sha256 is None:
            raise RuntimeError("model is not fitted")
        path.write_text(
            json.dumps(
                {"receipt": self._receipt, "model_sha256": self.model_sha256}, sort_keys=True
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> MAMLRegimeLearner:
        if path.stat().st_size > 16_000_000:
            raise ValueError("model artifact exceeds resource bound")
        payload = json.loads(path.read_text(encoding="utf-8"))
        receipt = payload["receipt"]
        if receipt.get("schema") != "maml_regime.v1" or _hash(receipt) != payload["model_sha256"]:
            raise ValueError("model artifact schema/hash mismatch")
        if (
            receipt.get("second_order") is not True
            or receipt.get("research_only") is not True
            or receipt.get("market_evidence") is not False
            or receipt.get("device") != "cpu"
        ):
            raise ValueError("persisted model honesty/protocol mismatch")
        if (
            receipt["implementation_sha256"]
            != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ):
            raise ValueError("model implementation source mismatch")
        config = dict(receipt["config"])
        for name in ("horizon", "embargo"):
            config[name] = timedelta(seconds=config[name])
        model = cls(MAMLConfig(**config))
        model._states = {
            name: ParameterState(
                tuple(tuple(s) for s in value["shapes"]), tuple(tuple(v) for v in value["values"])
            )
            for name, value in receipt["states"].items()
        }
        if set(model._states) != {"maml", "pooled", "scratch"} or any(
            s.shapes != _shapes(model.config) for s in model._states.values()
        ):
            raise ValueError("persisted initialization shapes/arms mismatch")
        model._receipt = receipt
        model._fit_cutoff = _clock(datetime.fromisoformat(receipt["fit_cutoff"]), "fit_cutoff")
        model._heldout_ids = tuple(receipt["heldout_task_ids"])
        model._training_ids = frozenset(receipt["training_point_ids"])
        if not 1 <= len(model._heldout_ids) <= model.config.max_tasks or len(
            set(model._heldout_ids)
        ) != len(model._heldout_ids):
            raise ValueError("persisted held-out task registry is invalid")
        for task_id in model._heldout_ids:
            _name(task_id, "heldout_task_id")
        if set(model._heldout_ids).intersection(receipt["training_task_ids"]):
            raise ValueError("persisted task partitions overlap")
        model._model_sha256 = payload["model_sha256"]
        return model


def _config_payload(config: MAMLConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["horizon"] = config.horizon.total_seconds()
    payload["embargo"] = config.embargo.total_seconds()
    return payload


@dataclass(frozen=True, slots=True)
class AllocationConstraints:
    caps: tuple[float, ...]
    budget: float = 1.0
    risk_aversion: float = 5.0
    variance_limit: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.caps, tuple) or not self.caps:
            raise ValueError("caps must be a nonempty immutable tuple")
        for cap in self.caps:
            _finite(cap, "cap")
            if not 0 <= cap <= 1:
                raise ValueError("caps must lie in [0, 1]")
        _finite(self.budget, "budget")
        _finite(self.risk_aversion, "risk_aversion")
        if not 0 <= self.budget <= 1 or self.risk_aversion <= 0:
            raise ValueError("budget must lie in [0,1], risk_aversion must be positive")
        if self.variance_limit is not None:
            _finite(self.variance_limit, "variance_limit")
            if self.variance_limit < 0:
                raise ValueError("variance_limit must be nonnegative")


@dataclass(frozen=True, slots=True)
class CovarianceEvidence:
    entity_ids: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    horizon: timedelta
    available_time: datetime
    source_id: str
    source_sha256: str
    synthetic: bool

    def __post_init__(self) -> None:
        _name(self.source_id, "covariance source_id")
        _sha(self.source_sha256, "covariance source_sha256")
        _clock(self.available_time, "covariance availability")
        if (
            not isinstance(self.entity_ids, tuple)
            or not self.entity_ids
            or len(self.entity_ids) > 128
            or len(set(self.entity_ids)) != len(self.entity_ids)
        ):
            raise ValueError("covariance entities must be unique immutable tuple")
        for entity_id in self.entity_ids:
            _name(entity_id, "covariance entity_id")
        if not isinstance(self.values, tuple) or any(not isinstance(v, tuple) for v in self.values):
            raise ValueError("covariance values must be immutable tuples")
        for row in self.values:
            for value in row:
                _finite(value, "covariance")
        _covariance(np.array(self.values, dtype=float), len(self.entity_ids))
        if (
            not isinstance(self.horizon, timedelta)
            or self.horizon <= timedelta(0)
            or not isinstance(self.synthetic, bool)
        ):
            raise ValueError("covariance requires positive horizon and boolean synthetic identity")


def _covariance(matrix: Array, n: int) -> Array:
    if (
        matrix.shape != (n, n)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)
    ):
        raise ValueError("covariance must be finite symmetric square matrix")
    if np.linalg.eigvalsh(matrix).min() < -1e-12 or np.any(np.diag(matrix) <= 0):
        raise ValueError("covariance must be positive semidefinite with positive diagonal")
    return matrix


@dataclass(frozen=True, slots=True)
class ResearchAllocation:
    entity_ids: tuple[str, ...]
    weights: tuple[float, ...]
    cash: float
    predicted_variance: float
    covariance_kind: str
    covariance_sha256: str
    forecast_sha256: str
    constraints_sha256: str
    synthetic: bool
    receipt_sha256: str = dataclass_field(init=False)
    research_only: bool = dataclass_field(init=False, default=True)
    market_evidence: bool = dataclass_field(init=False, default=False)
    trading_policy: bool = dataclass_field(init=False, default=False)

    def __post_init__(self) -> None:
        payload = {
            name: getattr(self, name)
            for name in (
                "entity_ids",
                "weights",
                "cash",
                "predicted_variance",
                "covariance_kind",
                "covariance_sha256",
                "forecast_sha256",
                "constraints_sha256",
                "synthetic",
            )
        }
        object.__setattr__(self, "receipt_sha256", _hash(payload))


def allocate_research(
    forecast: GaussianPrediction,
    constraints: AllocationConstraints,
    *,
    asof: datetime,
    covariance: CovarianceEvidence | None = None,
) -> ResearchAllocation:
    asof = _clock(asof, "allocation asof")
    if forecast.target_kind != "forward_simple_return":
        raise ValueError("allocation requires declared forward_simple_return forecast units")
    entities = tuple(p.entity_id for p in forecast.points)
    if len(entities) > 128:
        raise ValueError("allocation asset resource bound exceeded")
    if len(set(entities)) != len(entities) or len(constraints.caps) != len(entities):
        raise ValueError("allocation entity ordering/caps mismatch")
    if any(_clock(p.decision_time, "decision") != asof for p in forecast.points):
        raise ValueError("allocation forecasts must share current decision cutoff")
    synthetic = forecast.synthetic
    if covariance is None:
        matrix = np.diag(np.array(forecast.sigma) ** 2)
        kind = "diagonal_predictive_variance_PROXY_zero_correlation"
        covariance_hash = _hash({"matrix": matrix.tolist(), "kind": kind})
    else:
        if covariance.entity_ids != entities or covariance.horizon != forecast.horizon:
            raise ValueError("covariance entity/horizon alignment mismatch")
        if _clock(covariance.available_time, "covariance availability") > asof:
            raise ValueError("covariance unpublished at allocation cutoff")
        matrix = _covariance(np.array(covariance.values, dtype=float), len(entities))
        kind = "supplied_dated_covariance_evidence"
        covariance_hash = _hash(asdict(covariance))
        synthetic = synthetic or covariance.synthetic
    _covariance(matrix, len(entities))
    mu = np.array(forecast.mean)
    caps = np.array(constraints.caps)
    if constraints.budget == 0 or constraints.variance_limit == 0:
        weights = np.zeros(
            len(entities)
        )  # Conservative cash policy for a zero risk/budget request.
    else:
        constraint_list = [
            {
                "type": "ineq",
                "fun": lambda w: constraints.budget - w.sum(),
                "jac": lambda w: -np.ones_like(w),
            }
        ]
        if constraints.variance_limit is not None:
            limit = constraints.variance_limit
            constraint_list.append(
                {
                    "type": "ineq",
                    "fun": lambda w: limit - float(w @ matrix @ w),
                    "jac": lambda w: -2 * matrix @ w,
                }
            )
        result = minimize(
            lambda w: 0.5 * constraints.risk_aversion * float(w @ matrix @ w) - float(mu @ w),
            np.zeros(len(entities)),
            jac=lambda w: constraints.risk_aversion * matrix @ w - mu,
            bounds=tuple((0.0, float(cap)) for cap in caps),
            constraints=constraint_list,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not result.success:
            raise RuntimeError(f"research allocation optimization failed: {result.message}")
        weights = np.clip(result.x, 0, caps)
        if weights.sum() > constraints.budget:
            weights *= constraints.budget / weights.sum()
    variance = float(weights @ matrix @ weights)
    if (
        not np.isfinite(weights).all()
        or weights.sum() > constraints.budget + 1e-8
        or (constraints.variance_limit is not None and variance > constraints.variance_limit + 1e-8)
    ):
        raise RuntimeError("allocation constraint verification failed")
    return ResearchAllocation(
        entities,
        tuple(float(w) for w in weights),
        max(0.0, constraints.budget - float(weights.sum())),
        variance,
        kind,
        covariance_hash,
        forecast.forecast_sha256,
        _hash(asdict(constraints)),
        synthetic,
    )
