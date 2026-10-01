"""Offline isolation-forest anomaly research, with delayed annotation calibration.

Liu, Ting and Zhou (ICDM 2008), Isolation Forest:
https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
Training uses sklearn's real random isolation trees. The frozen tree traversal
below reproduces its numerical anomaly score without pickle or executable code.
Separate later annotations fit regularized sigmoid calibration; labels are an
external convention, never proof of insider misconduct or informed intent.
https://scikit-learn.org/stable/modules/calibration.html

Synthetic tests establish correctness only. Tape rights, independently audited
annotations, empirical effectiveness and operational surveillance remain open.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from quant_fund.metrics.probability import brier_score, expected_calibration_error, log_loss

_FEATURES = (
    "log_quantity",
    "relative_spread",
    "signed_price_distance_bps",
    "volume_zscore",
    "trade_rate",
    "order_cancel_ratio",
)
_MAX_ROWS = 4096
_MAX_BYTES = 16_000_000


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _clock(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clocks must be timezone-aware")
    return value.astimezone(UTC)


def _text(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("nonempty bounded text required")


def _count(value: int, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer outside resource budget")


@dataclass(frozen=True)
class TradePattern:
    """Caller-computed causal trade-log features; no identities inferred here."""

    pattern_id: str
    event_time: datetime
    available_time: datetime
    decision_time: datetime
    features: tuple[float, ...]
    source: str
    synthetic: bool

    def __post_init__(self) -> None:
        _text(self.pattern_id)
        _text(self.source)
        for name in ("event_time", "available_time", "decision_time"):
            object.__setattr__(self, name, _clock(getattr(self, name)))
        if not self.event_time <= self.available_time <= self.decision_time:
            raise ValueError("future features or invalid event clock")
        values = tuple(self.features)
        if len(values) != len(_FEATURES) or any(
            type(x) not in (float, int) or not math.isfinite(x) or abs(x) > 1e20 for x in values
        ):
            raise ValueError("six finite trade-pattern features required")
        if values[0] < 0 or values[1] < 0 or values[4] < 0 or not 0 <= values[5] <= 1:
            raise ValueError("invalid trade-pattern feature domain")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be boolean")
        object.__setattr__(self, "features", tuple(float(x) for x in values))

    def record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            **{
                name: getattr(self, name).isoformat()
                for name in ("event_time", "available_time", "decision_time")
            },
        }


@dataclass(frozen=True)
class PatternAnnotation:
    pattern_id: str
    suspicious: bool
    available_time: datetime
    source: str
    method: str
    synthetic: bool

    def __post_init__(self) -> None:
        for value in (self.pattern_id, self.source, self.method):
            _text(value)
        if type(self.suspicious) is not bool:
            raise ValueError("annotation must be boolean")
        if type(self.synthetic) is not bool:
            raise ValueError("annotation synthetic must be boolean")
        object.__setattr__(self, "available_time", _clock(self.available_time))

    def record(self) -> dict[str, Any]:
        return {**asdict(self), "available_time": self.available_time.isoformat()}


@dataclass(frozen=True)
class PatternScore:
    pattern_id: str
    decision_time: datetime
    anomaly_score: float
    suspicious_annotation_probability: float | None
    model_sha256: str
    synthetic: bool
    intent: str = "UNKNOWN"
    insider_misconduct: str = "UNKNOWN"


def _path_adjustment(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + np.euler_gamma) - 2.0 * (n - 1) / n


class TradeAnomalyModel:
    """One frozen unsupervised fit and optional later frozen sigmoid calibration."""

    def __init__(self, *, trees: int = 64, samples: int = 256, seed: int = 7) -> None:
        _count(trees, 1, 256)
        _count(samples, 2, 256)
        _count(seed, 0, 2**32 - 1)
        self.config = {"trees": trees, "samples": samples, "seed": seed}
        self._payload: dict[str, Any] | None = None
        self._identity: str | None = None

    @property
    def model_sha256(self) -> str:
        self._check()
        assert self._identity is not None
        return self._identity

    def _check(self) -> dict[str, Any]:
        if self._payload is None or self._identity is None:
            raise ValueError("model not fitted")
        if self.config != self._payload["config"] or _hash(self._payload) != self._identity:
            raise ValueError("frozen anomaly model mutated")
        return self._payload

    def _rows(self, rows: Sequence[TradePattern], *, asof: datetime) -> list[TradePattern]:
        _count(len(rows), 1, _MAX_ROWS)
        clock = _clock(asof)
        result = list(rows)
        if any(not isinstance(row, TradePattern) for row in result):
            raise ValueError("typed trade patterns required")
        if len({row.pattern_id for row in result}) != len(result):
            raise ValueError("duplicate pattern ids")
        if any(row.decision_time > clock for row in result):
            raise ValueError("pattern decision after cutoff")
        if len({(row.source, row.synthetic) for row in result}) != 1:
            raise ValueError("mixed source or synthetic identity")
        if self._payload is not None and (
            result[0].source != self._payload["source"]
            or result[0].synthetic is not self._payload["feature_synthetic"]
        ):
            raise ValueError("source identity differs from training")
        return result

    def fit(self, rows: Sequence[TradePattern], *, asof: datetime) -> TradeAnomalyModel:
        if self._payload is not None:
            raise ValueError("frozen fit; use a new model for retraining")
        values = self._rows(rows, asof=asof)
        if len(values) < 32:
            raise ValueError("at least 32 training patterns required")
        forest = IsolationForest(
            n_estimators=self.config["trees"],
            max_samples=min(self.config["samples"], len(values)),
            max_features=1.0,
            contamination="auto",
            random_state=self.config["seed"],
            n_jobs=1,
        )
        matrix = np.array([row.features for row in values], dtype=np.float32)
        forest.fit(matrix)
        trees = []
        for estimator in forest.estimators_:
            tree = estimator.tree_
            trees.append(
                {
                    "left": tree.children_left.tolist(),
                    "right": tree.children_right.tolist(),
                    "feature": tree.feature.tolist(),
                    "threshold": tree.threshold.tolist(),
                    "count": tree.n_node_samples.tolist(),
                }
            )
        payload = {
            "schema_version": "trade_anomaly_model_v1",
            "config": dict(self.config),
            "feature_names": list(_FEATURES),
            "trees": trees,
            "samples": int(forest.max_samples_),
            "fit_asof": _clock(asof).isoformat(),
            "source": values[0].source,
            "synthetic": values[0].synthetic,
            "feature_synthetic": values[0].synthetic,
            "training_ids": [row.pattern_id for row in values],
            "training_sha256": _hash([row.record() for row in values]),
            "calibration": None,
            "research_only": True,
            "market_evidence": False,
            "live_pnl_claim": False,
            "intent": "UNKNOWN",
            "insider_misconduct": "UNKNOWN",
            "annotation_independence_verified": False,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": {"numpy": np.__version__, "sklearn": sklearn.__version__},
        }
        self._payload, self._identity = payload, _hash(payload)
        return self

    def _raw(self, rows: Sequence[TradePattern]) -> np.ndarray:
        payload = self._check()
        # sklearn evaluates float32 features against float64 tree thresholds.
        matrix = np.array([row.features for row in rows], dtype=np.float32)
        depths = np.zeros(len(rows), dtype=float)
        for tree in payload["trees"]:
            for i, row in enumerate(matrix):
                node, depth = 0, 0
                while tree["left"][node] != -1:
                    node = (
                        tree["left"][node]
                        if row[tree["feature"][node]] <= tree["threshold"][node]
                        else tree["right"][node]
                    )
                    depth += 1
                depths[i] += depth + _path_adjustment(tree["count"][node])
        return np.asarray(
            2.0 ** (-depths / (len(payload["trees"]) * _path_adjustment(payload["samples"]))),
            dtype=float,
        )

    def _annotations(
        self, rows: list[TradePattern], annotations: Sequence[PatternAnnotation], *, asof: datetime
    ) -> np.ndarray:
        if len(annotations) != len(rows) or any(
            not isinstance(a, PatternAnnotation) for a in annotations
        ):
            raise ValueError("one typed annotation per pattern required")
        lookup = {a.pattern_id: a for a in annotations}
        if len(lookup) != len(annotations) or set(lookup) != {row.pattern_id for row in rows}:
            raise ValueError("annotation ids do not match patterns")
        if len({(a.source, a.method, a.synthetic) for a in annotations}) != 1:
            raise ValueError("mixed annotation conventions")
        for row in rows:
            annotation = lookup[row.pattern_id]
            if not row.event_time <= annotation.available_time <= _clock(asof):
                raise ValueError("future or impossible annotation clock")
        return np.array([float(lookup[row.pattern_id].suspicious) for row in rows])

    def calibrate(
        self,
        rows: Sequence[TradePattern],
        annotations: Sequence[PatternAnnotation],
        *,
        asof: datetime,
    ) -> TradeAnomalyModel:
        payload = self._check()
        if payload["calibration"] is not None:
            raise ValueError("calibration frozen")
        values = self._rows(rows, asof=asof)
        if (
            len(values) < 32
            or any(
                row.decision_time <= datetime.fromisoformat(payload["fit_asof"]) for row in values
            )
            or set(payload["training_ids"]) & {row.pattern_id for row in values}
        ):
            raise ValueError("calibration must be disjoint and later than training")
        y = self._annotations(values, annotations, asof=asof)
        if min(int(y.sum()), len(y) - int(y.sum())) < 4:
            raise ValueError("calibration requires at least four annotations per class")
        raw = self._raw(values)
        mean, scale = float(raw.mean()), max(float(raw.std()), 1e-12)
        calibrator = LogisticRegression(C=1.0, max_iter=1000, random_state=self.config["seed"])
        calibrator.fit(((raw - mean) / scale).reshape(-1, 1), y)
        calibration = {
            "asof": _clock(asof).isoformat(),
            "ids": [row.pattern_id for row in values],
            "sha256": _hash(
                {
                    "patterns": [r.record() for r in values],
                    "annotations": [a.record() for a in annotations],
                }
            ),
            "source": annotations[0].source,
            "method": annotations[0].method,
            "synthetic": annotations[0].synthetic,
            "mean": mean,
            "scale": scale,
            "coef": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
            "prevalence": float(y.mean()),
        }
        self._payload = {
            **payload,
            "calibration": calibration,
            "synthetic": payload["feature_synthetic"] or annotations[0].synthetic,
        }
        self._identity = _hash(self._payload)
        return self

    def predict(self, rows: Sequence[TradePattern]) -> tuple[PatternScore, ...]:
        payload = self._check()
        if not rows:
            raise ValueError("empty prediction")
        values = self._rows(rows, asof=max(row.decision_time for row in rows))
        calibration = payload["calibration"]
        cutoff = datetime.fromisoformat(calibration["asof"] if calibration else payload["fit_asof"])
        if any(row.decision_time < cutoff for row in values):
            raise ValueError("model or calibration after decision")
        scores = self._raw(values)
        probabilities = (
            expit(
                calibration["coef"] * (scores - calibration["mean"]) / calibration["scale"]
                + calibration["intercept"]
            )
            if calibration
            else None
        )
        identity = self.model_sha256
        return tuple(
            PatternScore(
                row.pattern_id,
                row.decision_time,
                float(scores[i]),
                None if probabilities is None else float(probabilities[i]),
                identity,
                payload["synthetic"],
            )
            for i, row in enumerate(values)
        )

    def evaluate(
        self,
        rows: Sequence[TradePattern],
        annotations: Sequence[PatternAnnotation],
        *,
        asof: datetime,
    ) -> dict[str, Any]:
        payload = self._check()
        calibration = payload["calibration"]
        if calibration is None:
            raise ValueError("probability scoring requires independent later calibration")
        values = self._rows(rows, asof=asof)
        if any(
            row.decision_time <= datetime.fromisoformat(calibration["asof"]) for row in values
        ) or (set(payload["training_ids"]) | set(calibration["ids"])) & {
            row.pattern_id for row in values
        }:
            raise ValueError("evaluation must be disjoint and later than calibration")
        y = self._annotations(values, annotations, asof=asof)
        if (
            annotations[0].source != calibration["source"]
            or annotations[0].method != calibration["method"]
        ):
            raise ValueError("evaluation annotation convention changed")
        p = np.array(
            [score.suspicious_annotation_probability for score in self.predict(values)], dtype=float
        )
        baseline = np.full(len(y), calibration["prevalence"])
        alerts = p >= 0.5  # Predeclared; never optimized against this holdout.
        tp = int(np.sum(alerts & (y == 1)))
        fp = int(np.sum(alerts & (y == 0)))
        fn = int(np.sum(~alerts & (y == 1)))
        return {
            "model_sha256": self.model_sha256,
            "evaluation_sha256": _hash(
                {
                    "patterns": [r.record() for r in values],
                    "annotations": [a.record() for a in annotations],
                    "asof": _clock(asof).isoformat(),
                }
            ),
            "n": len(y),
            "brier": brier_score(p, y),
            "log_loss": log_loss(p, y),
            "ece": expected_calibration_error(p, y, min(10, len(y))),
            "baseline_brier": brier_score(baseline, y),
            "baseline_log_loss": log_loss(baseline, y),
            "diagnostics": {
                "alert_threshold": 0.5,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "false_positives": fp,
                "true_positives": tp,
            },
            "probability_semantics": "supplied suspicious-pattern annotation; never insider misconduct",
            "synthetic": payload["synthetic"] or annotations[0].synthetic,
            "synthetic_feature_data": payload["feature_synthetic"],
            "synthetic_calibration_annotations": calibration["synthetic"],
            "synthetic_evaluation_annotations": annotations[0].synthetic,
            "research_only": True,
            "market_evidence": False,
            "live_pnl_claim": False,
            "annotation_independence_verified": False,
        }

    def snapshot(self) -> dict[str, Any]:
        payload = self._check()
        snapshot: dict[str, Any] = json.loads(
            json.dumps({"model_sha256": self.model_sha256, "payload": payload}, allow_nan=False)
        )
        return snapshot

    def save(self, path: Path) -> None:
        snapshot = self.snapshot()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot, sort_keys=True, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> TradeAnomalyModel:
        if path.stat().st_size > _MAX_BYTES:
            raise ValueError("anomaly artifact exceeds resource budget")
        with path.open("rb") as stream:
            raw = stream.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            raise ValueError("anomaly artifact exceeds resource budget")
        return cls.restore(json.loads(raw))

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> TradeAnomalyModel:
        if len(json.dumps(snapshot, allow_nan=False)) > _MAX_BYTES or set(snapshot) != {
            "model_sha256",
            "payload",
        }:
            raise ValueError("invalid anomaly artifact envelope")
        payload = snapshot["payload"]
        if (
            not isinstance(payload, dict)
            or _hash(payload) != snapshot["model_sha256"]
            or payload.get("schema_version") != "trade_anomaly_model_v1"
        ):
            raise ValueError("anomaly artifact identity mismatch")
        required = {
            "schema_version",
            "config",
            "feature_names",
            "trees",
            "samples",
            "fit_asof",
            "source",
            "synthetic",
            "feature_synthetic",
            "training_ids",
            "training_sha256",
            "calibration",
            "research_only",
            "market_evidence",
            "live_pnl_claim",
            "intent",
            "insider_misconduct",
            "annotation_independence_verified",
            "implementation_sha256",
            "runtime",
        }
        if set(payload) != required or payload["feature_names"] != list(_FEATURES):
            raise ValueError("anomaly artifact schema mismatch")
        for name, expected in {
            "research_only": True,
            "market_evidence": False,
            "live_pnl_claim": False,
            "annotation_independence_verified": False,
            "intent": "UNKNOWN",
            "insider_misconduct": "UNKNOWN",
        }.items():
            if payload[name] != expected or type(payload[name]) is not type(expected):
                raise ValueError("anomaly artifact honesty mismatch")
        if type(payload["synthetic"]) is not bool or type(payload["feature_synthetic"]) is not bool:
            raise ValueError("invalid source identity")
        _text(payload["source"])
        for name in ("implementation_sha256", "training_sha256"):
            if (
                not isinstance(payload[name], str)
                or len(payload[name]) != 64
                or any(c not in "0123456789abcdef" for c in payload[name])
            ):
                raise ValueError("invalid provenance hash")
        if not isinstance(payload["runtime"], dict) or set(payload["runtime"]) != {
            "numpy",
            "sklearn",
        }:
            raise ValueError("invalid runtime provenance")
        for value in payload["runtime"].values():
            _text(value)
        fitted_at = _clock(datetime.fromisoformat(payload["fit_asof"]))
        model = cls(**payload["config"])
        _count(payload["samples"], 2, model.config["samples"])
        if len(payload["trees"]) != model.config["trees"]:
            raise ValueError("forest size mismatch")
        for tree in payload["trees"]:
            if set(tree) != {"left", "right", "feature", "threshold", "count"}:
                raise ValueError("tree schema mismatch")
            n = len(tree["left"])
            _count(n, 1, 2 * payload["samples"] - 1)
            if any(len(values) != n for values in tree.values()):
                raise ValueError("tree dimensions mismatch")
            parents = [0] * n
            depths = [0] * n
            max_depth = math.ceil(math.log2(payload["samples"]))
            for i in range(n):
                if depths[i] > max_depth:
                    raise ValueError("tree exceeds fitted isolation depth budget")
                _count(tree["count"][i], 1, payload["samples"])
                threshold = tree["threshold"][i]
                if type(threshold) not in (float, int) or not math.isfinite(threshold):
                    raise ValueError("invalid tree threshold")
                left, right = tree["left"][i], tree["right"][i]
                if left == right == -1 and type(left) is int and type(right) is int:
                    if tree["feature"][i] != -2:
                        raise ValueError("invalid leaf feature")
                else:
                    _count(left, i + 1, n - 1)
                    _count(right, i + 1, n - 1)
                    _count(tree["feature"][i], 0, len(_FEATURES) - 1)
                    if (
                        left == right
                        or tree["count"][left] + tree["count"][right] != tree["count"][i]
                    ):
                        raise ValueError("invalid tree children")
                    parents[left] += 1
                    parents[right] += 1
                    depths[left] = depths[i] + 1
                    depths[right] = depths[i] + 1
            if parents != [0] + [1] * (n - 1) or tree["count"][0] != payload["samples"]:
                raise ValueError("tree must be one reachable acyclic partition")
        training_ids = payload["training_ids"]
        _count(len(training_ids), 32, _MAX_ROWS)
        for value in training_ids:
            _text(value)
        if len(set(training_ids)) != len(training_ids):
            raise ValueError("duplicate training ids")
        if payload["samples"] != min(model.config["samples"], len(training_ids)):
            raise ValueError("sample count disagrees with training configuration")
        calibration = payload["calibration"]
        if calibration is not None:
            if set(calibration) != {
                "asof",
                "ids",
                "sha256",
                "source",
                "method",
                "mean",
                "scale",
                "coef",
                "intercept",
                "prevalence",
                "synthetic",
            }:
                raise ValueError("calibration schema mismatch")
            if _clock(datetime.fromisoformat(calibration["asof"])) <= fitted_at:
                raise ValueError("invalid calibration cutoff")
            _count(len(calibration["ids"]), 32, _MAX_ROWS)
            if len(set(calibration["ids"])) != len(calibration["ids"]) or set(training_ids) & set(
                calibration["ids"]
            ):
                raise ValueError("calibration ids overlap")
            for value in (*calibration["ids"], calibration["source"], calibration["method"]):
                _text(value)
            for name in ("mean", "scale", "coef", "intercept", "prevalence"):
                if type(calibration[name]) not in (float, int) or not math.isfinite(
                    calibration[name]
                ):
                    raise ValueError("invalid calibration parameter")
            if calibration["scale"] <= 0 or not 0 < calibration["prevalence"] < 1:
                raise ValueError("invalid calibration scale or prevalence")
            if type(calibration["synthetic"]) is not bool:
                raise ValueError("invalid annotation evidence class")
        if payload["synthetic"] is not (
            payload["feature_synthetic"] or bool(calibration and calibration["synthetic"])
        ):
            raise ValueError("synthetic annotations cannot become empirical evidence")
        model._payload = json.loads(json.dumps(payload, allow_nan=False))
        model._identity = snapshot["model_sha256"]
        model._check()
        return model
