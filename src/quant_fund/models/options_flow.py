"""Offline options-flow classification of supplied annotation semantics.

BSM Greeks use ``models.options`` (Black–Scholes/Merton 1973). A real
logistic/gradient-boosted-tree ensemble learns timestamped informed-versus-
hedge annotations; a disjoint later chronological period fits Platt scaling.
Method references: sklearn GradientBoostingClassifier and probability
calibration documentation (https://scikit-learn.org/stable/modules/calibration.html).

Trade size or a volume spike is not evidence of actual informed/institutional
intent. Probabilities predict the supplied annotation convention only. No
labels means no fitted classifier/confidence. Tests use synthetic annotations;
licensed tape, independent real labels, streaming infrastructure and UI remain
external integration/evidence requirements. Multi-leg trades are unsupported.
There is no feed acquisition, network notification or broker connection.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from quant_fund.metrics.probability import brier_score, expected_calibration_error, log_loss
from quant_fund.models.options import bs_greeks

Array = NDArray[np.float64]
_MAX_TAPE = 10_000
_SEMANTICS = "Probability of supplied informed annotation versus hedge; not observed intent"
_FEATURE_NAMES = (
    "log_size",
    "log_premium",
    "delta",
    "gamma",
    "vega",
    "days_to_expiry",
    "implied_vol",
    "log_moneyness",
    "quote_price_position",
    "volume_zscore",
    "zscore_available",
    "block_flag",
    "relative_spread",
    "quote_age_seconds",
    "underlying_age_seconds",
)


def _fit_training_fold_scaler(training_fold: Array) -> StandardScaler:
    """Fit only the caller's chronology-filtered training annotations."""
    return StandardScaler().fit(training_fold)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, default=_json_clock).encode()
    ).hexdigest()


def _json_clock(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported hash value: {type(value).__name__}")


def _clock(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone aware")
    return value.astimezone(UTC)


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty unpadded string")


def _positive(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class OptionTrade:
    trade_id: str
    contract_id: str
    underlying: str
    strike: float
    option_type: Literal["call", "put"]
    expiry: datetime
    event_time: datetime
    available_time: datetime
    price: float
    size: int
    implied_vol: float
    underlying_price: float
    underlying_event_time: datetime
    underlying_available_time: datetime
    bid: float
    ask: float
    quote_event_time: datetime
    quote_available_time: datetime
    data_source: str
    synthetic: bool
    multi_leg: bool = False
    multiplier: float = 100.0
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("trade_id", "contract_id", "underlying", "data_source"):
            _text(getattr(self, name), name)
        for name in (
            "expiry",
            "event_time",
            "available_time",
            "underlying_event_time",
            "underlying_available_time",
            "quote_event_time",
            "quote_available_time",
        ):
            object.__setattr__(self, name, _clock(getattr(self, name), name))
        for name in (
            "strike",
            "price",
            "implied_vol",
            "underlying_price",
            "ask",
            "multiplier",
        ):
            _positive(getattr(self, name), name)
        if (
            isinstance(self.bid, bool)
            or not isinstance(self.bid, (float, int))
            or not math.isfinite(self.bid)
            or self.bid < 0
        ):
            raise ValueError("bid must be finite and nonnegative")
        if type(self.size) is not int or not 1 <= self.size <= 10_000_000:
            raise ValueError("size must be a bounded positive integer")
        if self.option_type not in {"call", "put"} or self.ask < self.bid:
            raise ValueError("option_type/quote spread is invalid")
        if type(self.synthetic) is not bool or type(self.multi_leg) is not bool:
            raise ValueError("synthetic and multi_leg must be explicit bools")
        if (
            isinstance(self.risk_free_rate, bool)
            or not isinstance(self.risk_free_rate, (float, int))
            or not math.isfinite(self.risk_free_rate)
            or abs(self.risk_free_rate) > 1
        ):
            raise ValueError("risk_free_rate must be finite")
        if self.available_time < self.event_time or self.expiry <= self.event_time:
            raise ValueError("trade availability/expiry clock is invalid")
        if (self.expiry - self.event_time).total_seconds() > 50 * 366 * 86_400:
            raise ValueError("expiry exceeds the bounded BSM horizon")
        for event, ready in (
            (self.underlying_event_time, self.underlying_available_time),
            (self.quote_event_time, self.quote_available_time),
        ):
            if event > self.event_time or ready < event or ready > self.available_time:
                raise ValueError("quote/underlying data must be point-in-time available")

    @property
    def source_sha256(self) -> str:
        return _hash(asdict(self))


def ingest_option_tape(line: str | Mapping[str, Any]) -> OptionTrade:
    """Strict JSON/dict parser; retains original event/availability clocks in UTC."""
    if isinstance(line, str):
        if len(line.encode()) > 16_384:
            raise ValueError("option tape line exceeds the resource bound")

        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            row: dict[str, Any] = {}
            for key, value in pairs:
                if key in row:
                    raise ValueError("duplicate JSON tape field")
                row[key] = value
            return row

        try:
            row = json.loads(line, object_pairs_hook=unique)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid option tape JSON") from exc
    else:
        row = dict(line)
    if not isinstance(row, dict):
        raise ValueError("option tape line must be an object")
    fields = OptionTrade.__dataclass_fields__
    optional = {"multi_leg", "multiplier", "risk_free_rate"}
    if set(row) - set(fields) or set(fields) - optional - set(row):
        raise ValueError("unknown or missing option tape fields")
    for key in (
        "expiry",
        "event_time",
        "available_time",
        "underlying_event_time",
        "underlying_available_time",
        "quote_event_time",
        "quote_available_time",
    ):
        if isinstance(row[key], str):
            try:
                row[key] = datetime.fromisoformat(row[key].replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"invalid {key} clock") from exc
    return OptionTrade(**row)


@dataclass(frozen=True)
class IntentAnnotation:
    trade_id: str
    intent: Literal["informed", "hedge"]
    available_time: datetime
    annotation_source: str
    evidence_sha256: str
    synthetic: bool

    def __post_init__(self) -> None:
        _text(self.trade_id, "trade_id")
        _text(self.annotation_source, "annotation_source")
        _text(self.evidence_sha256, "evidence_sha256")
        object.__setattr__(
            self, "available_time", _clock(self.available_time, "label availability")
        )
        if self.intent not in {"informed", "hedge"} or type(self.synthetic) is not bool:
            raise ValueError("intent must be a supplied informed/hedge annotation")
        if len(self.evidence_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.evidence_sha256
        ):
            raise ValueError("annotation evidence must carry a SHA-256 identity")


@dataclass(frozen=True)
class OptionsFlowConfig:
    lookback: int = 50
    min_history: int = 5
    block_size: int = 500
    max_quote_age_seconds: float = 60.0
    n_estimators: int = 64
    tree_depth: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        for name, low, high in (
            ("lookback", 2, 512),
            ("min_history", 2, 512),
            ("block_size", 1, 10_000_000),
            ("n_estimators", 2, 256),
            ("tree_depth", 1, 4),
            ("seed", 0, 2**32 - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be a bounded integer")
        if self.min_history > self.lookback:
            raise ValueError("min_history cannot exceed lookback")
        _positive(self.max_quote_age_seconds, "max_quote_age_seconds")


@dataclass(frozen=True)
class TradeFeatures:
    trade_id: str
    decision_time: datetime
    values: tuple[float, ...]
    volume_zscore: float | None
    n_history: int
    block_flag: bool
    source_sha256: str
    feature_sha256: str
    schema_sha256: str
    synthetic: bool


def trade_features(
    trade: OptionTrade,
    history: Sequence[OptionTrade],
    config: OptionsFlowConfig | None = None,
) -> TradeFeatures:
    cfg = config or OptionsFlowConfig()
    if trade.multi_leg:
        raise ValueError("unsupported multi-leg option trade")
    quote_age = (trade.event_time - trade.quote_event_time).total_seconds()
    underlying_age = (trade.event_time - trade.underlying_event_time).total_seconds()
    if max(quote_age, underlying_age) > cfg.max_quote_age_seconds:
        raise ValueError("stale option/underlying quote")
    if len(history) > _MAX_TAPE:
        raise ValueError("history exceeds the offline tape bound")
    past = sorted(
        (
            t
            for t in history
            if t.contract_id == trade.contract_id
            and t.trade_id != trade.trade_id
            and not t.multi_leg
            and t.available_time < trade.available_time
            and t.event_time <= trade.event_time
        ),
        key=lambda t: (t.available_time, t.trade_id),
    )[-cfg.lookback :]
    if any(
        (t.underlying, t.strike, t.option_type, t.expiry, t.multiplier)
        != (trade.underlying, trade.strike, trade.option_type, trade.expiry, trade.multiplier)
        for t in past
    ):
        raise ValueError("contract_id refers to inconsistent option contracts")
    if len({t.trade_id for t in past}) != len(past):
        raise ValueError("duplicate historical option trade identities")
    if len(past) < cfg.min_history:
        raise ValueError("insufficient trailing option trade history")
    sizes = np.asarray([t.size for t in past], dtype=float)
    sd = float(sizes.std())
    zscore = float((trade.size - sizes.mean()) / sd) if sd > 1e-12 else None
    days = (trade.expiry - trade.event_time).total_seconds() / 86_400
    greeks = bs_greeks(
        trade.underlying_price,
        trade.strike,
        days / 365.25,
        trade.implied_vol,
        trade.risk_free_rate,
        trade.option_type == "call",
    )
    midpoint = (trade.bid + trade.ask) / 2
    spread = trade.ask - trade.bid
    position = (trade.price - midpoint) / spread if spread > 0 else 0.0
    # Undefined trailing z-score is encoded with an explicit availability flag,
    # never presented as an observed zero z-score.
    values = (
        math.log1p(trade.size),
        math.log1p(trade.price * trade.size * trade.multiplier),
        greeks["delta"],
        greeks["gamma"],
        greeks["vega"],
        days,
        trade.implied_vol,
        math.log(trade.underlying_price / trade.strike),
        position,
        zscore if zscore is not None else 0.0,
        float(zscore is not None),
        float(trade.size >= cfg.block_size),
        spread / midpoint,
        quote_age,
        underlying_age,
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("non-finite option flow features")
    schema_hash = _hash({"feature_names": _FEATURE_NAMES, "config": asdict(cfg), "revision": 1})
    source_hash = _hash({"trade": trade.source_sha256, "history": [t.source_sha256 for t in past]})
    feature_hash = _hash({"schema": schema_hash, "source": source_hash, "values": values})
    return TradeFeatures(
        trade.trade_id,
        trade.available_time,
        values,
        zscore,
        len(past),
        trade.size >= cfg.block_size,
        source_hash,
        feature_hash,
        schema_hash,
        trade.synthetic or any(t.synthetic for t in past),
    )


@dataclass(frozen=True)
class FlowFitInfo:
    model_sha256: str
    training_data_sha256: str
    calibration_data_sha256: str
    feature_schema_sha256: str
    train_cutoff: str
    cutoff: str
    n_training: int
    n_calibration: int
    annotation_sources: tuple[str, ...]
    synthetic: bool
    calibration_brier: float
    calibration_log_loss: float
    calibration_ece: float
    label_semantics: str = field(default=_SEMANTICS, init=False)


@dataclass(frozen=True)
class FlowScore:
    trade_id: str
    decision_time: str
    status: str
    probability_informed_annotation: float | None
    model_sha256: str | None
    feature_sha256: str | None
    source_sha256: str
    feature_schema_sha256: str | None
    training_synthetic: bool | None
    input_synthetic: bool
    in_sample: bool | None
    annotation_sources: tuple[str, ...] = ()
    label_semantics: str = field(default=_SEMANTICS, init=False)
    research_only: bool = field(default=True, init=False)
    live_pnl_claim: bool = field(default=False, init=False)
    actual_intent_established: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _text(self.trade_id, "trade_id")
        _clock(datetime.fromisoformat(self.decision_time), "decision_time")
        p = self.probability_informed_annotation
        if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError("flow probability must be finite and bounded")
        if self.status == "scored" and (
            p is None
            or self.model_sha256 is None
            or self.feature_sha256 is None
            or self.feature_schema_sha256 is None
            or not self.annotation_sources
        ):
            raise ValueError("scored flow requires model, features and supplied annotation sources")
        if self.status != "scored" and p is not None:
            raise ValueError("unavailable flow cannot carry class confidence")


def _tape(trades: Sequence[OptionTrade]) -> list[OptionTrade]:
    if len(trades) > _MAX_TAPE or not all(isinstance(t, OptionTrade) for t in trades):
        raise ValueError("trades must be a bounded typed offline tape")
    if len({t.trade_id for t in trades}) != len(trades):
        raise ValueError("duplicate option trade identities")
    return sorted(trades, key=lambda t: (t.available_time, t.trade_id))


def _annotations(
    labels: Sequence[IntentAnnotation], trades: Sequence[OptionTrade]
) -> dict[str, IntentAnnotation]:
    if (
        not labels
        or len(labels) > _MAX_TAPE
        or not all(isinstance(a, IntentAnnotation) for a in labels)
    ):
        raise ValueError("timestamped intent annotations are required")
    by_id = {a.trade_id: a for a in labels}
    tape_ids = {t.trade_id for t in trades}
    if len(by_id) != len(labels) or not set(by_id) <= tape_ids:
        raise ValueError("annotations must uniquely reference tape trade identities")
    for trade in trades:
        annotation = by_id.get(trade.trade_id)
        if annotation is not None and annotation.available_time < trade.event_time:
            raise ValueError("intent annotation cannot be available before the trade event")
    return by_id


def _logit(probabilities: Array) -> Array:
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(p / (1 - p)).reshape(-1, 1), dtype=float)


class OptionsFlowModel:
    def __init__(self, config: OptionsFlowConfig | None = None) -> None:
        self.config = config or OptionsFlowConfig()
        self.fit_info: FlowFitInfo | None = None
        self._scaler: StandardScaler | None = None
        self._linear: LogisticRegression | None = None
        self._trees: GradientBoostingClassifier | None = None
        self._calibrator: LogisticRegression | None = None
        self._history: tuple[OptionTrade, ...] = ()
        self._fit_config: OptionsFlowConfig | None = None

    def fit(
        self,
        trades: Sequence[OptionTrade],
        annotations: Sequence[IntentAnnotation],
        *,
        train_cutoff: datetime,
        cutoff: datetime,
    ) -> OptionsFlowModel:
        self.fit_info = None  # failed refit cannot expose stale confidence
        tape = _tape(trades)
        labels = _annotations(annotations, tape)
        train_stop, stop = _clock(train_cutoff, "train_cutoff"), _clock(cutoff, "cutoff")
        if train_stop >= stop:
            raise ValueError("training cutoff must precede calibration cutoff")
        training: list[tuple[TradeFeatures, IntentAnnotation]] = []
        calibration: list[tuple[TradeFeatures, IntentAnnotation]] = []
        history: dict[str, list[OptionTrade]] = {}
        for trade in tape:
            if trade.available_time > stop:
                continue
            past = history.setdefault(trade.contract_id, [])
            label = labels.get(trade.trade_id)
            if label is not None:
                phase_stop = train_stop if trade.available_time <= train_stop else stop
                if label.available_time <= phase_stop:
                    try:
                        features = trade_features(trade, past, self.config)
                    except ValueError as exc:
                        if "insufficient trailing" not in str(exc):
                            raise
                    else:
                        target = training if trade.available_time <= train_stop else calibration
                        target.append((features, label))
            past.append(trade)
        for name, rows, minimum in (("training", training, 24), ("calibration", calibration, 20)):
            if len(rows) < minimum or {a.intent for _, a in rows} != {"informed", "hedge"}:
                raise ValueError(f"{name} needs sufficient available annotations from both classes")
        x_train = np.asarray([f.values for f, _ in training])
        x_cal = np.asarray([f.values for f, _ in calibration])
        y_train = np.asarray([int(a.intent == "informed") for _, a in training], dtype=float)
        y_cal = np.asarray([int(a.intent == "informed") for _, a in calibration], dtype=float)
        scaler = _fit_training_fold_scaler(x_train)
        linear = LogisticRegression(max_iter=500, random_state=self.config.seed).fit(
            scaler.transform(x_train), y_train
        )
        trees = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.tree_depth,
            random_state=self.config.seed,
        ).fit(scaler.transform(x_train), y_train)
        raw = (
            linear.predict_proba(scaler.transform(x_cal))[:, 1]
            + trees.predict_proba(scaler.transform(x_cal))[:, 1]
        ) / 2
        calibrator = LogisticRegression(max_iter=500, random_state=self.config.seed).fit(
            _logit(raw), y_cal
        )
        calibrated = np.asarray(calibrator.predict_proba(_logit(raw))[:, 1], dtype=float)
        train_hash = _hash([{"feature": asdict(f), "annotation": asdict(a)} for f, a in training])
        cal_hash = _hash([{"feature": asdict(f), "annotation": asdict(a)} for f, a in calibration])
        tree_payload = [
            {
                "children_left": t.tree_.children_left.tolist(),
                "children_right": t.tree_.children_right.tolist(),
                "feature": t.tree_.feature.tolist(),
                "threshold": t.tree_.threshold.tolist(),
                "value": t.tree_.value.tolist(),
            }
            for t in trees.estimators_.ravel()
        ]
        model_hash = _hash(
            {
                "revision": 1,
                "config": asdict(self.config),
                "training": train_hash,
                "calibration": cal_hash,
                "train_cutoff": train_stop,
                "cutoff": stop,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "linear_coef": linear.coef_.tolist(),
                "linear_intercept": linear.intercept_.tolist(),
                "tree_init_class_prior": trees.init_.class_prior_.tolist(),
                "trees": tree_payload,
                "calibration_coef": calibrator.coef_.tolist(),
                "calibration_intercept": calibrator.intercept_.tolist(),
            }
        )
        self._scaler, self._linear, self._trees, self._calibrator = (
            scaler,
            linear,
            trees,
            calibrator,
        )
        self._history = tuple(t for t in tape if t.available_time <= stop)
        self._fit_config = self.config
        self.fit_info = FlowFitInfo(
            model_hash,
            train_hash,
            cal_hash,
            training[0][0].schema_sha256,
            train_stop.isoformat(),
            stop.isoformat(),
            len(training),
            len(calibration),
            tuple(sorted({a.annotation_source for _, a in training + calibration})),
            any(f.synthetic or a.synthetic for f, a in training + calibration),
            brier_score(calibrated, y_cal),
            log_loss(calibrated, y_cal),
            expected_calibration_error(calibrated, y_cal),
        )
        return self

    def score_trade(self, trade: OptionTrade, history: Sequence[OptionTrade] = ()) -> FlowScore:
        info = self.fit_info
        if info is None:
            return FlowScore(
                trade.trade_id,
                trade.available_time.isoformat(),
                "untrained",
                None,
                None,
                None,
                trade.source_sha256,
                None,
                None,
                trade.synthetic,
                None,
            )
        if self.config != self._fit_config:
            raise RuntimeError("options flow config changed; retraining is required")
        combined = {t.trade_id: t for t in self._history}
        for prior in _tape(history):
            existing = combined.get(prior.trade_id)
            if existing is not None and existing.source_sha256 != prior.source_sha256:
                raise ValueError("conflicting historical option trade identity")
            combined[prior.trade_id] = prior
        try:
            features = trade_features(trade, tuple(combined.values()), self.config)
        except ValueError as exc:
            return FlowScore(
                trade.trade_id,
                trade.available_time.isoformat(),
                str(exc),
                None,
                info.model_sha256,
                None,
                trade.source_sha256,
                info.feature_schema_sha256,
                info.synthetic,
                trade.synthetic,
                trade.available_time.isoformat() <= info.cutoff,
            )
        assert self._scaler is not None and self._linear is not None
        assert self._trees is not None and self._calibrator is not None
        x = self._scaler.transform(np.asarray([features.values]))
        raw = (self._linear.predict_proba(x)[:, 1] + self._trees.predict_proba(x)[:, 1]) / 2
        probability = float(self._calibrator.predict_proba(_logit(np.asarray(raw)))[:, 1][0])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise FloatingPointError("invalid calibrated flow probability")
        return FlowScore(
            trade.trade_id,
            trade.available_time.isoformat(),
            "scored",
            probability,
            info.model_sha256,
            features.feature_sha256,
            features.source_sha256,
            features.schema_sha256,
            info.synthetic,
            features.synthetic,
            trade.available_time.isoformat() <= info.cutoff,
            info.annotation_sources,
        )

    def evaluate(
        self,
        trades: Sequence[OptionTrade],
        annotations: Sequence[IntentAnnotation],
        *,
        label_cutoff: datetime,
    ) -> dict[str, Any]:
        info = self.fit_info
        if info is None:
            raise RuntimeError("options flow model is not trained")
        tape = _tape(trades)
        labels = _annotations(annotations, tape)
        stop = _clock(label_cutoff, "label_cutoff")
        fit_stop = datetime.fromisoformat(info.cutoff)
        if stop <= fit_stop or any(t.available_time <= fit_stop for t in tape):
            raise ValueError("holdout trades and label cutoff must follow the model fit cutoff")
        records: list[dict[str, Any]] = []
        probabilities: list[float] = []
        targets: list[int] = []
        for trade in tape:
            annotation = labels.get(trade.trade_id)
            if (
                annotation is None
                or annotation.available_time > stop
                or trade.available_time > stop
            ):
                continue
            score = self.score_trade(trade, tape)
            if score.probability_informed_annotation is None:
                raise ValueError(f"holdout score unavailable: {score.status}")
            probabilities.append(score.probability_informed_annotation)
            targets.append(int(annotation.intent == "informed"))
            records.append({"score": asdict(score), "annotation": asdict(annotation)})
        if len(records) < 10:
            raise ValueError("at least ten labeled holdout scores are required")
        p, y = np.asarray(probabilities), np.asarray(targets, dtype=float)
        return {
            "model_sha256": info.model_sha256,
            "holdout_sha256": _hash(records),
            "label_cutoff": stop.isoformat(),
            "n_scored": len(records),
            "brier_score": brier_score(p, y),
            "log_loss": log_loss(p, y),
            "ece": expected_calibration_error(p, y),
            "label_semantics": _SEMANTICS,
            "synthetic": info.synthetic
            or any(t.synthetic for t in tape)
            or any(a.synthetic for a in labels.values()),
            "research_only": True,
            "actual_intent_established": False,
            "live_pnl_claim": False,
            "sota_established": False,
        }


def alert_unusual(
    scores: Sequence[FlowScore], *, threshold: float = 0.9
) -> tuple[dict[str, Any], ...]:
    """Return local research alerts only; no notification or intent claim."""
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and in (0, 1)")
    if len(scores) > _MAX_TAPE:
        raise ValueError("alert batch exceeds offline resource bound")
    return tuple(
        {
            "trade_id": s.trade_id,
            "time": s.decision_time,
            "probability_informed_annotation": s.probability_informed_annotation,
            "threshold": threshold,
            "model_sha256": s.model_sha256,
            "feature_sha256": s.feature_sha256,
            "source_sha256": s.source_sha256,
            "feature_schema_sha256": s.feature_schema_sha256,
            "annotation_sources": s.annotation_sources,
            "training_synthetic": s.training_synthetic,
            "input_synthetic": s.input_synthetic,
            "synthetic": s.training_synthetic is True or s.input_synthetic,
            "in_sample": s.in_sample,
            "label_semantics": s.label_semantics,
            "research_only": True,
            "actual_intent_established": False,
            "live_pnl_claim": False,
        }
        for s in scores
        if s.status == "scored"
        and s.probability_informed_annotation is not None
        and s.probability_informed_annotation >= threshold
    )
