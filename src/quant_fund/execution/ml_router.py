"""Offline learned fill/toxicity predictions and capacity-constrained routing.

This is a research baseline, with no venue connections or order submission.
Binary fill probabilities use regularized logistic regression and realized
adverse-selection cost conditional on a fill uses ridge Gaussian regression.
Models train only on outcomes already published by ``asof``. Evaluation is
strictly later than the training cutoff and reports proper probability and
Gaussian scores. Synthetic order logs are correctness evidence only.

The allocation objective is a linear per-unit expected cost under frozen
candidate predictions: p_fill * (fee + crossing cost + expected toxicity)
+ (1-p_fill) * opportunity cost. Sorting marginal costs is the exact optimum
for this restricted divisible-order problem. Size-dependent fill, queue
interactions, auctions and nonlinear market impact are NOT modeled; this
baseline must not be presented as a general optimal multi-venue router.
The patent mentioned by the PDF is prior art, not a clearance or license.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from quant_fund.metrics.scoring import crps_gaussian, log_score_gaussian


def _fit_training_fold_scaler(training_fold: np.ndarray) -> StandardScaler:
    """Normalize only the finalized rows selected by the caller's cutoff."""
    return StandardScaler().fit(training_fold)


def _clock(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _finite(value: float, name: str, *, nonnegative: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or abs(result) > 1e12 or (nonnegative and result < 0):
        raise ValueError(f"{name} must be finite and within its numeric bounds")
    return result


@dataclass(frozen=True)
class FillFeatures:
    price_distance_bps: float
    spread_bps: float
    queue_ahead: float
    displayed_depth: float
    volatility_bps: float
    seconds_to_deadline: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _finite(value, name, nonnegative=name != "price_distance_bps")

    def vector(self) -> list[float]:
        return [float(value) for value in asdict(self).values()]


@dataclass(frozen=True)
class OrderObservation:
    order_id: str
    venue: str
    decision_time: datetime
    feature_available_time: datetime
    outcome_available_time: datetime
    features: FillFeatures
    filled: bool
    toxicity_bps: float | None
    source: str
    synthetic: bool

    def __post_init__(self) -> None:
        if any(
            not isinstance(x, str) or not x.strip()
            for x in (self.order_id, self.venue, self.source)
        ):
            raise ValueError("order_id, venue and source are required")
        decision = _clock(self.decision_time, "decision_time")
        feature_available = _clock(self.feature_available_time, "feature_available_time")
        available = _clock(self.outcome_available_time, "outcome_available_time")
        if feature_available > decision:
            raise ValueError("order features must be available by the decision time")
        if not isinstance(self.features, FillFeatures):
            raise ValueError("features must satisfy the FillFeatures contract")
        if available <= decision:
            raise ValueError("order outcome must become available strictly after decision")
        if not isinstance(self.filled, bool) or not isinstance(self.synthetic, bool):
            raise ValueError("filled and synthetic must be explicit booleans")
        if "synthetic" in self.source.lower() and not self.synthetic:
            raise ValueError("synthetic source cannot claim empirical observations")
        if self.filled:
            if self.toxicity_bps is None:
                raise ValueError("filled orders require observed toxicity")
            _finite(self.toxicity_bps, "toxicity_bps")
        elif self.toxicity_bps is not None:
            raise ValueError("unfilled orders cannot supply realized fill toxicity")
        object.__setattr__(self, "decision_time", decision)
        object.__setattr__(self, "feature_available_time", feature_available)
        object.__setattr__(self, "outcome_available_time", available)


@dataclass(frozen=True)
class FillPrediction:
    p_fill: float
    toxicity_mean_bps: float
    toxicity_scale_bps: float
    model_sha256: str


@dataclass(frozen=True)
class ChildOrderProposal:
    venue: str
    quote_available_time: datetime
    feature_available_time: datetime
    price: float
    capacity: int
    fee_bps: float
    crossing_cost_bps: float
    features: FillFeatures

    def __post_init__(self) -> None:
        if not isinstance(self.venue, str) or not self.venue.strip():
            raise ValueError("venue is required")
        object.__setattr__(
            self, "quote_available_time", _clock(self.quote_available_time, "quote_available_time")
        )
        if _finite(self.price, "price") <= 0:
            raise ValueError("price must be positive")
        object.__setattr__(
            self,
            "feature_available_time",
            _clock(self.feature_available_time, "feature_available_time"),
        )
        if not isinstance(self.features, FillFeatures):
            raise ValueError("features must satisfy the FillFeatures contract")
        if (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, int)
            or self.capacity < 0
        ):
            raise ValueError("capacity must be a nonnegative integer")
        _finite(self.fee_bps, "fee_bps")  # negative exchange rebates are legitimate
        _finite(self.crossing_cost_bps, "crossing_cost_bps", nonnegative=True)


@dataclass(frozen=True)
class RoutedChild:
    venue: str
    quantity: int
    price: float
    p_fill: float
    expected_toxicity_bps: float
    expected_cost_bps: float


@dataclass(frozen=True)
class RoutePlan:
    decision_time: str
    side: str
    parent_quantity: int
    children: tuple[RoutedChild, ...]
    expected_filled_quantity: float
    mean_expected_cost_bps: float
    model_sha256: str
    synthetic_training: bool
    objective: str = "frozen_per_unit_linear_cost"
    research_only: bool = True
    live_pnl_claim: bool = False


class MLFillRouter:
    """Fit on finalized past order logs; freeze parameters and provenance."""

    def __init__(self, *, regularization: float = 1.0, seed: int = 0) -> None:
        if _finite(regularization, "regularization") <= 0:
            raise ValueError("regularization must be positive")
        self.regularization = regularization
        self.seed = seed
        self._fitted = False

    def _matrix(
        self, rows: Sequence[tuple[str, FillFeatures]], venues: tuple[str, ...] | None = None
    ) -> np.ndarray:
        known_venues = self.venues if venues is None else venues
        if not rows or len(rows) > 100_000:
            raise ValueError("a bounded nonempty feature batch is required")
        if any(venue not in known_venues for venue, _ in rows):
            raise ValueError("unseen venue cannot use a calibrated training-venue prediction")
        return np.asarray(
            [
                features.vector() + [float(venue == known) for known in known_venues]
                for venue, features in rows
            ],
            dtype=float,
        )

    def _require_fit(self) -> None:
        if not self._fitted:
            raise ValueError("router is not fitted")
        if self._current_model_hash() != self._model_sha256:
            raise ValueError("model parameters or provenance changed after fitting")

    def _current_model_hash(self) -> str:
        payload = {
            "training_sha256": self.training_sha256,
            "train_asof": self.train_asof.isoformat(),
            "training_ids": sorted(self.training_ids),
            "source": self.source,
            "synthetic": self.synthetic_training,
            "venues": self.venues,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "scaler_with_mean": self.scaler.with_mean,
            "scaler_with_std": self.scaler.with_std,
            "fill_coef": self.fill.coef_.tolist(),
            "fill_intercept": self.fill.intercept_.tolist(),
            "fill_classes": self.fill.classes_.tolist(),
            "toxicity_coef": self.toxicity.coef_.tolist(),
            "toxicity_intercept": float(self.toxicity.intercept_),
            "toxicity_scale": self.toxicity_scale,
            "regularization": self.regularization,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    @property
    def model_sha256(self) -> str:
        self._require_fit()
        return self._model_sha256

    def update_models(
        self, observations: Sequence[OrderObservation], *, asof: datetime
    ) -> dict[str, Any]:
        cutoff = _clock(asof, "asof")
        if not observations or len(observations) > 100_000:
            raise ValueError("bounded nonempty order logs are required")
        if len({row.order_id for row in observations}) != len(observations):
            raise ValueError("duplicate order identities are rejected")
        # The same future log suffix cannot alter fitted model parameters.
        eligible = [row for row in observations if row.outcome_available_time <= cutoff]
        if len(eligible) < 16:
            raise ValueError("at least 16 finalized observations are required")
        if (
            len({row.synthetic for row in eligible}) != 1
            or len({row.source for row in eligible}) != 1
        ):
            raise ValueError("training cannot silently mix data sources or evidence classes")
        y = np.asarray([int(row.filled) for row in eligible])
        if np.unique(y).size != 2:
            raise ValueError("fill probability training requires both filled and unfilled outcomes")
        filled = np.flatnonzero(y)
        if len(filled) < 8:
            raise ValueError("at least eight filled orders are required for toxicity fitting")
        venues = tuple(sorted({row.venue for row in eligible}))
        x = self._matrix([(row.venue, row.features) for row in eligible], venues)
        scaler = _fit_training_fold_scaler(x)
        z = scaler.transform(x)
        fill = LogisticRegression(
            C=1 / self.regularization, max_iter=500, random_state=self.seed
        ).fit(z, y)
        targets = np.asarray([eligible[int(i)].toxicity_bps for i in filled], dtype=float)
        toxicity = Ridge(alpha=self.regularization).fit(z[filled], targets)
        scale = max(float(np.sqrt(np.mean((targets - toxicity.predict(z[filled])) ** 2))), 1e-6)
        if not np.all(np.isfinite(fill.coef_)) or not np.all(np.isfinite(toxicity.coef_)):
            raise ValueError("non-finite model parameters are rejected")
        self.scaler, self.fill, self.toxicity, self.toxicity_scale = scaler, fill, toxicity, scale
        self.venues = venues
        self.train_asof = cutoff
        self.synthetic_training = eligible[0].synthetic
        self.source = eligible[0].source
        self.training_ids = frozenset(row.order_id for row in eligible)
        training_payload = [
            {
                **asdict(row),
                "decision_time": row.decision_time.isoformat(),
                "outcome_available_time": row.outcome_available_time.isoformat(),
                "feature_available_time": row.feature_available_time.isoformat(),
            }
            for row in eligible
        ]
        self.training_sha256 = hashlib.sha256(
            json.dumps(training_payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        self._model_sha256 = self._current_model_hash()
        self._fitted = True
        return {
            "n_orders": len(eligible),
            "n_filled": len(filled),
            "train_asof": cutoff.isoformat(),
            "training_sha256": self.training_sha256,
            "model_sha256": self.model_sha256,
            "source": self.source,
            "synthetic": self.synthetic_training,
            "research_only": True,
            "live_pnl_claim": False,
        }

    def predict_fill(
        self, child_order: ChildOrderProposal, *, decision_time: datetime
    ) -> FillPrediction:
        self._require_fit()
        clock = _clock(decision_time, "decision_time")
        if clock < self.train_asof:
            raise ValueError("model training cutoff is later than the prediction decision")
        if child_order.quote_available_time > clock or child_order.feature_available_time > clock:
            raise ValueError("child quote/features are not available at the prediction decision")
        x = self.scaler.transform(self._matrix([(child_order.venue, child_order.features)]))
        return FillPrediction(
            float(self.fill.predict_proba(x)[0, 1]),
            float(self.toxicity.predict(x)[0]),
            self.toxicity_scale,
            self.model_sha256,
        )

    def route_order(
        self,
        proposals: Sequence[ChildOrderProposal],
        *,
        quantity: int,
        side: str,
        decision_time: datetime,
        opportunity_cost_bps: float,
        limit_price: float | None = None,
    ) -> RoutePlan:
        self._require_fit()
        clock = _clock(decision_time, "decision_time")
        if clock < self.train_asof:
            raise ValueError("model training cutoff is later than the routing decision")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise ValueError("parent quantity must be a positive integer")
        if side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        _finite(opportunity_cost_bps, "opportunity_cost_bps", nonnegative=True)
        if (
            not proposals
            or len(proposals) > 64
            or len({p.venue for p in proposals}) != len(proposals)
        ):
            raise ValueError("one candidate per venue and at most 64 candidates are required")
        if limit_price is not None and _finite(limit_price, "limit_price") <= 0:
            raise ValueError("limit price must be positive")
        scored = []
        for proposal in proposals:
            if proposal.quote_available_time > clock:
                raise ValueError("candidate quote is not available at the decision time")
            if proposal.feature_available_time > clock:
                raise ValueError("candidate features are not available at the decision time")
            if limit_price is not None and (
                (side == "buy" and proposal.price > limit_price)
                or (side == "sell" and proposal.price < limit_price)
            ):
                continue
            prediction = self.predict_fill(proposal, decision_time=clock)
            cost = (
                prediction.p_fill
                * (proposal.fee_bps + proposal.crossing_cost_bps + prediction.toxicity_mean_bps)
                + (1 - prediction.p_fill) * opportunity_cost_bps
            )
            scored.append((cost, proposal.venue, proposal, prediction))
        if sum(p.capacity for _, _, p, _ in scored) < quantity:
            raise ValueError("eligible venue capacity cannot cover the parent order")
        remaining = quantity
        children: list[RoutedChild] = []
        for cost, _, proposal, prediction in sorted(scored, key=lambda item: (item[0], item[1])):
            take = min(remaining, proposal.capacity)
            if take > 0:
                children.append(
                    RoutedChild(
                        proposal.venue,
                        take,
                        proposal.price,
                        prediction.p_fill,
                        prediction.toxicity_mean_bps,
                        cost,
                    )
                )
                remaining -= take
            if remaining == 0:
                break
        return RoutePlan(
            clock.isoformat(),
            side,
            quantity,
            tuple(children),
            sum(child.quantity * child.p_fill for child in children),
            sum(child.quantity * child.expected_cost_bps for child in children) / quantity,
            self.model_sha256,
            self.synthetic_training,
        )

    def evaluate(
        self, observations: Sequence[OrderObservation], *, asof: datetime
    ) -> dict[str, Any]:
        self._require_fit()
        clock = _clock(asof, "evaluation_asof")
        if not observations or any(
            row.order_id in self.training_ids
            or row.decision_time <= self.train_asof
            or row.outcome_available_time > clock
            for row in observations
        ):
            raise ValueError("evaluation requires unseen, strictly post-training, finalized orders")
        if len({row.order_id for row in observations}) != len(observations):
            raise ValueError("duplicate evaluation orders are rejected")
        if any(
            row.source != self.source or row.synthetic != self.synthetic_training
            for row in observations
        ):
            raise ValueError("evaluation source and evidence class must match training")
        x = self.scaler.transform(self._matrix([(row.venue, row.features) for row in observations]))
        p = self.fill.predict_proba(x)[:, 1]
        y = np.asarray([int(row.filled) for row in observations])
        likelihood = np.where(y == 1, p, 1 - p)
        if np.any(likelihood <= 0):
            raise ValueError("zero event likelihood cannot be reported as a finite log score")
        indices = np.flatnonzero(y)
        if not len(indices):
            raise ValueError("toxicity evaluation needs at least one filled order")
        target = np.asarray([observations[int(i)].toxicity_bps for i in indices], dtype=float)
        mu = self.toxicity.predict(x[indices])
        scales = np.full(target.shape, self.toxicity_scale)
        return {
            "n_orders": len(observations),
            "n_filled": len(indices),
            "fill_brier_score": float(np.mean((p - y) ** 2)),
            "fill_binary_log_score": float(-np.mean(np.log(likelihood))),
            "toxicity_crps": float(crps_gaussian(target, mu, scales).mean()),
            "toxicity_log_score": float(log_score_gaussian(target, mu, scales).mean()),
            "model_sha256": self.model_sha256,
            "source": self.source,
            "synthetic": self.synthetic_training,
            "research_only": True,
            "live_pnl_claim": False,
            "market_evidence": False,
        }
