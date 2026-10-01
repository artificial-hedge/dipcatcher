"""Publication-aware economic-release extraction and learned event impact.

This bounded research engine extracts explicit Actual/Consensus fields or a
small set of BLS/BEA-style phrases. It is a lexical NLP parser, not a pretrained
news model, a live feed, or an arbitrary-document extraction claim. Consensus
must come from a separately timed pre-release source. Revisions remain distinct
vintages; a later revision never replaces the value visible at a past decision.

NumPy ridge regression learns post-decision log returns and log realized
variance from economic surprises and past price context. A disjoint later
calibration set fits Gaussian bias/scale and a QLIKE-derived variance multiplier.
Its ratio is the unconstrained scale optimum for the fixed base forecasts;
subsequent declared forecast clipping can change that optimum.
The no-surprise regressor has the same training/calibration budget. Spike flags
are observed-price diagnostics against a threshold learned solely from past
training prices, not proof of economic-event causation.

Primary sources: https://www.bls.gov/schedule/news_release/cpi.htm gives intraday
release times; https://alfred.stlouisfed.org/help notes revisions may be added
within one business day. ALFRED vintage dates cannot establish intraday receipt
times. Ridge objective: https://scikit-learn.org/stable/modules/linear_model.html
#ridge-regression-and-classification. All input clocks/rights/labels are supplied
and independently unverified. Synthetic tests are never market evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.scoring import crps_gaussian

type Array = NDArray[np.float64]
type EventKind = Literal["cpi_yoy", "cpi_mom", "payroll_change", "gdp_annualized", "policy_rate"]
type Unit = Literal["fraction", "persons"]
_KINDS: tuple[EventKind, ...] = (
    "cpi_yoy",
    "cpi_mom",
    "payroll_change",
    "gdp_annualized",
    "policy_rate",
)
_SCHEMA = "economic_events.v1"
_MAX_BYTES = 16_000_000
_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_ALIAS = {
    "cpi_yoy": r"cpi\s*(?:yoy|year[- ]over[- ]year)|over the last 12 months",
    "cpi_mom": r"cpi\s*(?:mom|month[- ]over[- ]month)|consumer price index",
    "payroll_change": r"nonfarm payroll|payroll change",
    "gdp_annualized": r"gross domestic product|gdp\s*(?:annualized|annualised)",
    "policy_rate": r"federal funds|policy rate",
}
_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_SUFFIX = r"(?:%|percent|percentage points?|basis points?|bps|thousand|million|k|m|persons|jobs)"


def _json(value: Any) -> str:
    def encode(item: Any) -> str:
        if isinstance(item, datetime):
            return _clock(item, "serialized clock").isoformat()
        raise TypeError("unsupported snapshot value")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=encode)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _text(value: str, name: str, bound: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > bound:
        raise ValueError(f"{name} requires a bounded nonempty string")
    return value


def _sha(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("lowercase SHA256 source evidence required")
    return value


def _number(value: float, name: str, bound: float = 1e12) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or abs(value) > bound
    ):
        raise ValueError(f"{name} must be finite and bounded")
    return float(value)


def _count(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} outside integer resource bound")
    return value


def _bool(value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError("synthetic flag must be an explicit Boolean")


def _clock(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} requires an intraday timezone-aware datetime")
    utc = value.astimezone(UTC)
    if utc.astimezone(value.tzinfo).replace(tzinfo=None) != value.replace(tzinfo=None):
        raise ValueError(f"{name} is a nonexistent local datetime")
    return utc


def local_release_time(value: datetime, timezone: str, *, fold: int | None = None) -> datetime:
    """Convert an explicit local calendar time, rejecting missing/DST ambiguity.

    An ambiguous fall-back clock requires fold=0/1; a spring gap is rejected.
    Date-only ALFRED vintages must not be passed as fabricated midnight clocks.
    """
    if not isinstance(value, datetime) or value.tzinfo is not None:
        raise ValueError("an explicit naive local datetime and timezone are required")
    _text(timezone, "timezone")
    if fold is not None:
        _count(fold, "fold", 0, 1)
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("unknown calendar timezone") from exc
    candidates = [value.replace(tzinfo=zone, fold=f) for f in (0, 1)]
    valid = [
        c for c in candidates if c.astimezone(UTC).astimezone(zone).replace(tzinfo=None) == value
    ]
    if not valid:
        raise ValueError("nonexistent local calendar time")
    if len(valid) == 2 and valid[0].utcoffset() != valid[1].utcoffset() and fold is None:
        raise ValueError("ambiguous local calendar time requires explicit fold")
    selected = candidates[fold or 0]
    if selected not in valid:
        raise ValueError("invalid fold for local calendar time")
    return selected.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    event_id: str
    series_id: str
    kind: EventKind
    period: str
    scheduled_at: datetime
    published_at: datetime
    available_at: datetime
    source_sha256: str
    synthetic: bool

    def __post_init__(self) -> None:
        for name in ("event_id", "series_id", "period"):
            _text(getattr(self, name), name)
        if self.kind not in _KINDS:
            raise ValueError("unknown economic event kind")
        if (
            not _clock(self.published_at, "calendar publication")
            <= _clock(self.available_at, "calendar receipt")
            <= _clock(self.scheduled_at, "scheduled release")
        ):
            raise ValueError("calendar must be published/received before its scheduled release")
        _sha(self.source_sha256)
        _bool(self.synthetic)

    @property
    def unit(self) -> Unit:
        return "persons" if self.kind == "payroll_change" else "fraction"


@dataclass(frozen=True, slots=True)
class ReleaseVintage:
    event_id: str
    vintage_id: str
    revision: int
    value: float
    unit: Unit
    released_at: datetime
    available_at: datetime
    source_sha256: str
    text_sha256: str
    synthetic: bool

    def __post_init__(self) -> None:
        _text(self.event_id, "event_id")
        _text(self.vintage_id, "vintage_id")
        _count(self.revision, "revision", 0, 128)
        _number(self.value, "release value", 1e8 if self.unit == "persons" else 10)
        if self.unit not in ("fraction", "persons"):
            raise ValueError("unknown release unit")
        if self.unit == "persons" and not float(self.value).is_integer():
            raise ValueError("reported payroll counts must be integral persons")
        if _clock(self.released_at, "release publication") > _clock(
            self.available_at, "release receipt"
        ):
            raise ValueError("release cannot be received before publication")
        for digest in (self.source_sha256, self.text_sha256):
            _sha(digest)
        _bool(self.synthetic)


@dataclass(frozen=True, slots=True)
class ConsensusForecast:
    event_id: str
    forecast_id: str
    value: float
    unit: Unit
    published_at: datetime
    available_at: datetime
    source_sha256: str
    synthetic: bool

    def __post_init__(self) -> None:
        _text(self.event_id, "event_id")
        _text(self.forecast_id, "forecast_id")
        _number(self.value, "consensus value", 1e8 if self.unit == "persons" else 10)
        if self.unit not in ("fraction", "persons"):
            raise ValueError("unknown consensus unit")
        if _clock(self.published_at, "consensus publication") > _clock(
            self.available_at, "consensus receipt"
        ):
            raise ValueError("consensus cannot precede its publication")
        _sha(self.source_sha256)
        _bool(self.synthetic)


def _extracted_value(number: str, suffix: str, unit: Unit) -> float:
    value = float(number.replace(",", ""))
    suffix = suffix.lower()
    if unit == "fraction":
        if suffix in ("%", "percent", "percentage point", "percentage points"):
            return value / 100
        if suffix in ("bps", "basis point", "basis points"):
            return value / 10000
        raise ValueError("percent/basis-point unit is mandatory for rate releases")
    if suffix in ("k", "thousand"):
        return value * 1000
    if suffix in ("m", "million"):
        return value * 1e6
    if suffix in ("persons", "jobs", ""):
        return value
    raise ValueError("payroll extraction requires explicit count units")


def extract_release(
    text: str,
    event: CalendarEvent,
    *,
    vintage_id: str,
    revision: int,
    released_at: datetime,
    available_at: datetime,
    source_sha256: str,
    synthetic: bool,
) -> ReleaseVintage:
    """Extract one declared event; uncertain/duplicate fields fail closed.

    Supports explicit `Actual: 3.2%` fields plus selected official-style CPI,
    payroll/GDP phrases. It never extracts consensus from post-release prose.
    Other event types/documents require a new explicitly tested grammar.
    """
    _text(text, "release text", 16384)
    if not isinstance(event, CalendarEvent) or re.search(_ALIAS[event.kind], text, re.I) is None:
        raise ValueError("release text does not identify the declared event kind")
    # A percent sign is not a word character: use a lookahead rather than \b.
    matches = list(
        re.finditer(
            rf"\b(?:actual|reported)\s*[:=]\s*({_NUMBER})\s*({_SUFFIX})?(?=\s|[;)]|[,.](?!\d)|$)",
            text,
            re.I,
        )
    )
    if len(matches) > 1:
        raise ValueError("ambiguous duplicate Actual fields")
    if matches:
        explicit_aliases = {
            **_ALIAS,
            "cpi_mom": r"cpi\s*(?:mom|month[- ]over[- ]month)|month[- ]over[- ]month|one[- ]month|monthly consumer price index",
        }
        identified = {
            kind for kind, alias in explicit_aliases.items() if re.search(alias, text, re.I)
        }
        if identified != {event.kind}:
            raise ValueError(
                "explicit Actual field needs an unambiguous event/rate-period identity"
            )
        if not matches[0].group(2):
            raise ValueError("explicit Actual fields require a declared numeric unit")
    negative = False
    if not matches:
        patterns = {
            "cpi_yoy": rf"over the last 12 months[^.\n]{{0,240}}?\b(increased|rose|decreased|declined)\s+({_NUMBER})\s+(percent)",
            "cpi_mom": rf"consumer price index[^.\n]{{0,240}}?\b(increased|rose|decreased|declined)\s+({_NUMBER})\s+(percent)\s+on a seasonally adjusted basis",
            "payroll_change": rf"nonfarm payroll[^.\n]{{0,180}}?\b(increased|rose|decreased|declined)(?:\s+by)?\s+({_NUMBER})\s*({_SUFFIX})?(?=\s|[;)]|[,.](?!\d)|$)",
            "gdp_annualized": rf"gross domestic product[^.\n]{{0,180}}?\b(increased|rose|decreased|declined)\s+at an annual rate of\s+({_NUMBER})\s+(percent)",
        }
        pattern = patterns.get(event.kind)
        found = list(re.finditer(pattern, text, re.I)) if pattern else []
        if len(found) != 1:
            raise ValueError("unsupported or ambiguous release language")
        verb, number, suffix = found[0].groups()
        negative = verb.lower() in ("decreased", "declined")
    else:
        number, suffix = matches[0].groups()
    value = _extracted_value(number, suffix or "", event.unit)
    if negative:
        if value < 0:
            raise ValueError("ambiguous double-negative release value")
        value = -value
    return ReleaseVintage(
        event.event_id,
        vintage_id,
        revision,
        value,
        event.unit,
        released_at,
        available_at,
        source_sha256,
        hashlib.sha256(text.encode()).hexdigest(),
        synthetic,
    )


@dataclass(frozen=True, slots=True)
class TimedPrice:
    record_id: str
    timestamp: datetime
    available_at: datetime
    price: float
    source_sha256: str
    synthetic: bool

    def __post_init__(self) -> None:
        _text(self.record_id, "price record_id")
        if _clock(self.timestamp, "price timestamp") > _clock(self.available_at, "price receipt"):
            raise ValueError("price receipt precedes event time")
        if _number(self.price, "price", 1e10) <= 0:
            raise ValueError("prices must be positive")
        _sha(self.source_sha256)
        _bool(self.synthetic)


def _prices(prices: tuple[TimedPrice, ...]) -> None:
    if (
        not isinstance(prices, tuple)
        or not 3 <= len(prices) <= 256
        or not all(isinstance(p, TimedPrice) for p in prices)
    ):
        raise ValueError("3..256 immutable typed price records required")
    if len({p.record_id for p in prices}) != len(prices) or any(
        _clock(a.timestamp, "price") >= _clock(b.timestamp, "price")
        for a, b in zip(prices, prices[1:], strict=False)
    ):
        raise ValueError("price identities/times must be unique and strictly ordered")


@dataclass(frozen=True, slots=True)
class EventObservation:
    record_id: str
    asset_id: str
    calendar: CalendarEvent
    releases: tuple[ReleaseVintage, ...]
    consensus: tuple[ConsensusForecast, ...]
    past_prices: tuple[TimedPrice, ...]
    decision_at: datetime

    def __post_init__(self) -> None:
        _text(self.record_id, "record_id")
        _text(self.asset_id, "asset_id")
        _clock(self.decision_at, "decision")
        if not isinstance(self.calendar, CalendarEvent):
            raise ValueError("typed calendar required")
        if (
            not isinstance(self.releases, tuple)
            or not 1 <= len(self.releases) <= 129
            or not all(isinstance(r, ReleaseVintage) for r in self.releases)
        ):
            raise ValueError("bounded typed release history required")
        if (
            not isinstance(self.consensus, tuple)
            or not 1 <= len(self.consensus) <= 128
            or not all(isinstance(c, ConsensusForecast) for c in self.consensus)
        ):
            raise ValueError("bounded typed consensus history required")
        records: tuple[ReleaseVintage | ConsensusForecast, ...] = (*self.releases, *self.consensus)
        for record in records:
            if record.event_id != self.calendar.event_id or record.unit != self.calendar.unit:
                raise ValueError("event identities/units must align")
        revisions = sorted(self.releases, key=lambda r: r.revision)
        if (
            revisions[0].revision != 0
            or len({r.revision for r in revisions}) != len(revisions)
            or len({r.vintage_id for r in revisions}) != len(revisions)
            or any(
                _clock(a.released_at, "release") >= _clock(b.released_at, "release")
                for a, b in zip(revisions, revisions[1:], strict=False)
            )
        ):
            raise ValueError(
                "release vintages require initial revision 0 and unique ordered publication clocks"
            )
        if len({c.forecast_id for c in self.consensus}) != len(self.consensus):
            raise ValueError("duplicate consensus identity")
        _prices(self.past_prices)


@dataclass(frozen=True, slots=True)
class EventOutcome:
    observation: EventObservation
    log_return: float
    realized_variance: float
    target_end: datetime
    available_at: datetime
    source_sha256: str
    synthetic: bool

    def __post_init__(self) -> None:
        if not isinstance(self.observation, EventObservation):
            raise ValueError("typed observation required")
        _number(self.log_return, "target log return", 10)
        if _number(self.realized_variance, "realized variance", 100) < 0:
            raise ValueError("realized variance must be nonnegative")
        if (
            not _clock(self.observation.decision_at, "decision")
            < _clock(self.target_end, "target end")
            <= _clock(self.available_at, "label receipt")
        ):
            raise ValueError("target horizon/label availability invalid")
        _sha(self.source_sha256)
        _bool(self.synthetic)


@dataclass(frozen=True, slots=True)
class EventConfig:
    asset_id: str
    kinds: tuple[EventKind, ...] = _KINDS
    horizon_seconds: int = 3600
    embargo_seconds: int = 3600
    release_delay_seconds: int = 0
    consensus_delay_seconds: int = 0
    price_delay_seconds: int = 0
    target_interval_seconds: int = 300
    context_interval_seconds: int = 60
    ridge_alpha: float = 1.0
    min_sigma: float = 1e-6
    variance_floor: float = 1e-12
    spike_quantile: float = 0.99

    def __post_init__(self) -> None:
        _text(self.asset_id, "asset_id")
        if (
            not isinstance(self.kinds, tuple)
            or not 1 <= len(self.kinds) <= 5
            or len(set(self.kinds)) != len(self.kinds)
            or any(k not in _KINDS for k in self.kinds)
        ):
            raise ValueError("1..5 declared unique event kinds required")
        _count(self.horizon_seconds, "horizon", 1, 86400)
        _count(self.target_interval_seconds, "target price interval", 1, 86400)
        if (
            self.horizon_seconds % self.target_interval_seconds
            or not 2 <= self.horizon_seconds // self.target_interval_seconds <= 255
        ):
            raise ValueError("target horizon needs 2..255 declared equal sampling intervals")
        _count(self.context_interval_seconds, "context/spike price interval", 1, 86400)
        if (
            self.horizon_seconds % self.context_interval_seconds
            or not 2 <= self.horizon_seconds // self.context_interval_seconds <= 255
        ):
            raise ValueError("spike horizon needs 2..255 declared equal sampling intervals")
        for name in (
            "embargo_seconds",
            "release_delay_seconds",
            "consensus_delay_seconds",
            "price_delay_seconds",
        ):
            _count(getattr(self, name), name, 0, 86400)
        for name in ("ridge_alpha", "min_sigma", "variance_floor"):
            if (
                not 0
                < _number(getattr(self, name), name, 1e4)
                <= (1 if name != "ridge_alpha" else 1e4)
            ):
                raise ValueError("positive bounded regularization/floors required")
        if not 0.5 <= _number(self.spike_quantile, "spike quantile", 1) < 1:
            raise ValueError("spike quantile must be in [0.5, 1)")


def _snapshot(observation: EventObservation, config: EventConfig) -> dict[str, Any]:
    if observation.asset_id != config.asset_id or observation.calendar.kind not in config.kinds:
        raise ValueError("asset or unknown event-kind mismatch")
    decision = _clock(observation.decision_at, "decision")
    calendar = observation.calendar
    if _clock(calendar.available_at, "calendar receipt") > decision:
        raise ValueError("calendar not available at decision")
    initial = min(observation.releases, key=lambda r: r.revision)
    first_release = _clock(initial.released_at, "initial release")
    visible = sorted(
        (
            r
            for r in observation.releases
            if _clock(r.available_at, "release receipt")
            + timedelta(seconds=config.release_delay_seconds)
            <= decision
        ),
        key=lambda r: r.revision,
    )
    if not visible or visible[0].revision != 0:
        raise ValueError("initial release not yet delivered")
    if decision - first_release > timedelta(days=7):
        raise ValueError("event decision exceeds bounded release window")
    eligible = [
        c
        for c in observation.consensus
        if _clock(c.available_at, "consensus receipt")
        + timedelta(seconds=config.consensus_delay_seconds)
        < first_release
    ]
    if not eligible:
        raise ValueError("independently timed pre-release consensus unavailable")
    latest_clock = max(_clock(c.available_at, "consensus") for c in eligible)
    latest = [c for c in eligible if _clock(c.available_at, "consensus") == latest_clock]
    if len(latest) != 1:
        raise ValueError("consensus ordering ambiguous at identical receipt time")
    consensus = latest[0]
    release = visible[-1]
    prior = visible[-2] if len(visible) > 1 else release
    prices = tuple(
        p
        for p in observation.past_prices
        if _clock(p.timestamp, "price") < first_release
        and _clock(p.available_at, "price receipt") + timedelta(seconds=config.price_delay_seconds)
        <= decision
    )
    if len(prices) < 3:
        raise ValueError("insufficient pre-release delivered price history")
    if any(
        _clock(b.timestamp, "past price") - _clock(a.timestamp, "past price")
        != timedelta(seconds=config.context_interval_seconds)
        for a, b in zip(prices, prices[1:], strict=False)
    ):
        raise ValueError("pre-release prices must use the declared context sampling interval")
    returns = np.diff(np.log(np.array([p.price for p in prices], dtype=float)))
    if np.any(np.abs(returns) > 10):
        raise ValueError("past log-return bound exceeded")
    surprise = release.value - consensus.value
    kinds = [float(calendar.kind == kind) for kind in config.kinds]
    features = [
        surprise,
        surprise**2,
        release.value - prior.value,
        (decision - _clock(release.released_at, "release")).total_seconds() / 86400,
        math.log(max(float(np.mean(returns**2)), config.variance_floor)),
        float(np.sum(returns)),
        *kinds,
        *(surprise * k for k in kinds),
    ]
    payload = {
        "record_id": observation.record_id,
        "event_id": calendar.event_id,
        "asset_id": observation.asset_id,
        "kind": calendar.kind,
        "decision_at": decision.isoformat(),
        "calendar": asdict(calendar),
        "visible_releases": [asdict(r) for r in visible],
        "consensus": asdict(consensus),
        "past_prices": [asdict(p) for p in prices],
        "features": features,
        "past_returns": returns.tolist(),
        "synthetic": calendar.synthetic
        or consensus.synthetic
        or any(r.synthetic for r in visible)
        or any(p.synthetic for p in prices),
    }
    return cast(dict[str, Any], json.loads(_json(payload)))


def _row(outcome: EventOutcome, config: EventConfig, asof: datetime) -> dict[str, Any]:
    snapshot = _snapshot(outcome.observation, config)
    if _clock(outcome.target_end, "target end") != _clock(
        outcome.observation.decision_at, "decision"
    ) + timedelta(seconds=config.horizon_seconds):
        raise ValueError("fixed target horizon mismatch")
    if _clock(outcome.available_at, "label receipt") > _clock(asof, "fit/evaluation asof"):
        raise ValueError("target label not yet published")
    # For N interval log returns, (sum r_i)^2 <= N * sum(r_i^2).
    n_intervals = config.horizon_seconds // config.target_interval_seconds
    squared_return = outcome.log_return**2
    variance_bound = n_intervals * outcome.realized_variance
    tolerance = 64 * np.finfo(float).eps * max(squared_return, variance_bound, np.finfo(float).tiny)
    if squared_return > variance_bound + tolerance:
        raise ValueError("incoherent return/realized-variance labels for declared sampling grid")
    return {
        "snapshot": snapshot,
        "log_return": outcome.log_return,
        "realized_variance": outcome.realized_variance,
        "target_end": _clock(outcome.target_end, "target end").isoformat(),
        "available_at": _clock(outcome.available_at, "label receipt").isoformat(),
        "source_sha256": outcome.source_sha256,
        "synthetic": snapshot["synthetic"] or outcome.synthetic,
    }


def _dataset(
    rows: Sequence[EventOutcome], config: EventConfig, asof: datetime, low: int
) -> list[dict[str, Any]]:
    if (
        isinstance(rows, (str, bytes))
        or not low <= len(rows) <= 4096
        or not all(isinstance(r, EventOutcome) for r in rows)
    ):
        raise ValueError("event dataset resource/type bound exceeded")
    if (
        sum(
            len(r.observation.releases)
            + len(r.observation.consensus)
            + len(r.observation.past_prices)
            for r in rows
        )
        > 32768
    ):
        raise ValueError("event dataset source-record resource bound exceeded")
    if len({r.observation.record_id for r in rows}) != len(rows) or len(
        {r.observation.calendar.event_id for r in rows}
    ) != len(rows):
        raise ValueError("rows/events must be unique; revisions cannot cross dataset rows")
    return [_row(r, config, asof) for r in rows]


def measure_reaction(
    observation: EventObservation,
    prices: tuple[TimedPrice, ...],
    *,
    config: EventConfig,
    asof: datetime,
) -> EventOutcome:
    """Measure a complete delivered price response on the declared sampling grid.

    Return is log(last/decision price); realized variance is the unannualized
    sum of squared interval log returns. Price/source completeness and release
    clocks remain externally attested. No future price enters decision inputs.
    """
    if not isinstance(observation, EventObservation) or not isinstance(config, EventConfig):
        raise ValueError("typed observation/config required")
    snapshot = _snapshot(observation, config)
    _prices(prices)
    asof = _clock(asof, "reaction measurement asof")
    start = _clock(observation.decision_at, "decision")
    end = start + timedelta(seconds=config.horizon_seconds)
    interval = timedelta(seconds=config.target_interval_seconds)
    expected = config.horizon_seconds // config.target_interval_seconds + 1
    if len(prices) != expected or any(
        _clock(p.timestamp, "reaction price") != start + i * interval for i, p in enumerate(prices)
    ):
        raise ValueError("complete declared decision-to-horizon sampling grid required")
    available = max(
        _clock(p.available_at, "reaction price receipt")
        + timedelta(seconds=config.price_delay_seconds)
        for p in prices
    )
    if available > asof:
        raise ValueError("reaction prices not yet delivered")
    returns = np.diff(np.log(np.array([p.price for p in prices])))
    if np.any(np.abs(returns) > 10):
        raise ValueError("reaction interval return bound exceeded")
    return EventOutcome(
        observation,
        float(np.sum(returns)),
        float(np.sum(returns**2)),
        end,
        available,
        _hash([asdict(p) for p in prices]),
        snapshot["synthetic"] or any(p.synthetic for p in prices),
    )


def _ridge(design: Array, target: Array, alpha: float) -> Array:
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def _design(x: Array, mean: Array, scale: Array, kinds: int, no_surprise: bool) -> Array:
    transformed = (x - mean) / scale
    if no_surprise:
        transformed[:, :3] = 0
        transformed[:, 6 + kinds :] = 0
    return np.column_stack((np.ones(len(x)), transformed))


def _learn(
    training: list[dict[str, Any]], calibration: list[dict[str, Any]], config: EventConfig
) -> dict[str, Any]:
    x = np.asarray([r["snapshot"]["features"] for r in training], dtype=float)
    xc = np.asarray([r["snapshot"]["features"] for r in calibration], dtype=float)
    y = np.array([r["log_return"] for r in training], dtype=float)
    v = np.array([r["realized_variance"] for r in training], dtype=float)
    yc = np.array([r["log_return"] for r in calibration], dtype=float)
    vc = np.array([r["realized_variance"] for r in calibration], dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    arms = {}
    for no_surprise in (False, True):
        z = _design(x, mean, scale, len(config.kinds), no_surprise)
        zc = _design(xc, mean, scale, len(config.kinds), no_surprise)
        beta = _ridge(z, y, config.ridge_alpha)
        variance_beta = _ridge(z, np.log(np.maximum(v, config.variance_floor)), config.ridge_alpha)
        bias = float(np.mean(yc - zc @ beta))
        sigma = max(float(np.sqrt(np.mean((yc - zc @ beta - bias) ** 2))), config.min_sigma)
        base_variance = np.exp(
            np.clip(zc @ variance_beta, math.log(config.variance_floor), math.log(100))
        )
        multiplier = max(float(np.mean(vc / base_variance)), config.variance_floor)
        arms["no_surprise" if no_surprise else "surprise"] = {
            "beta": beta.tolist(),
            "variance_beta": variance_beta.tolist(),
            "bias": bias,
            "sigma": sigma,
            "variance_multiplier": multiplier,
        }
    past = [v for r in training for v in r["snapshot"]["past_returns"]]
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "arms": arms,
        "spike_threshold": max(
            float(np.quantile(np.abs(past), config.spike_quantile)), config.min_sigma
        ),
        "variance_floor_count": int(np.sum(v < config.variance_floor)),
    }


@dataclass(frozen=True, slots=True)
class EventPrediction:
    record_id: str
    event_id: str
    decision_at: datetime
    target_end: datetime
    mean_log_return: float
    sigma_log_return: float
    realized_variance_forecast: float
    method: Literal["surprise", "no_surprise"]
    model_sha256: str
    input_sha256: str
    synthetic: bool


class EconomicEventModel:
    """Frozen CPU event regressors; saved models replay their numeric fitting.

    Embedded causal snapshots bind exact training/calibration inputs. Loading
    rechecks source/schema/clocks, preprocessing, both fits, calibration and
    threshold. Self-contained hashes do not authenticate source provenance.
    """

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = cast(dict[str, Any], json.loads(_json(body)))
        self.model_sha256 = _hash(self._body)
        self.config = EventConfig(
            **{**self._body["config"], "kinds": tuple(self._body["config"]["kinds"])}
        )

    @classmethod
    def fit(
        cls,
        training: Sequence[EventOutcome],
        calibration: Sequence[EventOutcome],
        *,
        config: EventConfig,
        train_asof: datetime,
        calibration_asof: datetime,
    ) -> EconomicEventModel:
        if not isinstance(config, EventConfig):
            raise ValueError("typed event config required")
        train_clock = _clock(train_asof, "train asof")
        cal_clock = _clock(calibration_asof, "calibration asof")
        train = _dataset(training, config, train_clock, 16)
        cal = _dataset(calibration, config, cal_clock, 8)
        if any(sum(r["snapshot"]["kind"] == kind for r in train) < 4 for kind in config.kinds):
            raise ValueError(
                "each declared event kind requires at least four observed training events"
            )
        if train_clock >= cal_clock or min(
            _clock(r.observation.decision_at, "cal decision") for r in calibration
        ) <= train_clock + timedelta(seconds=config.embargo_seconds):
            raise ValueError("calibration must be chronologically embargoed after training asof")
        train_ids = {r.observation.record_id for r in training}
        train_events = {r.observation.calendar.event_id for r in training}
        if train_ids & {r.observation.record_id for r in calibration} or train_events & {
            r.observation.calendar.event_id for r in calibration
        }:
            raise ValueError("training/calibration rows/events must be disjoint")
        body = {
            "schema": _SCHEMA,
            "config": asdict(config),
            "train_asof": train_clock.isoformat(),
            "calibration_asof": cal_clock.isoformat(),
            "training": train,
            "calibration": cal,
            "learned": _learn(train, cal, config),
            "training_data_sha256": _hash(train),
            "calibration_data_sha256": _hash(cal),
            "implementation_sha256": _IMPLEMENTATION_SHA256,
            "numpy_version": np.__version__,
            "synthetic": any(r["synthetic"] for r in (*train, *cal)),
            "research_only": True,
            "market_evidence": False,
            "authenticated_sources": False,
            "causal_effect_identified": False,
        }
        if len(_json(body).encode()) > _MAX_BYTES:
            raise ValueError("model snapshot resource bound exceeded")
        return cls(body)

    def _seal(self) -> None:
        if _hash(self._body) != self.model_sha256 or asdict(self.config) != {
            **self._body["config"],
            "kinds": tuple(self._body["config"]["kinds"]),
        }:
            raise ValueError("frozen model/config seal mismatch")

    def predict(
        self,
        observation: EventObservation,
        *,
        method: Literal["surprise", "no_surprise"] = "surprise",
    ) -> EventPrediction:
        self._seal()
        return self._prediction(observation, method=method)

    def _prediction(
        self, observation: EventObservation, *, method: Literal["surprise", "no_surprise"]
    ) -> EventPrediction:
        if method not in ("surprise", "no_surprise") or not isinstance(
            observation, EventObservation
        ):
            raise ValueError("typed observation and declared method required")
        if _clock(observation.decision_at, "decision") <= datetime.fromisoformat(
            self._body["calibration_asof"]
        ) + timedelta(seconds=self.config.embargo_seconds):
            raise ValueError("prediction must follow frozen calibration and embargo")
        if observation.record_id in {
            r["snapshot"]["record_id"]
            for r in (*self._body["training"], *self._body["calibration"])
        } or observation.calendar.event_id in {
            r["snapshot"]["event_id"] for r in (*self._body["training"], *self._body["calibration"])
        }:
            raise ValueError("prediction row/event overlaps fitted evidence")
        snapshot = _snapshot(observation, self.config)
        learned = self._body["learned"]
        z = _design(
            np.array([snapshot["features"]], dtype=float),
            np.array(learned["mean"]),
            np.array(learned["scale"]),
            len(self.config.kinds),
            method == "no_surprise",
        )
        arm = learned["arms"][method]
        mean = float((z @ np.array(arm["beta"]))[0] + arm["bias"])
        variance = float(
            np.exp(
                np.clip(
                    (z @ np.array(arm["variance_beta"]))[0],
                    math.log(self.config.variance_floor),
                    math.log(100),
                )
            )
            * arm["variance_multiplier"]
        )
        variance = min(100.0, max(self.config.variance_floor, variance))
        if not math.isfinite(mean) or abs(mean) > 10:
            raise ValueError("forecast mean extrapolation outside bounded domain")
        return EventPrediction(
            observation.record_id,
            observation.calendar.event_id,
            _clock(observation.decision_at, "decision"),
            _clock(observation.decision_at, "decision")
            + timedelta(seconds=self.config.horizon_seconds),
            mean,
            arm["sigma"],
            variance,
            method,
            self.model_sha256,
            _hash(snapshot),
            self._body["synthetic"] or snapshot["synthetic"],
        )

    def evaluate(self, holdout: Sequence[EventOutcome], *, asof: datetime) -> dict[str, Any]:
        self._seal()
        rows = _dataset(holdout, self.config, asof, 8)
        y = np.array([r.log_return for r in holdout], dtype=float)
        variance = np.array([r.realized_variance for r in holdout], dtype=float)
        outcomes = []
        for method in ("surprise", "no_surprise"):
            predictions = [self._prediction(row.observation, method=method) for row in holdout]
            mean = np.array([p.mean_log_return for p in predictions])
            sigma = np.array([p.sigma_log_return for p in predictions])
            forecast_variance = np.array([p.realized_variance_forecast for p in predictions])
            outcomes.append(
                {
                    "method": method,
                    "gaussian_crps": float(np.mean(crps_gaussian(y, mean, sigma))),
                    "gaussian_negative_log_score": float(
                        np.mean(
                            np.log(sigma)
                            + 0.5 * math.log(2 * math.pi)
                            + 0.5 * ((y - mean) / sigma) ** 2
                        )
                    ),
                    "variance_qlike_unnormalized": float(
                        np.mean(np.log(forecast_variance) + variance / forecast_variance)
                    ),
                    "return_r2_diagnostic": _r2(y, mean),
                    "variance_r2_diagnostic": _r2(variance, forecast_variance),
                    "forecast_sha256": _hash([asdict(p) for p in predictions]),
                }
            )
        return {
            "outcomes": outcomes,
            "rows": len(rows),
            "holdout_data_sha256": _hash(rows),
            "model_sha256": self.model_sha256,
            "synthetic": self._body["synthetic"] or any(r["synthetic"] for r in rows),
            "research_only": True,
            "market_evidence": False,
            "r2_is_diagnostic": True,
            "qlike_definition": "log(forecast variance)+realized variance/forecast variance; outcome-only constant omitted; zero realized variance retained",
            "comparison": "same ridge feature dimension and training/calibration rows; surprise terms zeroed only in no_surprise arm",
            "causal_effect_identified": False,
        }

    def detect_spikes(
        self, observation: EventObservation, prices: tuple[TimedPrice, ...], *, asof: datetime
    ) -> dict[str, Any]:
        """Observed post-decision changes, never a future-price feature."""
        prediction = self.predict(observation)
        _prices(prices)
        asof = _clock(asof, "spike observation asof")
        start = _clock(observation.decision_at, "decision")
        end = start + timedelta(seconds=self.config.horizon_seconds)
        visible = [
            p
            for p in prices
            if start <= _clock(p.timestamp, "price") <= end
            and _clock(p.available_at, "price receipt")
            + timedelta(seconds=self.config.price_delay_seconds)
            <= asof
        ]
        if len(visible) < 2 or _clock(visible[0].timestamp, "first post price") != start:
            raise ValueError("a delivered decision-time anchor and subsequent price are required")
        if any(
            _clock(b.timestamp, "spike price") - _clock(a.timestamp, "spike price")
            != timedelta(seconds=self.config.context_interval_seconds)
            for a, b in zip(visible, visible[1:], strict=False)
        ):
            raise ValueError("spike prices must use the same declared interval as training context")
        returns = np.diff(np.log(np.array([p.price for p in visible])))
        threshold = self._body["learned"]["spike_threshold"]
        return {
            "spike": bool(np.any(np.abs(returns) > threshold)),
            "spike_threshold": threshold,
            "observed_through": _clock(visible[-1].timestamp, "observed end").isoformat(),
            "complete_horizon": _clock(visible[-1].timestamp, "observed end") == end,
            "visible_price_sha256": _hash([asdict(p) for p in visible]),
            "synthetic": prediction.synthetic or any(p.synthetic for p in visible),
            "causal_effect_identified": False,
            "market_evidence": False,
        }

    def metadata(self) -> dict[str, Any]:
        self._seal()
        return {
            "model_sha256": self.model_sha256,
            "training_data_sha256": self._body["training_data_sha256"],
            "calibration_data_sha256": self._body["calibration_data_sha256"],
            "implementation_sha256": self._body["implementation_sha256"],
            "config_sha256": _hash(self._body["config"]),
            "training_rows": len(self._body["training"]),
            "calibration_rows": len(self._body["calibration"]),
            "train_asof": self._body["train_asof"],
            "calibration_asof": self._body["calibration_asof"],
            "synthetic": self._body["synthetic"],
            "research_only": True,
            "market_evidence": False,
            "authenticated_sources": False,
            "causal_effect_identified": False,
            "parser": "bounded lexical extraction, not general news NLP",
            "variance_training": "ridge squared error on log(max(realized variance, declared floor)); QLIKE-derived mean(realized/base) ratio calibration is an unconstrained scale optimum before final floor/cap clipping",
            "spike_threshold": self._body["learned"]["spike_threshold"],
            "zero_variance_training_floor_count": self._body["learned"]["variance_floor_count"],
            "variance_prediction_bounds": [self.config.variance_floor, 100.0],
            "realized_variance_units": "unannualized sum of squared log returns on the declared target sampling grid",
        }

    def save(self, path: Path | str) -> str:
        self._seal()
        payload = _json({"body": self._body, "model_sha256": self.model_sha256})
        if len(payload.encode()) > _MAX_BYTES:
            raise ValueError("model snapshot resource bound exceeded")
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(payload)
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def load(cls, path: Path | str) -> EconomicEventModel:
        path = Path(path)
        if path.stat().st_size > _MAX_BYTES:
            raise ValueError("model snapshot resource bound exceeded")
        with path.open("rb") as stream:
            raw = stream.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            raise ValueError("model snapshot resource bound exceeded")
        try:
            payload = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise ValueError("invalid bounded JSON model") from exc
        if not isinstance(payload, dict) or set(payload) != {"body", "model_sha256"}:
            raise ValueError("snapshot envelope mismatch")
        body = payload["body"]
        if (
            not isinstance(body, dict)
            or set(body)
            != {
                "schema",
                "config",
                "train_asof",
                "calibration_asof",
                "training",
                "calibration",
                "learned",
                "training_data_sha256",
                "calibration_data_sha256",
                "implementation_sha256",
                "numpy_version",
                "synthetic",
                "research_only",
                "market_evidence",
                "authenticated_sources",
                "causal_effect_identified",
            }
            or body.get("schema") != _SCHEMA
        ):
            raise ValueError("snapshot schema mismatch")
        if (
            _hash(body) != _sha(payload["model_sha256"])
            or body["implementation_sha256"] != _IMPLEMENTATION_SHA256
            or body["numpy_version"] != np.__version__
        ):
            raise ValueError("snapshot hash/source/runtime identity mismatch")
        for flag, expected in {
            "research_only": True,
            "market_evidence": False,
            "authenticated_sources": False,
            "causal_effect_identified": False,
        }.items():
            if body[flag] is not expected:
                raise ValueError("snapshot honesty flags mismatch")
        if not isinstance(body["config"], dict) or set(body["config"]) != set(
            asdict(EventConfig("schema probe"))
        ):
            raise ValueError("snapshot config schema mismatch")
        if not isinstance(body["config"]["kinds"], list) or not all(
            isinstance(k, str) for k in body["config"]["kinds"]
        ):
            raise ValueError("snapshot event-kind sequence schema mismatch")
        if not isinstance(body["train_asof"], str) or not isinstance(body["calibration_asof"], str):
            raise ValueError("snapshot training/calibration clock schema mismatch")
        config = EventConfig(**{**body["config"], "kinds": tuple(body["config"]["kinds"])})
        train_asof = _clock(datetime.fromisoformat(body["train_asof"]), "train asof")
        calibration_asof = _clock(
            datetime.fromisoformat(body["calibration_asof"]), "calibration asof"
        )
        restored = cls.fit(
            _restore_rows(body["training"], config, train_asof, 16),
            _restore_rows(body["calibration"], config, calibration_asof, 8),
            config=config,
            train_asof=train_asof,
            calibration_asof=calibration_asof,
        )
        if restored.model_sha256 != payload["model_sha256"]:
            raise ValueError("snapshot numeric fitting/data/provenance replay mismatch")
        return restored


def _r2(actual: Array, prediction: Array) -> float | None:
    total = float(np.sum((actual - actual.mean()) ** 2))
    return None if total == 0 else 1 - float(np.sum((actual - prediction) ** 2)) / total


def _restore_rows(
    payload: Any, config: EventConfig, asof: datetime, minimum: int
) -> tuple[EventOutcome, ...]:
    if not isinstance(payload, list) or not minimum <= len(payload) <= 4096:
        raise ValueError("snapshot dataset bound exceeded")
    rows = []
    for raw in payload:
        if not isinstance(raw, dict) or set(raw) != {
            "snapshot",
            "log_return",
            "realized_variance",
            "target_end",
            "available_at",
            "source_sha256",
            "synthetic",
        }:
            raise ValueError("snapshot row schema mismatch")
        snapshot = raw["snapshot"]
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "record_id",
            "event_id",
            "asset_id",
            "kind",
            "decision_at",
            "calendar",
            "visible_releases",
            "consensus",
            "past_prices",
            "features",
            "past_returns",
            "synthetic",
        }:
            raise ValueError("snapshot observation schema mismatch")
        if (
            not isinstance(snapshot["decision_at"], str)
            or not isinstance(raw["target_end"], str)
            or not isinstance(raw["available_at"], str)
        ):
            raise ValueError("snapshot decision/target clock schema mismatch")
        calendar = _load_record(
            CalendarEvent, snapshot["calendar"], ("scheduled_at", "published_at", "available_at")
        )
        releases = snapshot["visible_releases"]
        prices = snapshot["past_prices"]
        if (
            not isinstance(releases, list)
            or not 1 <= len(releases) <= 129
            or not isinstance(prices, list)
            or not 3 <= len(prices) <= 256
        ):
            raise ValueError("snapshot history resource bound exceeded")
        observation = EventObservation(
            snapshot["record_id"],
            snapshot["asset_id"],
            calendar,
            tuple(
                _load_record(ReleaseVintage, r, ("released_at", "available_at")) for r in releases
            ),
            (
                _load_record(
                    ConsensusForecast, snapshot["consensus"], ("published_at", "available_at")
                ),
            ),
            tuple(_load_record(TimedPrice, p, ("timestamp", "available_at")) for p in prices),
            datetime.fromisoformat(snapshot["decision_at"]),
        )
        outcome = EventOutcome(
            observation,
            raw["log_return"],
            raw["realized_variance"],
            datetime.fromisoformat(raw["target_end"]),
            datetime.fromisoformat(raw["available_at"]),
            raw["source_sha256"],
            raw["synthetic"],
        )
        # Aggregate synthetic flags may include an external target annotation.
        if _row(outcome, config, asof) != raw:
            raise ValueError("snapshot causal feature/provenance replay mismatch")
        rows.append(outcome)
    return tuple(rows)


def _load_record[T](cls: type[T], payload: Any, clocks: tuple[str, ...]) -> T:
    if not isinstance(payload, dict):
        raise ValueError("snapshot record must be an object")
    data = dict(payload)
    for name in clocks:
        if name not in data or not isinstance(data[name], str):
            raise ValueError("snapshot clock schema mismatch")
        data[name] = datetime.fromisoformat(data[name])
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError("snapshot record schema mismatch") from exc
