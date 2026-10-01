"""Condition-aware off-exchange research analytics; buyer intent remains unknown.

Primary documentation inspected for the explicitly supported mappings:
https://utpplan.com/DOC/UtpBinaryOutputSpec-3.0a.pdf (sale conditions §§3.14,
message types §1.3; FINRA originator D is a reporting origin, not an ATS ID).
https://www.ctaplan.com/publicdocs/ctaplan/CTS_Pillar_Input_Specification.pdf
(CTS participant input 2.7f, four condition byte positions and message types).
https://www.finra.org/filing-reporting/otc-transparency and
https://www.finra.org/rules-guidance/rulebooks/finra-rules/6110 distinguish ATS
from non-ATS and publish attributed aggregates with delays. Such aggregates
cannot substitute for entitled, contemporaneous print/quote histories.

The parser accepts normalized records preserving source message/condition codes;
it is not a SIP binary packet decoder. Unknown versions/codes/positions fail
closed. UTP B means bunched; CTS B means average price. Local quote-sign policies
are more conservative than SIP volume/price update policies. Corrections and
cancellations invalidate the referenced original only when published; replacement
corrections are excluded, so these are filtered measures, not official SIP totals.

The disclosed DIX-like statistic is positive quote-proxy off-exchange volume /
nonzero signed quote-proxy off-exchange volume. It is neither official/branded
DIX, FINRA short-sale volume, nor hidden buyer intent. Midpoint trades have no
inferred sign. Accumulation/distribution classifications concern independently
supplied annotations only: no labels means unavailable classification. Multiclass
logistic learning uses train-only scaling and a strictly later one-versus-rest
Platt calibration partition; normalized sigmoid probabilities are not a guarantee
of calibration. Brier/log proper scores and the ECE calibration diagnostic
require another chronological holdout.

Data entitlements and quote/venue/label provenance must be independently audited.
Caller entitlement attestations and source hashes do not establish legal rights.
All outputs/alerts are local research objects; there is no external dispatch,
broker connectivity, market evidence or intent inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from sklearn.linear_model import LogisticRegression

ConditionSchema = Literal["UTP_3.0a", "CTS_INPUT_2.7f"]
Phase = Literal["accumulation", "distribution", "no_signal"]
PHASES: tuple[Phase, ...] = ("accumulation", "distribution", "no_signal")
INDICATOR_DEFINITION = (
    "positive_quote_proxy_off_exchange_volume/nonzero_signed_quote_proxy_off_exchange_volume;"
    "midpoint_unknown;not_official_DIX;not_short_sale_volume;not_buyer_intent"
)
FEATURE_NAMES = (
    "off_exchange_volume_share",
    "identified_ats_share",
    "non_ats_share",
    "unknown_off_exchange_share",
    "off_exchange_signed_fraction",
    "positive_quote_proxy_share",
    "lit_signed_fraction",
    "log_off_exchange_volume",
    "log_mean_print_size",
    "log_p95_print_size",
    "large_print_volume_share",
    "lit_price_change_bps",
    "static_price_signed_flow_measure",
    "signed_off_exchange_volume_share",
)

# Categories describe source byte positions, not interchangeable vendor codes.
_GROUPS = {
    "UTP_3.0a": ("@CNRY", " FO456789", " LTUZ", " 1ABDEGHIKMPQSVWX"),
    "CTS_INPUT_2.7f": (" CNR", " FO456789", " LTUZ", " BEHIKMPQVX"),
}
_MESSAGES = {
    "UTP_3.0a": {"TM": "trade", "TN": "trade", "TO": "cancel", "TP": "correct", "TQ": "prior_day"},
    "CTS_INPUT_2.7f": {
        "T:L": "trade",
        "T:T": "trade",
        "T:R": "trade",
        "T:H": "trade",
        "T:X": "cancel",
        "T:C": "correct",
        "T:O": "correct",
        "P:T": "prior_day",
        "P:R": "prior_day",
        "P:X": "cancel",
        "P:E": "cancel",
        "P:C": "correct",
        "P:O": "correct",
    },
}
_NAMES = {
    " ": "unspecified",
    "@": "regular",
    "A": "acquisition",
    "B": "bunched",
    "C": "cash",
    "D": "distribution_condition",
    "E": "reserved",
    "F": "intermarket_sweep",
    "G": "bunched_sold",
    "H": "price_variation",
    "I": "odd_lot",
    "K": "special_rule_trade",
    "L": "sold_last",
    "M": "official_close",
    "N": "reserved",
    "O": "opening_print",
    "P": "prior_reference_price",
    "Q": "official_open",
    "R": "seller_settlement",
    "S": "split",
    "T": "extended_hours",
    "U": "extended_hours_out_of_sequence",
    "V": "contingent",
    "W": "average_price",
    "X": "cross_or_auction",
    "Y": "yellow_flag",
    "Z": "out_of_sequence",
    "1": "stopped_stock",
    "4": "derivatively_priced",
    "5": "reopening_print",
    "6": "closing_print",
    "7": "qualified_contingent",
    "8": "reserved",
    "9": "corrected_official_close",
}


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
        raise ValueError(f"{name} must be a lowercase SHA256")


def _name(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def _time(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _number(value: float, name: str, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")


@dataclass(frozen=True, slots=True)
class DataProvenance:
    source_ids: tuple[str, ...]
    dataset_sha256: str
    entitlement_reference: str
    entitlement_attested: bool
    synthetic: bool

    def __post_init__(self) -> None:
        if not isinstance(self.source_ids, tuple) or not self.source_ids:
            raise ValueError("source_ids must be a nonempty tuple")
        for source in self.source_ids:
            _name(source, "source_id")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("duplicate source_id")
        _sha(self.dataset_sha256, "dataset_sha256")
        _name(self.entitlement_reference, "entitlement_reference")
        if not isinstance(self.entitlement_attested, bool) or not isinstance(self.synthetic, bool):
            raise ValueError("entitlement_attested and synthetic must be boolean")


@dataclass(frozen=True, slots=True)
class VenueAttribution:
    """Independently timed supplied venue evidence, not inferred from a TRF code."""

    kind: Literal["ats", "non_ats"]
    venue_id: str | None
    valid_from: datetime
    valid_to: datetime
    available_time: datetime
    basis: str
    source_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in ("ats", "non_ats"):
            raise ValueError("venue kind must be ats or non_ats")
        if self.venue_id is not None:
            _name(self.venue_id, "venue_id")
        if _time(self.valid_to, "valid_to") <= _time(self.valid_from, "valid_from"):
            raise ValueError("venue validity interval is inverted")
        _time(self.available_time, "available_time")
        _name(self.basis, "venue basis")
        _sha(self.source_sha256, "venue source_sha256")


@dataclass(frozen=True, slots=True)
class ConsolidatedPrint:
    security_id: str
    print_id: str
    source_id: str
    condition_schema: ConditionSchema
    message_type: str
    event_time: datetime
    published_at: datetime
    price: float
    size: float
    market_center_id: str
    off_exchange: bool
    sale_conditions: str
    reference_print_id: str | None = None
    attribution: VenueAttribution | None = None

    def __post_init__(self) -> None:
        for field in ("security_id", "print_id", "source_id", "message_type", "market_center_id"):
            _name(getattr(self, field), field)
        if self.condition_schema not in _GROUPS:
            raise ValueError("unsupported condition_schema")
        if _time(self.published_at, "published_at") < _time(self.event_time, "event_time"):
            raise ValueError("print publication precedes execution")
        _number(self.price, "price")
        _number(self.size, "size", zero=True)
        if not isinstance(self.off_exchange, bool):
            raise ValueError("off_exchange must be boolean")
        if self.condition_schema == "UTP_3.0a" and self.off_exchange != (
            self.market_center_id == "D"
        ):
            raise ValueError("UTP FINRA originator D/off_exchange mapping mismatch")
        if not isinstance(self.sale_conditions, str) or len(self.sale_conditions) != 4:
            raise ValueError("sale_conditions must preserve exactly four source bytes")
        if self.reference_print_id is not None:
            _name(self.reference_print_id, "reference_print_id")
        if _MESSAGES[self.condition_schema].get(self.message_type) in ("cancel", "correct") and (
            self.reference_print_id is None or self.reference_print_id == self.print_id
        ):
            raise ValueError("modification requires a different reference_print_id")
        if self.attribution is not None and not isinstance(self.attribution, VenueAttribution):
            raise ValueError("attribution must be typed VenueAttribution")
        if self.attribution is not None and not self.off_exchange:
            raise ValueError("off-exchange attribution conflicts with on-exchange print")


@dataclass(frozen=True, slots=True)
class LitQuote:
    security_id: str
    quote_id: str
    source_id: str
    event_time: datetime
    published_at: datetime
    bid: float
    ask: float

    def __post_init__(self) -> None:
        for field in ("security_id", "quote_id", "source_id"):
            _name(getattr(self, field), field)
        if _time(self.published_at, "published_at") < _time(self.event_time, "event_time"):
            raise ValueError("quote publication precedes event")
        _number(self.bid, "bid")
        _number(self.ask, "ask")
        # Locked/crossed quotes are retained in the audit but cannot provide sign.


def parse_consolidated_print(
    row: Mapping[str, Any],
    *,
    source_id: str,
    condition_schema: ConditionSchema,
) -> ConsolidatedPrint:
    """Parse a normalized vendor row without trimming condition-byte spaces.

    Required field names follow ConsolidatedPrint; source/version come from the
    adapter, never guessed from condition codes. Datetimes accept aware datetime
    values or ISO8601 strings. Numeric JSON strings and truthy boolean strings
    are rejected. Attribution must be separately supplied typed evidence.
    """

    def clock(field: str) -> datetime:
        value = row[field]
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"invalid {field}") from exc
        _time(value, field)
        return cast(datetime, value)

    try:
        return ConsolidatedPrint(
            security_id=row["security_id"],
            print_id=row["print_id"],
            source_id=source_id,
            condition_schema=condition_schema,
            message_type=row["message_type"],
            event_time=clock("event_time"),
            published_at=clock("published_at"),
            price=row["price"],
            size=row["size"],
            market_center_id=row["market_center_id"],
            off_exchange=row["off_exchange"],
            sale_conditions=row["sale_conditions"],
            reference_print_id=row.get("reference_print_id"),
            attribution=row.get("attribution"),
        )
    except KeyError as exc:
        raise ValueError(f"missing print field {exc.args[0]}") from exc


@dataclass(frozen=True, slots=True)
class ConditionDecision:
    retained_volume: bool
    permits_quote_sign: bool
    descriptions: tuple[str, ...]
    reasons: tuple[str, ...]


def classify_conditions(print_: ConsolidatedPrint) -> ConditionDecision:
    groups = _GROUPS[print_.condition_schema]
    codes = print_.sale_conditions
    if any(code not in allowed for code, allowed in zip(codes, groups, strict=True)):
        return ConditionDecision(False, False, (), ("unknown_or_mispositioned_condition",))
    names = _NAMES.copy()
    if print_.condition_schema == "CTS_INPUT_2.7f":
        names.update(
            {"B": "average_price", "E": "automatic_execution", " ": "regular_or_unspecified"}
        )
    descriptions = tuple(names[c] for c in codes)
    if any(c in "N8" for c in codes) or (print_.condition_schema == "UTP_3.0a" and "E" in codes):
        return ConditionDecision(False, False, descriptions, ("reserved_condition_fail_closed",))
    if any(c in "MQ9" for c in codes):
        return ConditionDecision(False, False, descriptions, ("official_price_message_not_volume",))
    if print_.size == 0:
        return ConditionDecision(False, False, descriptions, ("zero_size",))
    average = "W" in codes if print_.condition_schema == "UTP_3.0a" else "B" in codes
    reasons: list[str] = []
    if average or any(c in "CHPRV47" for c in codes):
        reasons.append("non_current_price_condition_volume_only")
    if any(c in "LZUG" for c in codes):
        reasons.append("late_or_out_of_sequence_volume_only")
    if "T" in codes:
        reasons.append("extended_hours_volume_only")
    # Auctions, aggregated executions, odd lots (which can override other special
    # condition detail), and special settlements do not receive inferred side.
    special = "ABDS1KO56XI" if print_.condition_schema == "UTP_3.0a" else "KO56XI"
    if any(c in special for c in codes):
        reasons.append("special_execution_volume_only")
    if "Y" in codes:
        reasons.append("yellow_flag_volume_only")
    return ConditionDecision(
        True,
        not reasons,
        descriptions,
        tuple(reasons) if reasons else ("regular_or_sweep_retained",),
    )


@dataclass(frozen=True, slots=True)
class WindowConfig:
    lookback: timedelta = timedelta(minutes=5)
    max_quote_age: timedelta = timedelta(seconds=1)
    max_reporting_delay: timedelta = timedelta(seconds=10)
    large_print_size: float = 1000.0

    def __post_init__(self) -> None:
        for name in ("lookback", "max_quote_age", "max_reporting_delay"):
            value = getattr(self, name)
            if not isinstance(value, timedelta) or value <= timedelta(0):
                raise ValueError(f"{name} must be positive timedelta")
        _number(self.large_print_size, "large_print_size")


@dataclass(frozen=True, slots=True)
class ClassifiedPrint:
    print_id: str
    source_id: str
    venue_class: str
    venue_id: str | None
    retained_volume: bool
    quote_sign: int | None
    quote_id: str | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OffExchangeWindow:
    security_id: str
    start: datetime
    cutoff: datetime
    audits: tuple[ClassifiedPrint, ...]
    features: tuple[float, ...]
    off_exchange_volume: float
    lit_volume: float
    off_exchange_signed_volume: float
    lit_signed_volume: float
    positive_quote_proxy_share: float | None
    available: bool
    data_sha256: str
    config_sha256: str
    source_mapping_sha256: str
    provenance: DataProvenance
    implementation_sha256: str
    receipt_sha256: str = dataclass_field(init=False)
    indicator_definition: str = INDICATOR_DEFINITION
    buyer_intent: str = dataclass_field(init=False, default="unknown")
    research_only: bool = dataclass_field(init=False, default=True)
    market_evidence: bool = dataclass_field(init=False, default=False)

    def __post_init__(self) -> None:
        _name(self.security_id, "security_id")
        if _time(self.start, "window start") >= _time(self.cutoff, "window cutoff"):
            raise ValueError("window interval is inverted")
        if (
            not isinstance(self.features, tuple)
            or len(self.features) != len(FEATURE_NAMES)
            or not np.isfinite(self.features).all()
        ):
            raise ValueError("window features must be finite and match fixed schema")
        for name in ("off_exchange_volume", "lit_volume"):
            _number(getattr(self, name), name, zero=True)
        for name, limit in (
            ("off_exchange_signed_volume", self.off_exchange_volume),
            ("lit_signed_volume", self.lit_volume),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or abs(value) > limit + 1e-8
            ):
                raise ValueError("signed volume must be finite and bounded by retained volume")
        if not isinstance(self.available, bool) or self.available != (self.off_exchange_volume > 0):
            raise ValueError("window availability must match eligible off-exchange volume")
        if (
            self.positive_quote_proxy_share is not None
            and not 0 <= self.positive_quote_proxy_share <= 1
        ):
            raise ValueError("quote-proxy share must lie in [0, 1] or be unavailable")
        if not isinstance(self.audits, tuple) or any(
            not isinstance(a, ClassifiedPrint) for a in self.audits
        ):
            raise ValueError("window audits must be an immutable typed tuple")
        if not isinstance(self.provenance, DataProvenance):
            raise ValueError("typed window provenance required")
        if not self.provenance.synthetic and not self.provenance.entitlement_attested:
            raise ValueError("window requires independently reviewed entitlement attestation")
        for name in (
            "data_sha256",
            "config_sha256",
            "source_mapping_sha256",
            "implementation_sha256",
        ):
            _sha(getattr(self, name), name)
        payload = {
            name: getattr(self, name)
            for name in (
                "security_id",
                "start",
                "cutoff",
                "features",
                "off_exchange_volume",
                "lit_volume",
                "off_exchange_signed_volume",
                "lit_signed_volume",
                "positive_quote_proxy_share",
                "available",
                "data_sha256",
                "config_sha256",
                "source_mapping_sha256",
                "implementation_sha256",
                "indicator_definition",
            )
        }
        payload["audits"] = [asdict(a) for a in self.audits]
        payload["provenance"] = asdict(self.provenance)
        object.__setattr__(self, "receipt_sha256", _hash(payload))


def _venue(print_: ConsolidatedPrint, cutoff: datetime) -> tuple[str, str | None]:
    if not print_.off_exchange:
        return "on_exchange", None
    evidence = print_.attribution
    if evidence is None or _time(evidence.available_time, "venue availability") > cutoff:
        return "unknown_off_exchange", None
    if not (
        _time(evidence.valid_from, "valid_from")
        <= _time(print_.event_time, "event_time")
        < _time(evidence.valid_to, "valid_to")
    ):
        return "unknown_off_exchange", None
    suffix = "venue_identified" if evidence.venue_id is not None else "unattributed"
    return f"{evidence.kind}_{suffix}", evidence.venue_id


def analyze_off_exchange(
    prints: Sequence[ConsolidatedPrint],
    quotes: Sequence[LitQuote],
    *,
    security_id: str,
    cutoff: datetime,
    provenance: DataProvenance,
    config: WindowConfig | None = None,
) -> OffExchangeWindow:
    """Causal trailing (start, cutoff] event window, published-by-cutoff ledger.

    Corrective messages are applied regardless of original event-window membership.
    Quote sign uses only an unlocked/uncrossed quote whose event and publication
    are both <= execution, with bounded age. This is an aggressor-side proxy only.
    """
    cutoff = _time(cutoff, "cutoff")
    _name(security_id, "security_id")
    config = config or WindowConfig()
    if not isinstance(provenance, DataProvenance):
        raise ValueError("typed provenance required")
    if not provenance.synthetic and not provenance.entitlement_attested:
        raise ValueError(
            "non-synthetic data requires independently reviewed entitlement attestation"
        )
    if any(not isinstance(p, ConsolidatedPrint) for p in prints) or any(
        not isinstance(q, LitQuote) for q in quotes
    ):
        raise ValueError("typed print and quote records required")
    if any(p.source_id not in provenance.source_ids for p in prints) or any(
        q.source_id not in provenance.source_ids for q in quotes
    ):
        raise ValueError("record source missing from provenance")
    start = cutoff - config.lookback
    visible = [
        p
        for p in prints
        if p.security_id == security_id and _time(p.published_at, "publication") <= cutoff
    ]
    known_quotes = [
        q
        for q in quotes
        if q.security_id == security_id and _time(q.published_at, "publication") <= cutoff
    ]
    if len({(p.source_id, p.print_id) for p in visible}) != len(visible):
        raise ValueError("duplicate visible print id")
    if len({(q.source_id, q.quote_id) for q in known_quotes}) != len(known_quotes):
        raise ValueError("duplicate visible quote id")
    visible.sort(key=lambda p: (_time(p.published_at, "publication"), p.source_id, p.print_id))
    modifications = {
        (p.source_id, p.reference_print_id): _MESSAGES[p.condition_schema][p.message_type]
        for p in visible
        if _MESSAGES[p.condition_schema].get(p.message_type) in ("cancel", "correct")
    }
    known_ids = {
        (p.source_id, p.print_id)
        for p in visible
        if _MESSAGES[p.condition_schema].get(p.message_type) == "trade"
    }
    audits: list[ClassifiedPrint] = []
    retained: list[tuple[ConsolidatedPrint, str, int | None]] = []
    for p in visible:
        event = _time(p.event_time, "execution")
        venue, venue_id = _venue(p, cutoff)
        action = _MESSAGES[p.condition_schema].get(p.message_type)
        reasons: list[str] = []
        condition = classify_conditions(p)
        keep, sign, quote_id = False, None, None
        if action in ("cancel", "correct"):
            reasons.append(f"{action}_message_excluded")
            if (p.source_id, p.reference_print_id) not in known_ids:
                reasons.append("reference_original_unavailable")
        elif action != "trade":
            reasons.append("unknown_or_prior_day_message_excluded")
        elif (p.source_id, p.print_id) in modifications:
            reasons.append(
                f"original_invalidated_by_published_{modifications[(p.source_id, p.print_id)]}"
            )
        elif not start < event <= cutoff:
            reasons.append("outside_trailing_event_window")
        else:
            keep = condition.retained_volume
            reasons.extend(condition.reasons)
            permit_sign = condition.permits_quote_sign
            if _time(p.published_at, "publication") - event > config.max_reporting_delay:
                permit_sign = False
                reasons.append("reporting_delay_volume_only")
            if keep and permit_sign:
                candidates = [
                    q
                    for q in known_quotes
                    if _time(q.event_time, "quote event") <= event
                    and _time(q.published_at, "quote publication") <= event
                ]
                if not candidates:
                    reasons.append("no_contemporaneous_published_lit_quote")
                else:
                    q = max(
                        candidates,
                        key=lambda q: (
                            _time(q.event_time, "quote event"),
                            _time(q.published_at, "quote publication"),
                            q.source_id,
                            q.quote_id,
                        ),
                    )
                    quote_id = q.quote_id
                    tied = [
                        candidate
                        for candidate in candidates
                        if _time(candidate.event_time, "quote event")
                        == _time(q.event_time, "quote event")
                        and _time(candidate.published_at, "quote publication")
                        == _time(q.published_at, "quote publication")
                    ]
                    if len({(candidate.bid, candidate.ask) for candidate in tied}) > 1:
                        reasons.append("ambiguous_contemporaneous_quotes")
                    elif event - _time(q.event_time, "quote event") > config.max_quote_age:
                        reasons.append("stale_lit_quote")
                    elif q.ask <= q.bid:
                        reasons.append("locked_or_crossed_lit_quote")
                    elif not q.bid <= p.price <= q.ask:
                        reasons.append("print_outside_lit_spread")
                    else:
                        midpoint = (q.bid + q.ask) / 2
                        sign = 1 if p.price > midpoint else -1 if p.price < midpoint else 0
                        reasons.append(
                            "midpoint_sign_unknown" if sign == 0 else "quote_midpoint_sign_proxy"
                        )
            if keep:
                retained.append((p, venue, sign))
        audits.append(
            ClassifiedPrint(
                p.print_id, p.source_id, venue, venue_id, keep, sign, quote_id, tuple(reasons)
            )
        )

    off = [(p, venue, sign) for p, venue, sign in retained if p.off_exchange]
    lit = [(p, sign) for p, _, sign in retained if not p.off_exchange]
    off_volume = float(sum(p.size for p, _, _ in off))
    lit_volume = float(sum(p.size for p, _ in lit))
    signed_off_denominator = float(sum(p.size for p, _, sign in off if sign in (-1, 1)))
    signed_lit_denominator = float(sum(p.size for p, sign in lit if sign in (-1, 1)))
    signed_off = float(sum(p.size * sign for p, _, sign in off if sign in (-1, 1)))
    signed_lit = float(sum(p.size * sign for p, sign in lit if sign in (-1, 1)))
    positive = float(sum(p.size for p, _, sign in off if sign == 1))
    proxy_share = positive / signed_off_denominator if signed_off_denominator else None
    sizes = np.array([p.size for p, _, _ in off], dtype=float)
    lit_priceforming = sorted(
        [(p, sign) for p, sign in lit if sign is not None],
        key=lambda item: _time(item[0].event_time, "execution"),
    )
    price_change = (
        10000 * (lit_priceforming[-1][0].price / lit_priceforming[0][0].price - 1)
        if len(lit_priceforming) > 1
        else 0.0
    )
    total = off_volume + lit_volume

    def volume_by_kind(prefix: str) -> float:
        return float(sum(p.size for p, venue, _ in off if venue.startswith(prefix)))

    features = (
        off_volume / total if total else 0.0,
        volume_by_kind("ats_venue_identified") / off_volume if off_volume else 0.0,
        volume_by_kind("non_ats") / off_volume if off_volume else 0.0,
        volume_by_kind("unknown_off_exchange") / off_volume if off_volume else 0.0,
        signed_off / signed_off_denominator if signed_off_denominator else 0.0,
        proxy_share if proxy_share is not None else 0.5,
        signed_lit / signed_lit_denominator if signed_lit_denominator else 0.0,
        math.log1p(off_volume),
        math.log1p(float(np.mean(sizes))) if sizes.size else 0.0,
        math.log1p(float(np.quantile(sizes, 0.95))) if sizes.size else 0.0,
        float(sum(p.size for p, _, _ in off if p.size >= config.large_print_size)) / off_volume
        if off_volume
        else 0.0,
        price_change,
        abs(signed_lit) / (1 + abs(price_change)),
        signed_off_denominator / off_volume if off_volume else 0.0,
    )
    if not np.isfinite(features).all():
        raise ValueError("nonfinite aggregate features rejected")
    condition_hash = _hash({"groups": _GROUPS, "messages": _MESSAGES, "names": _NAMES})
    data_rows = []
    for p in visible:
        row = asdict(p)
        if (
            p.attribution is not None
            and _time(p.attribution.available_time, "venue availability") > cutoff
        ):
            row["attribution"] = None
        data_rows.append(row)
    data_hash = _hash({"prints": data_rows, "quotes": [asdict(q) for q in known_quotes]})
    config_hash = _hash(asdict(config))
    return OffExchangeWindow(
        security_id,
        start,
        cutoff,
        tuple(audits),
        tuple(float(x) for x in features),
        off_volume,
        lit_volume,
        signed_off,
        signed_lit,
        proxy_share,
        off_volume > 0,
        data_hash,
        config_hash,
        condition_hash,
        provenance,
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class WindowAnnotation:
    """Supplied independent labels for the exact feature window, published later."""

    annotation_id: str
    window_sha256: str
    phase: Phase
    observed_start: datetime
    observed_end: datetime
    available_time: datetime
    source_id: str
    methodology: str
    independent_of_print_features: bool
    synthetic: bool

    def __post_init__(self) -> None:
        for field in ("annotation_id", "source_id", "methodology"):
            _name(getattr(self, field), field)
        _sha(self.window_sha256, "window_sha256")
        if self.phase not in PHASES:
            raise ValueError("unsupported annotation phase")
        if _time(self.observed_start, "observed_start") >= _time(self.observed_end, "observed_end"):
            raise ValueError("annotation interval is inverted")
        if _time(self.available_time, "label availability") < _time(
            self.observed_end, "observed_end"
        ):
            raise ValueError("label availability precedes observation end")
        if self.independent_of_print_features is not True or not isinstance(self.synthetic, bool):
            raise ValueError(
                "labels require explicit independent methodology and synthetic identity"
            )


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    regularization_c: float = 1.0
    platt_c: float = 1.0
    alert_threshold: float = 0.8
    ece_bins: int = 10

    def __post_init__(self) -> None:
        _number(self.regularization_c, "regularization_c")
        _number(self.platt_c, "platt_c")
        if isinstance(self.alert_threshold, bool) or not 0 < self.alert_threshold <= 1:
            raise ValueError("alert_threshold must lie in (0, 1]")
        if (
            isinstance(self.ece_bins, bool)
            or not isinstance(self.ece_bins, int)
            or self.ece_bins < 1
        ):
            raise ValueError("ece_bins must be positive integer")


@dataclass(frozen=True, slots=True)
class PhaseSignal:
    status: Literal["available", "unavailable"]
    reason: str
    phase: Phase | None
    probabilities: tuple[float, float, float] | None
    window_sha256: str
    model_sha256: str | None
    synthetic: bool
    buyer_intent: str = dataclass_field(init=False, default="unknown")
    target_semantics: str = "supplied_independent_window_annotations"
    research_only: bool = dataclass_field(init=False, default=True)
    market_evidence: bool = dataclass_field(init=False, default=False)


@dataclass(frozen=True, slots=True)
class ResearchAlert:
    security_id: str
    cutoff: datetime
    phase: Phase
    probability: float
    window_sha256: str
    model_sha256: str
    alert_sha256: str
    synthetic: bool
    indicator_definition: str = INDICATOR_DEFINITION
    buyer_intent: str = dataclass_field(init=False, default="unknown")
    local_only: bool = dataclass_field(init=False, default=True)
    research_only: bool = dataclass_field(init=False, default=True)
    market_evidence: bool = dataclass_field(init=False, default=False)


class OffExchangePhaseClassifier:
    """Calibrated learning of supplied annotations; no unsupervised intent labels."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self._config = config or ClassifierConfig()
        self._base: LogisticRegression | None = None
        self._calibrators: tuple[LogisticRegression, ...] = ()
        self._mean = np.zeros(len(FEATURE_NAMES))
        self._scale = np.ones(len(FEATURE_NAMES))
        self._calibration_cutoff: datetime | None = None
        self._model_sha256: str | None = None
        self._synthetic = False
        self._reason = "missing_independent_annotations"
        self._data_label = "UNMEASURED"

    @property
    def config(self) -> ClassifierConfig:
        return self._config

    @property
    def model_sha256(self) -> str | None:
        return self._model_sha256

    @staticmethod
    def _labelled(
        windows: Sequence[OffExchangeWindow],
        annotations: Sequence[WindowAnnotation],
        asof: datetime,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if not windows or len(windows) != len(annotations):
            raise ValueError("windows and annotations must have equal nonzero length")
        if any(not isinstance(w, OffExchangeWindow) for w in windows) or any(
            not isinstance(a, WindowAnnotation) for a in annotations
        ):
            raise ValueError("typed windows and independent annotations required")
        if len({w.receipt_sha256 for w in windows}) != len(windows):
            raise ValueError("duplicate window")
        if len({a.annotation_id for a in annotations}) != len(annotations):
            raise ValueError("duplicate annotation_id")
        for w, a in zip(windows, annotations, strict=True):
            if not w.available:
                raise ValueError("annotated window has no eligible off-exchange volume")
            if (
                a.window_sha256 != w.receipt_sha256
                or _time(a.observed_start, "observed_start") != w.start
                or _time(a.observed_end, "observed_end") != w.cutoff
            ):
                raise ValueError("annotation window hash/interval alignment mismatch")
            if _time(a.available_time, "label availability") > asof or w.cutoff > asof:
                raise ValueError("annotation or window unpublished at partition cutoff")
            if a.synthetic != w.provenance.synthetic:
                raise ValueError("annotation/window synthetic identity mismatch")
        return np.array([w.features for w in windows]), np.array(
            [PHASES.index(a.phase) for a in annotations]
        )

    def fit(
        self,
        training_windows: Sequence[OffExchangeWindow],
        training_annotations: Sequence[WindowAnnotation] | None,
        *,
        calibration_windows: Sequence[OffExchangeWindow],
        calibration_annotations: Sequence[WindowAnnotation] | None,
        training_cutoff: datetime,
        calibration_cutoff: datetime,
    ) -> OffExchangePhaseClassifier:
        train_cutoff = _time(training_cutoff, "training_cutoff")
        calibration_cutoff = _time(calibration_cutoff, "calibration_cutoff")
        if self._base is not None:
            raise RuntimeError("fitted classifier is frozen; use a fresh instance")
        if training_annotations is None or calibration_annotations is None:
            self._reason = "missing_independent_annotations"
            return self
        if calibration_cutoff <= train_cutoff:
            raise ValueError("calibration cutoff must follow training cutoff")
        x, y = self._labelled(training_windows, training_annotations, train_cutoff)
        cx, cy = self._labelled(calibration_windows, calibration_annotations, calibration_cutoff)
        if any(w.start <= train_cutoff for w in calibration_windows):
            raise ValueError("Platt calibration windows must begin strictly after training cutoff")
        if any(np.count_nonzero(labels == k) < 2 for labels in (y, cy) for k in range(3)):
            raise ValueError(
                "training and calibration require at least two independent labels per class"
            )
        for windows in (training_windows, calibration_windows):
            by_entity: dict[str, list[OffExchangeWindow]] = {}
            for w in windows:
                by_entity.setdefault(w.security_id, []).append(w)
            for group in by_entity.values():
                ordered = sorted(group, key=lambda w: w.start)
                if any(a.cutoff > b.start for a, b in zip(ordered, ordered[1:], strict=False)):
                    raise ValueError("labelled windows cannot overlap within a security")
        train_ids = {a.annotation_id for a in training_annotations}
        if train_ids.intersection(a.annotation_id for a in calibration_annotations):
            raise ValueError("annotation partitions must be disjoint")
        if len({w.config_sha256 for w in (*training_windows, *calibration_windows)}) != 1:
            raise ValueError("window configuration must match across partitions")
        self._mean = x.mean(axis=0)
        deviation = x.std(axis=0)
        self._scale = np.where(deviation < 1e-8, 1.0, deviation)
        base = LogisticRegression(C=self.config.regularization_c, solver="lbfgs", max_iter=2000)
        base.fit((x - self._mean) / self._scale, y)
        logits = base.decision_function((cx - self._mean) / self._scale)
        calibrators = []
        for k in range(3):
            platt = LogisticRegression(C=self.config.platt_c, solver="lbfgs", max_iter=2000)
            platt.fit(logits[:, k : k + 1], (cy == k).astype(int))
            calibrators.append(platt)
        self._base, self._calibrators = base, tuple(calibrators)
        for model in (base, *calibrators):
            model.coef_.setflags(write=False)
            model.intercept_.setflags(write=False)
        self._mean.setflags(write=False)
        self._scale.setflags(write=False)
        self._calibration_cutoff = calibration_cutoff
        self._synthetic = any(
            w.provenance.synthetic for w in (*training_windows, *calibration_windows)
        )
        all_synthetic = all(
            w.provenance.synthetic for w in (*training_windows, *calibration_windows)
        )
        self._data_label = (
            "SYNTHETIC"
            if all_synthetic
            else "MIXED"
            if self._synthetic
            else "SUPPLIED_NON_SYNTHETIC"
        )
        self._window_config_sha256 = training_windows[0].config_sha256
        self._model_sha256 = _hash(
            {
                "algorithm": "multinomial_logistic_then_later_ovr_Platt_normalized.v1",
                "configuration": asdict(self.config),
                "features": FEATURE_NAMES,
                "phases": PHASES,
                "training_cutoff": train_cutoff,
                "calibration_cutoff": calibration_cutoff,
                "mean": self._mean.tolist(),
                "scale": self._scale.tolist(),
                "base_coefficients": base.coef_.tolist(),
                "base_intercepts": base.intercept_.tolist(),
                "platt": [
                    {"coefficients": p.coef_.tolist(), "intercepts": p.intercept_.tolist()}
                    for p in calibrators
                ],
                "training_windows": [w.receipt_sha256 for w in training_windows],
                "calibration_windows": [w.receipt_sha256 for w in calibration_windows],
                "annotations": [
                    asdict(a) for a in (*training_annotations, *calibration_annotations)
                ],
                "window_config_sha256": self._window_config_sha256,
                "synthetic": self._synthetic,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )
        self._reason = "supplied_annotations_later_Platt_calibration"
        return self

    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": "multinomial_logistic_then_later_ovr_Platt_normalized.v1",
            "status": "available" if self._base is not None else "unavailable",
            "reason": self._reason,
            "configuration": asdict(self.config),
            "config_sha256": _hash(asdict(self.config)),
            "model_sha256": self.model_sha256,
            "calibration_cutoff": self._calibration_cutoff,
            "features": FEATURE_NAMES,
            "phases": PHASES,
            "data_label": self._data_label,
            "target_semantics": "supplied_independent_window_annotations",
            "independence_evidence": "caller_attestation_requires_external_audit",
            "indicator_definition": INDICATOR_DEFINITION,
            "buyer_intent": "unknown",
            "research_only": True,
            "market_evidence": False,
            "local_alerts_only": True,
            "condition_versions": tuple(_GROUPS),
        }

    def predict(self, window: OffExchangeWindow) -> PhaseSignal:
        if not isinstance(window, OffExchangeWindow):
            raise ValueError("typed window required")
        synthetic = self._synthetic or window.provenance.synthetic
        if self._base is None:
            return PhaseSignal(
                "unavailable", self._reason, None, None, window.receipt_sha256, None, synthetic
            )
        if not window.available:
            return PhaseSignal(
                "unavailable",
                "no_eligible_off_exchange_volume",
                None,
                None,
                window.receipt_sha256,
                self.model_sha256,
                synthetic,
            )
        if self._calibration_cutoff is None or window.start <= self._calibration_cutoff:
            raise ValueError("prediction window must begin after calibration cutoff")
        if window.config_sha256 != self._window_config_sha256:
            raise ValueError("prediction window configuration mismatch")
        logits = self._base.decision_function(
            (np.array([window.features]) - self._mean) / self._scale
        )
        probabilities = np.array(
            [p.predict_proba(logits[:, k : k + 1])[0, 1] for k, p in enumerate(self._calibrators)]
        )
        probabilities = np.maximum(probabilities, 1e-12)
        probabilities /= probabilities.sum()
        result = cast(tuple[float, float, float], tuple(float(p) for p in probabilities))
        return PhaseSignal(
            "available",
            self._reason,
            PHASES[int(np.argmax(probabilities))],
            result,
            window.receipt_sha256,
            self.model_sha256,
            synthetic,
        )

    def evaluate(
        self,
        windows: Sequence[OffExchangeWindow],
        annotations: Sequence[WindowAnnotation],
        *,
        asof: datetime,
    ) -> dict[str, float]:
        _, y = self._labelled(windows, annotations, _time(asof, "evaluation asof"))
        signals = [self.predict(w) for w in windows]
        if any(s.probabilities is None for s in signals):
            raise ValueError("classification unavailable; cannot fabricate evaluation scores")
        probabilities = np.array([s.probabilities for s in signals], dtype=float)
        one_hot = np.eye(3)[y]
        brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        log_score = float(-np.mean(np.log(probabilities[np.arange(len(y)), y])))
        confidence, predicted = probabilities.max(axis=1), probabilities.argmax(axis=1)
        bin_ids = np.minimum(
            (confidence * self.config.ece_bins).astype(int), self.config.ece_bins - 1
        )
        ece = 0.0
        for b in range(self.config.ece_bins):
            mask = bin_ids == b
            if mask.any():
                ece += float(
                    mask.mean() * abs(np.mean(predicted[mask] == y[mask]) - confidence[mask].mean())
                )
        return {"multiclass_brier": brier, "log_score": log_score, "ece": ece}

    def alert(self, window: OffExchangeWindow) -> ResearchAlert | None:
        signal = self.predict(window)
        if signal.probabilities is None or signal.phase in (None, "no_signal"):
            return None
        probability = max(signal.probabilities)
        if probability < self.config.alert_threshold or signal.model_sha256 is None:
            return None
        payload = {
            "security_id": window.security_id,
            "cutoff": window.cutoff,
            "phase": signal.phase,
            "probability": probability,
            "window_sha256": window.receipt_sha256,
            "model_sha256": signal.model_sha256,
            "synthetic": signal.synthetic,
            "indicator_definition": INDICATOR_DEFINITION,
            "research_only": True,
            "local_only": True,
        }
        return ResearchAlert(
            window.security_id,
            window.cutoff,
            signal.phase,
            probability,
            window.receipt_sha256,
            signal.model_sha256,
            _hash(payload),
            signal.synthetic,
        )
