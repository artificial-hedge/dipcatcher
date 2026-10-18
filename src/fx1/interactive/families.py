"""Planner-facing research-family catalog.

Loaded from the harness command ``research-families``. Goals are classified
against required families, live optional families, and the full retired set
when the document includes ``retired``. A document that only carries
``retired_sample`` classifies that sample; every other name is UNKNOWN.
UNKNOWN and RETIRED both fail closed for new selection.

``execution_constraint`` reports whether the research runner can execute a
chosen subset. It is false today: selection narrows eligibility and reporting.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any

CATALOG_SCHEMA_VERSION = 1

_EXPLICIT_TOKEN = re.compile(r"\bfamily:([a-z0-9_]+)\b")
_REASON_LIMIT = 200
_UNAVAILABLE_LINE = (
    "family catalog unavailable — family eligibility could not be checked; "
    "no family-level claims will be made (fail-closed)"
)
_EXECUTION_NOTE = (
    "note: the research command runs its configured scorecard; family selection "
    "narrows eligibility and reporting, not the benches executed."
)

_WORD = re.compile(r"[a-z0-9_]+")
_CacheKey = tuple[frozenset[str], frozenset[str], frozenset[str], tuple[tuple[str, str], ...]]
_Matcher = tuple[dict[str, str], int]
_MATCHER_CACHE: dict[_CacheKey, _Matcher] = {}


class FamilyState(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    RETIRED = "retired"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FamilyCatalog:
    """One snapshot of the benchmark family catalog."""

    schema_version: int
    catalog_version: int
    receipt_schema_version: int
    required: frozenset[str]
    live_optional: frozenset[str]
    retired_count: int
    retired_sample: frozenset[str]
    order: tuple[str, ...]
    generated_at: str
    source: str = "harness"
    execution_constraint: bool = False
    note: str = ""
    # Empty means the document omitted ``retired`` and only the sample classifies.
    retired: frozenset[str] = frozenset()

    @classmethod
    def from_json(cls, text: str) -> FamilyCatalog:
        """Parse a ``research-families`` JSON document.

        Raises ``ValueError`` with ``unsupported family catalog schema`` when
        ``schema_version`` is not :data:`CATALOG_SCHEMA_VERSION`. Every other
        malformed field raises ``ValueError`` and is rejected (fail closed).
        """
        payload = _as_object(text)
        constrained, note = _runner_fields(payload)
        sample = _require_str_list(payload.get("retired_sample"), "retired_sample")
        count = _require_int(payload.get("retired_count"), "retired_count")
        return cls(
            schema_version=_require_schema(payload),
            catalog_version=_require_int(payload.get("catalog_version"), "catalog_version"),
            receipt_schema_version=_require_int(
                payload.get("receipt_schema_version"), "receipt_schema_version"
            ),
            required=frozenset(_require_str_list(payload.get("required"), "required")),
            live_optional=frozenset(
                _require_str_list(payload.get("live_optional"), "live_optional")
            ),
            retired_count=count,
            retired_sample=frozenset(sample),
            order=tuple(_require_str_list(payload.get("order"), "order")),
            generated_at=_require_str(payload.get("generated_at"), "generated_at"),
            source="harness",
            execution_constraint=constrained,
            note=note,
            retired=_retired_membership(payload, sample, count),
        )

    @classmethod
    def unavailable(cls, reason: str) -> FamilyCatalog:
        """Empty catalog. Callers must fail closed and make no family claims."""
        return cls(
            schema_version=CATALOG_SCHEMA_VERSION,
            catalog_version=0,
            receipt_schema_version=0,
            required=frozenset(),
            live_optional=frozenset(),
            retired_count=0,
            retired_sample=frozenset(),
            order=(),
            generated_at="",
            source="unavailable",
            execution_constraint=False,
            note=reason,
        )

    @property
    def available(self) -> bool:
        return self.source != "unavailable"

    def state_of(self, family: str) -> FamilyState:
        """Classify ``family``.

        ``retired`` is the full set when the document carried one. Otherwise
        only ``retired_sample`` is retired. A name in none of the sets is
        UNKNOWN. UNKNOWN and RETIRED both fail closed.
        """
        if family in self.required:
            return FamilyState.REQUIRED
        if family in self.live_optional:
            return FamilyState.OPTIONAL
        if family in self.retired or family in self.retired_sample:
            return FamilyState.RETIRED
        return FamilyState.UNKNOWN

    def eligible(self) -> frozenset[str]:
        return self.required | self.live_optional

    def describe(self) -> str:
        if not self.available:
            return f"family catalog unavailable: {self.note}"
        summary = (
            f"family catalog v{self.catalog_version} "
            f"(receipt schema v{self.receipt_schema_version}): "
            f"{len(self.required)} required · {len(self.live_optional)} live optional · "
            f"{self.retired_count} retired · source {self.source}"
        )
        if self.note:
            return f"{summary} · note: {self.note}"
        return summary


@dataclass(frozen=True)
class _Hit:
    start: int
    end: int
    name: str
    matched_text: str
    explicit: bool


@dataclass(frozen=True)
class FamilyMention:
    name: str
    state: FamilyState
    matched_text: str


@dataclass(frozen=True)
class FamilySelection:
    eligible: tuple[FamilyMention, ...]
    retired: tuple[FamilyMention, ...]
    unknown: tuple[FamilyMention, ...]
    catalog_source: str
    ordered: tuple[FamilyMention, ...] = field(default_factory=tuple, repr=False)

    @property
    def mentioned(self) -> bool:
        return bool(self.eligible or self.retired or self.unknown)

    @property
    def refused(self) -> bool:
        """True when every mention is retired or unknown."""
        return bool(self.retired or self.unknown) and not self.eligible

    def explain(self) -> list[str]:
        if self.catalog_source == "unavailable":
            return [_UNAVAILABLE_LINE]
        mentions = self.ordered or (*self.eligible, *self.retired, *self.unknown)
        return [_explain_mention(mention) for mention in mentions]


DEFAULT_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "vol": "volatility",
        "volatility forecast": "volatility",
        "realized volatility": "volatility",
        "qlike": "volatility",
        "tail risk": "tail",
        "var": "tail",
        "value at risk": "tail",
        "expected shortfall": "tail",
        "kupiec": "tail",
        "crps": "distribution",
        "pinball": "distribution",
        "quantile": "distribution",
        "quantiles": "distribution",
        "coverage": "conformal",
        "calibration": "conformal",
        "calibrated": "conformal",
        "prediction interval": "conformal",
        "e-values": "evalues",
        "e-value": "evalues",
        "e values": "evalues",
        "order book": "northset",
        "lob": "northset",
        "microstructure": "northset",
        "rank ic": "ranking",
        "ic": "ranking",
        "rankings": "ranking",
        "bandit": "quantile_bandit",
        "cross validation": "cv_plus",
        "cross-validation": "cv_plus",
        "purged cv": "cpcv",
        "regimes": "regime",
        "hmm": "regime",
        "drawdowns": "drawdown",
        "illiquidity": "liquidity",
        "jackknife": "jackknife_plus",
        "weighted conformal": "weighted_conformal",
        "interval risk": "interval_risk",
        "online crc": "online_crc",
        "localized conformal": "localized_conformal",
        "conformal rank": "conformal_rank",
        "portfolio conformal": "portfolio_conformal",
    }
)


def load_catalog(harness: Any, *, cache: dict[str, FamilyCatalog] | None = None) -> FamilyCatalog:
    """Load ``research-families`` through ``harness``. Never raises.

    A cache hit returns the stored catalog with ``source="cached"``. A
    non-zero exit, a parse failure, or any exception from the harness becomes
    :meth:`FamilyCatalog.unavailable` using the first 200 characters of the
    stderr text or the exception message. Only a successful load is stored.
    """
    if cache is not None:
        cached = _cached_catalog(cache)
        if cached is not None:
            return cached
    try:
        result = harness.run("research-families")
    except Exception as exc:  # noqa: BLE001 — catalog load is fail-closed
        return FamilyCatalog.unavailable(_short_reason(exc))
    return _catalog_from_result(result, cache)


def resolve_families(
    goal: str,
    catalog: FamilyCatalog,
    *,
    aliases: Mapping[str, str] | None = None,
) -> FamilySelection:
    """Classify family mentions in ``goal``.

    Matching is whole-word (and whole-phrase). Underscored names also match
    their space and hyphen spellings. ``family:<name>`` tokens are always
    classified; a free-text word that is not a catalog name or an alias is
    not a mention. Longest phrase wins. Duplicate names keep the first hit.
    An unavailable catalog sends every mention to ``unknown``.
    """
    table: Mapping[str, str] = DEFAULT_ALIASES if aliases is None else aliases
    forced_unknown = not catalog.available
    eligible: list[FamilyMention] = []
    retired: list[FamilyMention] = []
    unknown: list[FamilyMention] = []
    ordered: list[FamilyMention] = []
    for hit in _ordered_hits(goal.lower(), catalog, table):
        mention = _mention_for(hit, catalog, forced_unknown=forced_unknown)
        if mention is None:
            continue
        ordered.append(mention)
        _store_mention(mention, eligible, retired, unknown)
    source = "unavailable" if forced_unknown else catalog.source
    return FamilySelection(
        eligible=tuple(eligible),
        retired=tuple(retired),
        unknown=tuple(unknown),
        catalog_source=source,
        ordered=tuple(ordered),
    )


def families_prompt_block(selection: FamilySelection, catalog: FamilyCatalog) -> str:
    """Prompt text for a goal that mentioned families. Empty when it did not."""
    if not selection.mentioned:
        return ""
    lines = [
        f"RESEARCH FAMILIES (catalog v{catalog.catalog_version}, source {catalog.source}):",
        *selection.explain(),
    ]
    if not catalog.execution_constraint:
        lines.append(_EXECUTION_NOTE)
    return "\n".join(lines)


def _as_object(text: str) -> dict[str, Any]:
    try:
        payload: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"family catalog JSON is not valid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("family catalog JSON must be an object")
    return {str(key): item for key, item in payload.items()}


def _require_schema(payload: Mapping[str, Any]) -> int:
    version = payload.get("schema_version")
    if (
        isinstance(version, int)
        and not isinstance(version, bool)
        and version == CATALOG_SCHEMA_VERSION
    ):
        return version
    raise ValueError(
        f"unsupported family catalog schema {version!r}; expected {CATALOG_SCHEMA_VERSION}"
    )


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"family catalog field {field_name} must be an int")
    return value


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"family catalog field {field_name} must be a string")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"family catalog field {field_name} must be a bool")
    return value


def _retired_membership(
    payload: Mapping[str, Any], sample: list[str], count: int
) -> frozenset[str]:
    """Return the full retired set, or empty when the document omitted it.

    A present ``retired`` list must be sorted, unique, the same length as
    ``retired_count``, and prefixed by ``retired_sample`` (at most 20 names).
    """
    if "retired" not in payload:
        return frozenset()
    retired = _require_str_list(payload.get("retired"), "retired")
    if len(retired) != count:
        raise ValueError("family catalog field retired_count must equal len(retired)")
    if sample != retired[:20]:
        raise ValueError("family catalog field retired_sample must be the first 20 retired names")
    if len(set(retired)) != len(retired):
        raise ValueError("family catalog field retired must not contain duplicates")
    if retired != sorted(retired):
        raise ValueError("family catalog field retired must be sorted")
    return frozenset(retired)


def _require_str_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"family catalog field {field_name} must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"family catalog field {field_name} must be a list of strings")
        items.append(item)
    return items


def _runner_fields(payload: Mapping[str, Any]) -> tuple[bool, str]:
    runner = payload.get("runner")
    if not isinstance(runner, dict):
        raise ValueError("family catalog field runner must be an object")
    fields = {str(key): item for key, item in runner.items()}
    constrained = _require_bool(fields.get("execution_constraint"), "runner.execution_constraint")
    note = _require_str(fields.get("note"), "runner.note")
    return constrained, note


def _short_reason(text: object) -> str:
    raw = str(text).strip()
    return raw[:_REASON_LIMIT] or "research-families failed"


def _cached_catalog(cache: dict[str, FamilyCatalog]) -> FamilyCatalog | None:
    current = cache.get("catalog")
    if current is None:
        return None
    if current.source == "harness":
        current = replace(current, source="cached")
        cache["catalog"] = current
    return current


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _catalog_from_result(result: Any, cache: dict[str, FamilyCatalog] | None) -> FamilyCatalog:
    exit_code = getattr(result, "exit_code", 1)
    stderr = _as_text(getattr(result, "stderr", ""))
    if exit_code != 0:
        reason = stderr or f"research-families exited {exit_code}"
        return FamilyCatalog.unavailable(_short_reason(reason))
    try:
        catalog = FamilyCatalog.from_json(_as_text(getattr(result, "stdout", "")))
    except Exception as exc:  # noqa: BLE001 — malformed catalog is unavailable
        return FamilyCatalog.unavailable(_short_reason(exc))
    if cache is not None:
        cache["catalog"] = catalog
    return catalog


def _explain_mention(mention: FamilyMention) -> str:
    if mention.state is FamilyState.REQUIRED:
        return f"family '{mention.name}' — eligible (required)"
    if mention.state is FamilyState.OPTIONAL:
        return f"family '{mention.name}' — eligible (optional)"
    if mention.state is FamilyState.RETIRED:
        return (
            f"family '{mention.name}' — retired; new selection blocked "
            "(historical receipts still verify)"
        )
    return f"'{mention.name}' — not a known family; no suitable family"


def _forms(name: str) -> tuple[str, ...]:
    lowered = name.lower()
    if "_" not in lowered:
        return (lowered,)
    return (lowered, lowered.replace("_", " "), lowered.replace("_", "-"))


def _classification_names(catalog: FamilyCatalog) -> frozenset[str]:
    return catalog.required | catalog.live_optional | catalog.retired | catalog.retired_sample


def _surface_map(catalog: FamilyCatalog, aliases: Mapping[str, str]) -> dict[str, str]:
    surfaces: dict[str, str] = {}
    for name in sorted(_classification_names(catalog)):
        for form in _forms(name):
            surfaces.setdefault(form, name)
    for phrase, target in aliases.items():
        surfaces.setdefault(phrase.lower(), target)
    return surfaces


def _alias_key(aliases: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((phrase.lower(), target) for phrase, target in aliases.items()))


def _max_tokens(surfaces: dict[str, str]) -> int:
    if not surfaces:
        return 1
    return max(len(_WORD.findall(phrase)) for phrase in surfaces)


def _matcher(catalog: FamilyCatalog, aliases: Mapping[str, str]) -> _Matcher:
    key: _CacheKey = (
        catalog.required,
        catalog.live_optional,
        _classification_names(catalog),
        _alias_key(aliases),
    )
    cached = _MATCHER_CACHE.get(key)
    if cached is not None:
        return cached
    surfaces = _surface_map(catalog, aliases)
    compiled = (surfaces, _max_tokens(surfaces))
    _MATCHER_CACHE[key] = compiled
    return compiled


def _explicit_hits(goal: str) -> tuple[_Hit, ...]:
    return tuple(
        _Hit(
            start=match.start(),
            end=match.end(),
            name=match.group(1),
            matched_text=match.group(0),
            explicit=True,
        )
        for match in _EXPLICIT_TOKEN.finditer(goal)
    )


def _blocked(start: int, end: int, spans: tuple[tuple[int, int], ...]) -> bool:
    return any(start < right and end > left for left, right in spans)


def _contiguous_surface(
    goal: str, tokens: list[tuple[int, int, str]], start: int, end: int
) -> str | None:
    parts = [tokens[start][2]]
    for index in range(start, end):
        gap = goal[tokens[index][1] : tokens[index + 1][0]]
        if gap not in {" ", "-"}:
            return None
        parts.append(gap)
        parts.append(tokens[index + 1][2])
    return "".join(parts)


def _longest_phrase(
    goal: str,
    tokens: list[tuple[int, int, str]],
    index: int,
    surfaces: dict[str, str],
    max_tokens: int,
) -> tuple[int, int, str, str, int] | None:
    """Longest catalog surface starting at ``tokens[index]``.

    The tuple is ``(start, end, name, surface, next_index)``. A longer window
    replaces a shorter one so ``weighted conformal`` wins over ``conformal``.
    """
    last = min(len(tokens), index + max_tokens)
    found: tuple[int, int, str, str, int] | None = None
    for end in range(index, last):
        surface = _contiguous_surface(goal, tokens, index, end)
        if surface is None:
            break
        name = surfaces.get(surface)
        if name is not None:
            found = (tokens[index][0], tokens[end][1], name, surface, end + 1)
    return found


def _phrase_hits(
    goal: str,
    surfaces: dict[str, str],
    max_tokens: int,
    spans: tuple[tuple[int, int], ...],
) -> tuple[_Hit, ...]:
    if not surfaces:
        return ()
    tokens = [(match.start(), match.end(), match.group()) for match in _WORD.finditer(goal)]
    hits: list[_Hit] = []
    index = 0
    while index < len(tokens):
        found = _longest_phrase(goal, tokens, index, surfaces, max_tokens)
        if found is None:
            index += 1
            continue
        start, end, name, surface, next_index = found
        if not _blocked(start, end, spans):
            hits.append(
                _Hit(
                    start=start,
                    end=end,
                    name=name,
                    matched_text=surface,
                    explicit=False,
                )
            )
        index = next_index
    return tuple(hits)


def _ordered_hits(
    goal: str, catalog: FamilyCatalog, aliases: Mapping[str, str]
) -> tuple[_Hit, ...]:
    explicit = _explicit_hits(goal)
    spans = tuple((hit.start, hit.end) for hit in explicit)
    surfaces, max_tokens = _matcher(catalog, aliases)
    phrases = _phrase_hits(goal, surfaces, max_tokens, spans)
    ordered = sorted((*explicit, *phrases), key=lambda hit: (hit.start, hit.end))
    seen: set[str] = set()
    unique: list[_Hit] = []
    for hit in ordered:
        if hit.name in seen:
            continue
        seen.add(hit.name)
        unique.append(hit)
    return tuple(unique)


def _mention_for(
    hit: _Hit, catalog: FamilyCatalog, *, forced_unknown: bool
) -> FamilyMention | None:
    state = FamilyState.UNKNOWN if forced_unknown else catalog.state_of(hit.name)
    if state is FamilyState.UNKNOWN and not hit.explicit and not forced_unknown:
        return None
    return FamilyMention(name=hit.name, state=state, matched_text=hit.matched_text)


def _store_mention(
    mention: FamilyMention,
    eligible: list[FamilyMention],
    retired: list[FamilyMention],
    unknown: list[FamilyMention],
) -> None:
    if mention.state is FamilyState.REQUIRED or mention.state is FamilyState.OPTIONAL:
        eligible.append(mention)
        return
    if mention.state is FamilyState.RETIRED:
        retired.append(mention)
        return
    unknown.append(mention)


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "DEFAULT_ALIASES",
    "FamilyCatalog",
    "FamilyMention",
    "FamilySelection",
    "FamilyState",
    "families_prompt_block",
    "load_catalog",
    "resolve_families",
]
