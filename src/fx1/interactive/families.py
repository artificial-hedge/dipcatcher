"""Planner-facing research-family catalog.

Loaded from the harness command ``research-families``. Goals are classified
against required families, live optional families, and the full retired set
when the document includes ``retired``. A document that only carries
``retired_sample`` classifies that sample; every other name is UNKNOWN.
UNKNOWN and RETIRED both fail closed for new selection.

``execution_constraint`` reports whether the research runner can execute a
chosen subset. It is false today: named-family goals cannot launch an
unconstrained scorecard automatically.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
from types import MappingProxyType
from typing import Any, cast

CATALOG_SCHEMA_VERSION = 2

_EXPLICIT_TOKEN = re.compile(r"\bfamily:([a-z0-9_]+)\b")
_LOOKUP_WORD = re.compile(r"[a-z0-9]+")
_FAMILY_NAME = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")
_LOOKUP_MAX_PARTS = 5
_LOOKUP_MAX_HASHES = 300
_REASON_LIMIT = 200
_UNAVAILABLE_LINE = (
    "family catalog unavailable — family eligibility could not be checked; "
    "no family-level claims will be made (fail-closed)"
)
_EXECUTION_NOTE = (
    "note: the research runner has no per-family selector; automatic research is "
    "excluded when a goal names a family, while explicit research runs the "
    "configured scorecard."
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
    """One planner-facing snapshot; ``live_optional`` is the justified subset."""

    schema_version: int
    catalog_version: int
    receipt_schema_version: int
    required: frozenset[str]
    live_optional: frozenset[str]
    retired_count: int
    retired_sample: frozenset[str]
    order: tuple[str, ...]
    generated_at: str
    live_optional_total: int = 0
    exploratory_count: int = 0
    omitted_matches: frozenset[str] = frozenset()
    lookup_goal_sha256: str = ""
    source: str = "harness"
    execution_constraint: bool = False
    note: str = ""
    # Empty means the document omitted ``retired`` and only the sample classifies.
    retired: frozenset[str] = frozenset()

    @classmethod
    def from_json(cls, text: str, *, goal: str | None = None) -> FamilyCatalog:
        """Parse a ``research-families`` JSON document.

        Raises ``ValueError`` with ``unsupported family catalog schema`` when
        ``schema_version`` is not :data:`CATALOG_SCHEMA_VERSION`. Every other
        malformed field raises ``ValueError`` and is rejected (fail closed).
        A goal requires a matching, bounded lookup response before research
        can be planned or executed for that exact goal.
        """
        payload = _as_object(text)
        schema_version = _require_schema(payload)
        constrained, note = _runner_fields(payload)
        sample = _require_str_list(payload.get("retired_sample"), "retired_sample")
        count = _require_int(payload.get("retired_count"), "retired_count")
        return cls(
            schema_version=schema_version,
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
            live_optional_total=live_optional_total,
            exploratory_count=exploratory_count,
            omitted_matches=omitted_matches,
            lookup_goal_sha256=goal_sha256(goal) if goal is not None else "",
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

    def covers_goal(self, goal: str) -> bool:
        """True only for a validated lookup bound to this exact goal."""
        return self.available and self.lookup_goal_sha256 == goal_sha256(goal)

    def describe(self) -> str:
        if not self.available:
            return f"family catalog unavailable: {self.note}"
        summary = (
            f"family catalog v{self.catalog_version} "
            f"(receipt schema v{self.receipt_schema_version}): "
            f"{len(self.required)} required · {len(self.live_optional)} selectable optional · "
            f"{self.exploratory_count} exploratory of {self.live_optional_total} live optional · "
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


def load_catalog(
    harness: Any, *, cache: dict[str, FamilyCatalog] | None = None, goal: str | None = None
) -> FamilyCatalog:
    """Load ``research-families`` through ``harness``. Never raises.

    A base-catalog cache hit returns the stored catalog with ``source="cached"``.
    Goal-specific lookups always call the harness and only cache a stripped
    base catalog. A non-zero exit, interruption, parse failure, or exception
    from the harness becomes :meth:`FamilyCatalog.unavailable` using the first
    200 characters of the failure reason.
    """
    if cache is not None and goal is None:
        cached = _cached_catalog(cache)
        if cached is not None:
            return cached
    try:
        extra_args = ["--lookup-hashes", goal_lookup_hashes(goal)] if goal is not None else None
        result = (
            harness.run("research-families", extra_args=extra_args)
            if extra_args is not None
            else harness.run("research-families")
        )
    except Exception as exc:  # noqa: BLE001 — catalog load is fail-closed
        return FamilyCatalog.unavailable(_short_reason(exc))
    return _catalog_from_result(result, cache, goal=goal)


def resolve_families(
    goal: str,
    catalog: FamilyCatalog,
    *,
    aliases: Mapping[str, str] | None = None,
) -> FamilySelection:
    """Classify family mentions in ``goal``.

    Matching is whole-word (and whole-phrase). Underscored names also match
    their space and hyphen spellings. A validated goal lookup also recognizes
    eligible and omitted names across punctuation-separated words. The
    ``family:<name>`` tokens are always classified; a free-text word that is
    not a catalog name or an alias is not a mention. Longest phrase wins.
    Duplicate names keep the first hit. An unavailable catalog sends every
    mention to ``unknown``.
    """
    table: Mapping[str, str] = DEFAULT_ALIASES if aliases is None else aliases
    forced_unknown = not catalog.available
    eligible: list[FamilyMention] = []
    retired: list[FamilyMention] = []
    unknown: list[FamilyMention] = []
    ordered: list[FamilyMention] = []
    hits = list(_ordered_hits(goal.lower(), catalog, table))
    # The lab lookup tokenizes punctuation as word separators. A slash or
    # newline can therefore match a catalog name even when the human-readable
    # phrase matcher (which accepts only spaces, hyphens, and underscores)
    # does not. The validated lookup must still block that named research run.
    if catalog.covers_goal(goal):
        candidate_hashes = set(goal_lookup_hashes(goal).split(","))
        mentioned_names = {hit.name for hit in hits}
        hash_matched = {
            name
            for name in catalog.eligible()
            if sha256(name.encode("utf-8")).hexdigest() in candidate_hashes
        }
        for name in sorted((catalog.omitted_matches | hash_matched) - mentioned_names):
            hits.append(
                _Hit(start=len(goal), end=len(goal), name=name, matched_text=name, explicit=False)
            )
    for hit in hits:
        mention = _mention_for(hit, catalog, forced_unknown=forced_unknown)
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


def goal_sha256(goal: str) -> str:
    """Bind a lookup to the exact goal used by both planning and execution."""
    return sha256(goal.encode("utf-8")).hexdigest()


def goal_lookup_hashes(goal: str) -> str:
    """Hash bounded candidate family names without exposing the goal in argv."""
    if len(goal.encode("utf-8")) > 2048:
        raise ValueError("family lookup goal exceeds 2048 UTF-8 bytes")
    words = _LOOKUP_WORD.findall(goal.lower())
    if not words:
        raise ValueError("family lookup goal contains no searchable words")
    hashes: set[str] = set()
    for start in range(len(words)):
        for width in range(1, min(_LOOKUP_MAX_PARTS, len(words) - start) + 1):
            name = "_".join(words[start : start + width])
            hashes.add(sha256(name.encode("ascii")).hexdigest())
            if len(hashes) > _LOOKUP_MAX_HASHES:
                raise ValueError("family lookup exceeds 300 name candidates")
    return ",".join(sorted(hashes))


def _lookup_matches(
    payload: Mapping[str, Any], goal: str | None, selectable: frozenset[str]
) -> frozenset[str]:
    lookup = payload.get("lookup")
    if goal is None:
        if "lookup" in payload:
            raise ValueError("unexpected goal-specific family lookup")
        return frozenset()
    if not isinstance(lookup, dict):
        raise ValueError("goal-specific family lookup missing")
    request = goal_lookup_hashes(goal)
    expected = sha256(request.encode("ascii")).hexdigest()
    if lookup.get("request_sha256") != expected:
        raise ValueError("goal-specific family lookup request mismatch")
    raw_matches = lookup.get("omitted_matches")
    if not isinstance(raw_matches, list) or len(raw_matches) > _LOOKUP_MAX_HASHES:
        raise ValueError("goal-specific family lookup matches malformed")
    requested = set(request.split(","))
    names: list[str] = []
    for name in raw_matches:
        if (
            not isinstance(name, str)
            or _FAMILY_NAME.fullmatch(name) is None
            or len(name.split("_")) > _LOOKUP_MAX_PARTS
            or sha256(name.encode("ascii")).hexdigest() not in requested
            or name in selectable
        ):
            raise ValueError("goal-specific family lookup returned an invalid name")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("goal-specific family lookup returned duplicate names")
    return frozenset(names)


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


def _catalog_from_result(
    result: Any, cache: dict[str, FamilyCatalog] | None, *, goal: str | None
) -> FamilyCatalog:
    if bool(getattr(result, "cancelled", False)):
        return FamilyCatalog.unavailable("research-families cancelled")
    if bool(getattr(result, "timed_out", False)):
        return FamilyCatalog.unavailable("research-families timed out")
    exit_code = getattr(result, "exit_code", 1)
    stderr = _as_text(getattr(result, "stderr", ""))
    if exit_code != 0:
        reason = stderr or f"research-families exited {exit_code}"
        return FamilyCatalog.unavailable(_short_reason(reason))
    try:
        catalog = FamilyCatalog.from_json(_as_text(getattr(result, "stdout", "")), goal=goal)
    except Exception as exc:  # noqa: BLE001 — malformed catalog is unavailable
        return FamilyCatalog.unavailable(_short_reason(exc))
    if cache is not None:
        cache["catalog"] = replace(catalog, omitted_matches=frozenset(), lookup_goal_sha256="")
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
    return f"'{mention.name}' — not a planner-selectable family; no suitable family"


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


def _mention_for(hit: _Hit, catalog: FamilyCatalog, *, forced_unknown: bool) -> FamilyMention:
    state = FamilyState.UNKNOWN if forced_unknown else catalog.state_of(hit.name)
    # Aliases in the curated map name research families. If a catalog omits an
    # alias target, keep the mention UNKNOWN so it cannot launch the full
    # unconstrained scorecard through a missing-name path.
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


class FamilyClientError(ValueError):
    """The family surface response broke the read-only contract."""


SURFACE_SCHEMA = "dipcatcher.family_surface.v1"

_FAMILY_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_STDERR_LIMIT = 500
_MAX_GOAL = 4096
_MAX_FAMILIES = 23
_MISSING = object()

Runner = Callable[[list[str]], tuple[int, str, str]]


def _stderr_snippet(stderr: str) -> str:
    if len(stderr) <= _STDERR_LIMIT:
        return stderr
    return stderr[:_STDERR_LIMIT]


def _parse_json(stdout: str) -> Any:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise FamilyClientError(f"family surface returned invalid JSON: {exc}") from exc


def _require_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FamilyClientError("family surface payload must be a JSON object")
    schema = payload.get("schema")
    if schema != SURFACE_SCHEMA:
        raise FamilyClientError(
            f"family surface schema must be {SURFACE_SCHEMA!r} (got {schema!r})"
        )
    return payload


def _require_executable_required(item: Any, *, where: str) -> None:
    """Reject a record the harness should already have filtered out."""
    if not isinstance(item, dict):
        raise FamilyClientError(f"{where} must be an object")
    _reject_item_schema(item, where)
    if item.get("executable") is not True:
        raise FamilyClientError(f"{where} must have executable true")
    qualification = item.get("qualification")
    if qualification != "required":
        raise FamilyClientError(f"{where} qualification must be 'required' (got {qualification!r})")
    family_id = item.get("id")
    if not isinstance(family_id, str) or _FAMILY_ID.fullmatch(family_id) is None:
        raise FamilyClientError(f"{where} id must match {_FAMILY_ID.pattern}")


def _reject_item_schema(item: Mapping[str, Any], where: str) -> None:
    if "schema" not in item:
        return
    schema = item.get("schema")
    if schema != SURFACE_SCHEMA:
        raise FamilyClientError(f"{where} schema must be {SURFACE_SCHEMA!r} (got {schema!r})")


def _require_select_shape(payload: dict[str, Any]) -> None:
    missing = [key for key in ("schema", "selected", "exclusions") if key not in payload]
    if missing:
        raise FamilyClientError(f"select payload missing keys: {', '.join(missing)}")
    selected = payload["selected"]
    if not isinstance(selected, list):
        raise FamilyClientError("select payload 'selected' must be a list")
    _require_exclusions(payload["exclusions"])
    for index, item in enumerate(selected):
        _require_executable_required(item, where=f"selected[{index}]")


def _require_exclusions(exclusions: Any) -> None:
    if not isinstance(exclusions, list):
        raise FamilyClientError("select payload 'exclusions' must be a list")
    for index, item in enumerate(exclusions):
        _require_exclusion(item, index)


def _require_exclusion(item: Any, index: int) -> None:
    if not isinstance(item, dict):
        raise FamilyClientError(f"exclusions[{index}] must be an object")
    family_id = item.get("id")
    reason = item.get("reason")
    if not isinstance(family_id, str) or _FAMILY_ID.fullmatch(family_id) is None:
        raise FamilyClientError(f"exclusions[{index}] id is invalid")
    if not isinstance(reason, str) or not reason.strip():
        raise FamilyClientError(f"exclusions[{index}] reason must be non-empty text")


def _reject_present_schema(payload: Any) -> None:
    """A missing schema stays tolerated; a present wrong schema does not."""
    if not isinstance(payload, dict) or "schema" not in payload:
        return
    schema = payload.get("schema")
    if schema != SURFACE_SCHEMA:
        raise FamilyClientError(
            f"family surface schema must be {SURFACE_SCHEMA!r} (got {schema!r})"
        )


def _preview(text: object) -> str:
    if not isinstance(text, str):
        return repr(type(text).__name__)
    if len(text) <= 80:
        return repr(text)
    return repr(f"{text[:32]}...({len(text)} chars)")


def _require_goal_arg(goal: str) -> None:
    if not isinstance(goal, str) or "\x00" in goal or not goal.strip():
        raise FamilyClientError("goal must be non-empty text without NUL")
    if len(goal) > _MAX_GOAL:
        raise FamilyClientError(f"goal exceeds {_MAX_GOAL} characters")


def _require_max_arg(max_families: int) -> None:
    if isinstance(max_families, bool) or not isinstance(max_families, int):
        raise FamilyClientError("max_families must be an integer")
    if max_families < 0 or max_families > _MAX_FAMILIES:
        raise FamilyClientError(f"max_families must be an integer in 0..{_MAX_FAMILIES}")


def _require_family_arg(family_id: str) -> None:
    if not isinstance(family_id, str) or _FAMILY_ID.fullmatch(family_id) is None:
        shown = _preview(family_id)
        pattern = _FAMILY_ID.pattern
        raise FamilyClientError(f"family id must match {pattern} (got {shown})")


def _family_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        families = payload.get("families")
        if isinstance(families, list):
            return families
    raise FamilyClientError("list payload must be a list or an object with a families list")


def _runner_triple(result: object) -> tuple[int, str, str]:
    if not isinstance(result, tuple) or len(result) != 3:
        raise FamilyClientError("family runner must return (exit_code, stdout, stderr)")
    code: object = result[0]
    stdout: object = result[1]
    stderr: object = result[2]
    if isinstance(code, bool) or not isinstance(code, int):
        raise FamilyClientError("family runner exit_code must be an int")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise FamilyClientError("family runner stdout and stderr must be strings")
    return code, stdout, stderr


class FamilyClient:
    """Call the read-only family surface through an injected runner."""

    def __init__(self, runner: Runner) -> None:
        self._runner = runner

    def select(self, goal: str, *, max_families: int = 4) -> dict[str, Any]:
        """Choose qualified families for ``goal``.

        The parsed object is returned unchanged when it matches the contract.
        A selected record that is not executable and required is a contract
        break: it is rejected, not dropped.
        """
        _require_goal_arg(goal)
        _require_max_arg(max_families)
        argv = ["--select", goal, "--max", str(max_families)]
        payload = _require_object(self._payload(argv))
        _require_select_shape(payload)
        return payload

    def status(self, family_id: str) -> dict[str, Any]:
        """Return the surface record for one family id."""
        _require_family_arg(family_id)
        payload = _require_object(self._payload(["--id", family_id]))
        got = payload.get("id")
        if "id" not in payload or not isinstance(got, str):
            raise FamilyClientError("status payload must contain schema and id")
        if _FAMILY_ID.fullmatch(got) is None or got != family_id:
            raise FamilyClientError(f"status id {got!r} does not match requested {family_id!r}")
        return payload

    def list_executable(self) -> list[dict[str, Any]]:
        """Return families the surface marks executable and required."""
        payload = self._payload([])
        _reject_present_schema(payload)
        items = _family_items(payload)
        for index, item in enumerate(items):
            _require_executable_required(item, where=f"families[{index}]")
        return cast(list[dict[str, Any]], items)

    def _payload(self, argv: list[str]) -> Any:
        _code, stdout, _stderr = self._run(argv)
        return _parse_json(stdout)

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        code, stdout, stderr = _runner_triple(self._runner(argv))
        if code != 0:
            raise FamilyClientError(f"family surface exited {code}: {_stderr_snippet(stderr)}")
        return code, stdout, stderr


def selected_ids(payload: dict[str, Any]) -> list[str]:
    """Ids from ``payload['selected']`` in order, keeping the first of each id."""
    if not isinstance(payload, dict):
        raise FamilyClientError("selected_ids payload must be an object")
    selected = payload.get("selected")
    if not isinstance(selected, list):
        raise FamilyClientError("selected_ids payload must contain a selected list")
    ids: list[str] = []
    for index, item in enumerate(selected):
        if not isinstance(item, dict):
            raise FamilyClientError(f"selected[{index}] must be an object")
        family_id = item.get("id")
        if not isinstance(family_id, str) or _FAMILY_ID.fullmatch(family_id) is None:
            raise FamilyClientError(f"selected[{index}] id is invalid")
        ids.append(family_id)
    return list(dict.fromkeys(ids))


def harness_family_runner(harness: Any) -> Runner:
    """Return a runner that calls ``harness.run('families', extra_args)``."""

    def runner(argv: list[str]) -> tuple[int, str, str]:
        result = harness.run("families", argv)
        code = getattr(result, "exit_code", _MISSING)
        if code is _MISSING:
            raise FamilyClientError("harness families result has no exit_code")
        stdout = getattr(result, "stdout", _MISSING)
        stderr = getattr(result, "stderr", _MISSING)
        if isinstance(code, bool) or not isinstance(code, int):
            raise FamilyClientError("harness families exit_code must be an int")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise FamilyClientError("harness families stdout and stderr must be strings")
        return code, stdout, stderr

    return runner

__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "DEFAULT_ALIASES",
    "FamilyCatalog",
    "FamilyMention",
    "FamilySelection",
    "FamilyState",
    "families_prompt_block",
    "goal_lookup_hashes",
    "goal_sha256",
    "load_catalog",
    "resolve_families",
    "FamilyClient",
    "FamilyClientError",
    "SURFACE_SCHEMA",
    "harness_family_runner",
    "selected_ids",
]
