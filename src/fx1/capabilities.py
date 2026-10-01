"""Searchable, generated skill/plugin/feature recipes for the fx-1 harness.

The bundled seed contains one million distinct catalog records. Records are
compact deterministic recipes over the real harness command registry and
datasource registry; they do not claim to be one million independent code
implementations. Operators can add reviewed JSONL metadata packs through
``FX1_CAPABILITY_ROOTS``. Packs may only point at already registered commands
or datasource adapters, so catalog metadata cannot execute arbitrary code.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import struct
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from fx1.harness import HARNESS_REGISTRY, HarnessCommand

CAPABILITY_SCHEMA = "fx1.capabilities/v1"
PACK_SCHEMA = "fx1.capability-pack-entry/v1"
SEED_COUNT = 1_000_001
SEED_VERSION = 1
SEED_MAGIC = b"FX1CAP1\0"
SEED_PATH = Path(__file__).with_name("capability_seed_v1.bin.gz")
CAPABILITY_ROOTS_ENV = "FX1_CAPABILITY_ROOTS"
MAX_PAGE_SIZE = 100
MAX_PACK_ENTRY_BYTES = 16_384
_HEADER = struct.Struct(">8sII32s")
_ROW = struct.Struct(">I8BH")
_NONE = 255

CapabilityKind = Literal["skill", "plugin", "feature"]

KINDS: tuple[CapabilityKind, ...] = ("skill", "plugin", "feature")
MARKETS = ("cn", "hk", "us", "global", "crypto")
ASSETS = (
    "equity",
    "fund",
    "bond",
    "index",
    "macro",
    "news",
    "filings",
    "research",
    "enterprise",
    "crypto",
    "sentiment",
    "industry_chain",
)

WORKFLOWS_BY_ROLE: dict[str, tuple[str, ...]] = {
    "data_engine": (
        "inspect-source-availability",
        "route-point-in-time-data",
        "ingest-with-hash-manifest",
        "build-causal-features",
        "build-time-aligned-labels",
        "check-data-quality",
        "trace-source-lineage",
        "prepare-reproducible-dataset",
    ),
    "evaluation": (
        "score-forecast-calibration",
        "compare-proper-scores",
        "run-walk-forward-evaluation",
        "audit-multiple-testing",
        "inspect-order-book-evidence",
        "evaluate-feature-families",
        "produce-research-report",
        "replay-with-explicit-costs",
        "check-synthetic-correctness",
        "verify-evaluation-inputs",
    ),
    "verification": (
        "verify-immutable-receipt",
        "check-point-in-time-contract",
        "audit-data-provenance",
        "validate-promotion-gates",
        "inspect-runtime-health",
        "verify-model-artifact",
        "check-reproducibility",
        "review-evidence-eligibility",
    ),
    "model_training": (
        "build-gate-passed-curriculum",
        "train-with-eval-before-fit",
        "optimize-under-validation-gates",
        "record-training-provenance",
        "run-shadow-paper-simulation",
        "review-training-costs",
    ),
}
PLUGIN_WORKFLOWS = (
    "describe-available-apis",
    "probe-local-availability",
    "route-by-market-and-asset",
    "fetch-with-observation-date",
    "record-payload-hash",
    "preserve-source-lineage",
    "check-credential-names-only",
    "use-host-mcp-runtime-when-required",
)
WORKFLOWS = tuple(dict.fromkeys((*sum(WORKFLOWS_BY_ROLE.values(), ()), *PLUGIN_WORKFLOWS)))
DATA_DISCIPLINES = (
    "receipt-bound",
    "point-in-time",
    "synthetic-correctness-only",
    "availability-time-aware",
    "walk-forward",
    "proper-scoring",
    "source-lineage-preserved",
    "fail-closed",
)
OUTPUTS = (
    "json-summary",
    "research-report",
    "immutable-receipt",
    "scorecard",
    "dataset-manifest",
    "feature-panel",
    "verification-result",
    "tool-call-plan",
)


class CapabilityPackEntry(BaseModel):
    """One operator-authored catalog card; it cannot name executable code."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["fx1.capability-pack-entry/v1"] = Field(
        default=PACK_SCHEMA, alias="schema"
    )
    id: str = Field(min_length=1, max_length=128)
    kind: CapabilityKind
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4_000)
    command: str | None = None
    source: str | None = None
    tags: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*", value):
            raise ValueError("id may contain only letters, digits, '.', '_', ':', '/', and '-'")
        return value


class CapabilityCatalogError(ValueError):
    """The bundled seed or an operator-provided metadata pack is invalid."""


class CapabilitySearchArguments(BaseModel):
    """Validated arguments for the AI-facing discovery tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = ""
    kind: CapabilityKind | None = None
    command: str | None = None
    source: str | None = None
    market: str | None = None
    asset: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=MAX_PAGE_SIZE)


CAPABILITY_SEARCH_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_capabilities",
        "description": (
            "Search skills, registered datasource plugins, and feature recipes "
            "available through the dipcatcher harness. Results are discovery "
            "metadata, not research evidence."
        ),
        "parameters": CapabilitySearchArguments.model_json_schema(),
    },
}


@lru_cache(maxsize=1)
def _commands() -> tuple[HarnessCommand, ...]:
    return tuple(command for command in HARNESS_REGISTRY if command.name != "capability-search")


@lru_cache(maxsize=1)
def _source_specs() -> list[Any]:
    from fx1.data.sources.registry import list_sources

    return list_sources()


def _layout_digest() -> bytes:
    """Bind dimension codes to the registries used when the seed was written."""
    payload = {
        "commands": [
            (command.name, command.role.value, command.description) for command in _commands()
        ],
        "sources": [
            (source.name, source.kind.value, source.markets, source.assets)
            for source in _source_specs()
        ],
        "kinds": KINDS,
        "markets": MARKETS,
        "assets": ASSETS,
        "workflows_by_role": WORKFLOWS_BY_ROLE,
        "plugin_workflows": PLUGIN_WORKFLOWS,
        "disciplines": DATA_DISCIPLINES,
        "outputs": OUTPUTS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _seed_row(index: int) -> tuple[int, ...]:
    """Return deterministic dimension codes for one generated catalog record."""
    commands = _commands()
    sources = _source_specs()
    kind_index = index % len(KINDS)
    sequence = index // len(KINDS)

    if KINDS[kind_index] == "plugin":
        source_index = sequence % len(sources)
        source = sources[source_index]
        market = source.markets[(sequence // len(sources)) % len(source.markets)]
        asset = source.assets[
            (sequence // (len(sources) * len(source.markets))) % len(source.assets)
        ]
        workflow = PLUGIN_WORKFLOWS[
            (sequence // (len(sources) * len(source.markets) * len(source.assets)))
            % len(PLUGIN_WORKFLOWS)
        ]
        command_index = _NONE
    else:
        command_index = sequence % len(commands)
        command = commands[command_index]
        role_workflows = WORKFLOWS_BY_ROLE[command.role.value]
        workflow = role_workflows[(sequence // len(commands)) % len(role_workflows)]
        source_index = _NONE
        market = MARKETS[(sequence // (len(commands) * len(role_workflows))) % len(MARKETS)]
        asset = ASSETS[
            (sequence // (len(commands) * len(role_workflows) * len(MARKETS))) % len(ASSETS)
        ]

    discipline = DATA_DISCIPLINES[(sequence // 13) % len(DATA_DISCIPLINES)]
    output = OUTPUTS[(sequence // 71) % len(OUTPUTS)]
    variant = sequence % 65_536
    return (
        kind_index,
        command_index,
        source_index,
        MARKETS.index(market),
        ASSETS.index(asset),
        WORKFLOWS.index(workflow),
        DATA_DISCIPLINES.index(discipline),
        OUTPUTS.index(output),
        variant,
    )


def write_seed_catalog(path: Path = SEED_PATH, *, count: int = SEED_COUNT) -> int:
    """Write the deterministic binary seed as gzip-compressed dimension codes."""
    if count < 1 or count > 0xFFFFFFFF:
        raise ValueError("count must be in the uint32 range")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
                stream.write(_HEADER.pack(SEED_MAGIC, SEED_VERSION, count, _layout_digest()))
                for index in range(count):
                    stream.write(_ROW.pack(index, *_seed_row(index)))
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _entry_from_row(row: tuple[int, ...]) -> dict[str, Any]:
    (
        entry_id,
        kind_index,
        command_index,
        source_index,
        market_index,
        asset_index,
        workflow_index,
        discipline_index,
        output_index,
        variant,
    ) = row
    kind = KINDS[kind_index]
    market = MARKETS[market_index]
    asset = ASSETS[asset_index]
    workflow = WORKFLOWS[workflow_index]
    discipline = DATA_DISCIPLINES[discipline_index]
    output = OUTPUTS[output_index]
    unique_suffix = f"{entry_id:07d}"

    if kind == "plugin":
        source = _source_specs()[source_index]
        command_name = None
        role = "data_engine"
        name = f"plugin/{source.name}/{market}/{asset}/{workflow}/{discipline}/{output}/{unique_suffix}"
        entrypoint = (
            "MCP runtime required"
            if source.kind.value == "mcp"
            else f"fx1 sources describe {source.name}"
        )
        description = (
            f"Generated datasource recipe for {source.display} ({source.name}). "
            f"Use {workflow}; this adapter supports {market}/{asset}. Preserve "
            f"{discipline} and produce {output}. The source registry and its "
            "availability probe remain authoritative."
        )
    else:
        command = _commands()[command_index]
        source = None
        command_name = command.name
        role = command.role.value
        name = f"{kind}/{command.name}/{market}/{asset}/{workflow}/{discipline}/{output}/{unique_suffix}"
        entrypoint = f"dipcatcher {command.name}"
        description = (
            f"Generated {kind} recipe backed by registered command {command.name!r}. "
            f"Focus: {workflow} for {market}/{asset}. Respect {discipline}; "
            f"return {output}. Harness command: {command.description}"
        )

    return {
        "id": f"seed:{entry_id:07d}",
        "kind": kind,
        "name": name,
        "description": description,
        "command": command_name,
        "source": source.name if source is not None else None,
        "role": role,
        "market": market,
        "asset": asset,
        "workflow": workflow,
        "data_discipline": discipline,
        "output": output,
        "entrypoint": entrypoint,
        "generated": True,
        "metadata_trust": "repository-generated recipe",
        "origin": "dipcatcher-generated-capability-seed-v1",
        "variant": variant,
    }


def _iter_seed_rows(path: Path = SEED_PATH) -> Iterator[tuple[int, ...]]:
    try:
        stream = gzip.open(path, "rb")
    except OSError as exc:
        raise CapabilityCatalogError(
            f"capability seed is unavailable at {path}; regenerate it with "
            "scripts/seed_fx1_capabilities.py"
        ) from exc
    with stream:
        header = stream.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise CapabilityCatalogError("capability seed has a truncated header")
        magic, version, count, layout_digest = _HEADER.unpack(header)
        if magic != SEED_MAGIC or version != SEED_VERSION:
            raise CapabilityCatalogError("capability seed magic/version is unsupported")
        if layout_digest != _layout_digest():
            raise CapabilityCatalogError(
                "capability seed was generated from a different command/source registry; "
                "regenerate it with scripts/seed_fx1_capabilities.py"
            )
        for expected_id in range(count):
            packed = stream.read(_ROW.size)
            if len(packed) != _ROW.size:
                raise CapabilityCatalogError(f"capability seed ended before record {expected_id}")
            row = _ROW.unpack(packed)
            if row[0] != expected_id:
                raise CapabilityCatalogError(
                    f"capability seed record order breaks at {expected_id}"
                )
            yield row
        if stream.read(1):
            raise CapabilityCatalogError("capability seed contains trailing bytes")


def _pack_paths() -> Iterator[Path]:
    raw = os.environ.get(CAPABILITY_ROOTS_ENV, "")
    for item in raw.split(os.pathsep):
        if not item.strip():
            continue
        root = Path(item).expanduser()
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from sorted(root.glob("capabilities.jsonl"))


def _external_entries() -> Iterator[dict[str, Any]]:
    commands_by_name = {command.name: command for command in HARNESS_REGISTRY}
    sources_by_name = {source.name: source for source in _source_specs()}
    command_names = set(commands_by_name)
    source_names = set(sources_by_name)
    for path in _pack_paths():
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                if len(line.encode("utf-8")) > MAX_PACK_ENTRY_BYTES:
                    raise CapabilityCatalogError(
                        f"capability entry at {path}:{line_number} exceeds "
                        f"{MAX_PACK_ENTRY_BYTES} bytes"
                    )
                try:
                    raw_entry = json.loads(line)
                    entry = CapabilityPackEntry.model_validate(raw_entry)
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise CapabilityCatalogError(
                        f"invalid capability entry at {path}:{line_number}: {exc}"
                    ) from exc
                if entry.command is not None and entry.command not in command_names:
                    raise CapabilityCatalogError(
                        f"{path}:{line_number} references unregistered command {entry.command!r}"
                    )
                if entry.source is not None and entry.source not in source_names:
                    raise CapabilityCatalogError(
                        f"{path}:{line_number} references unregistered source {entry.source!r}"
                    )
                if entry.kind in {"skill", "feature"} and entry.command is None:
                    raise CapabilityCatalogError(
                        f"{path}:{line_number} {entry.kind} requires a registered command"
                    )
                if entry.kind == "plugin" and entry.source is None:
                    raise CapabilityCatalogError(f"{path}:{line_number} plugin requires a source")
                command = commands_by_name.get(entry.command) if entry.command is not None else None
                source_spec = (
                    sources_by_name.get(entry.source) if entry.source is not None else None
                )
                entrypoint = (
                    f"dipcatcher {entry.command}"
                    if entry.command is not None
                    else (
                        "MCP runtime required"
                        if source_spec is not None and source_spec.kind.value == "mcp"
                        else f"fx1 sources describe {entry.source}"
                    )
                )
                yield {
                    "id": f"external:{entry.id}",
                    "kind": entry.kind,
                    "name": entry.name,
                    "description": entry.description,
                    "command": entry.command,
                    "source": entry.source,
                    "role": command.role.value if command is not None else "data_engine",
                    "entrypoint": entrypoint,
                    "tags": entry.tags,
                    "generated": False,
                    "metadata_trust": "operator-provided; review before use",
                    "origin": str(path),
                }


def _search_text(entry: dict[str, Any]) -> str:
    values = (
        entry.get("id"),
        entry.get("kind"),
        entry.get("name"),
        entry.get("description"),
        entry.get("command"),
        entry.get("source"),
        entry.get("role"),
        entry.get("market"),
        entry.get("asset"),
        entry.get("workflow"),
        entry.get("data_discipline"),
        entry.get("output"),
        entry.get("entrypoint"),
        " ".join(entry.get("tags", [])),
    )
    return " ".join(str(value) for value in values if value).casefold()


def _row_matches(
    row: tuple[int, ...],
    *,
    query_terms: tuple[str, ...],
    kind: CapabilityKind | None,
    filters: dict[str, str | None],
) -> bool:
    (
        entry_id,
        kind_index,
        command_index,
        source_index,
        market_index,
        asset_index,
        workflow_index,
        discipline_index,
        output_index,
        _variant,
    ) = row
    row_kind = KINDS[kind_index]
    command = _commands()[command_index] if command_index != _NONE else None
    source = _source_specs()[source_index] if source_index != _NONE else None
    market = MARKETS[market_index]
    asset = ASSETS[asset_index]
    if kind is not None and row_kind != kind:
        return False
    if filters["command"] is not None and (
        command is None or command.name.casefold() != filters["command"]
    ):
        return False
    if filters["source"] is not None and (
        source is None or source.name.casefold() != filters["source"]
    ):
        return False
    if filters["market"] is not None and market.casefold() != filters["market"]:
        return False
    if filters["asset"] is not None and asset.casefold() != filters["asset"]:
        return False
    if not query_terms:
        return True

    searchable_parts = [
        f"seed:{entry_id:07d}",
        row_kind,
        market,
        asset,
        WORKFLOWS[workflow_index],
        DATA_DISCIPLINES[discipline_index],
        OUTPUTS[output_index],
    ]
    if command is not None:
        searchable_parts.extend(
            (
                "generated recipe registered command harness command dipcatcher",
                command.name,
                command.role.value,
                command.description,
            )
        )
    if source is not None:
        searchable_parts.extend(
            (
                "generated datasource recipe registered source adapter availability probe",
                source.name,
                source.display,
                source.notes,
                "mcp runtime required" if source.kind.value == "mcp" else "fx1 sources describe",
            )
        )
    searchable = " ".join(searchable_parts).casefold()
    return all(term in searchable for term in query_terms)


def search_capabilities(
    query: str = "",
    *,
    kind: CapabilityKind | None = None,
    command: str | None = None,
    source: str | None = None,
    market: str | None = None,
    asset: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict[str, Any]:
    """Return a bounded page of generated and operator-provided capability cards."""
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    query_terms = tuple(part.casefold() for part in re.findall(r"[\w.-]+", query))
    filters = {
        "command": command.casefold() if command else None,
        "source": source.casefold() if source else None,
        "market": market.casefold() if market else None,
        "asset": asset.casefold() if asset else None,
    }
    results: list[dict[str, Any]] = []
    matched = 0
    has_more = False

    for entry in _external_entries():
        if kind is not None and entry["kind"] != kind:
            continue
        if any(
            value is not None and str(entry.get(field) or "").casefold() != value
            for field, value in filters.items()
        ):
            continue
        searchable = _search_text(entry)
        if query_terms and not all(term in searchable for term in query_terms):
            continue
        if matched < offset:
            matched += 1
            continue
        if len(results) >= limit:
            has_more = True
            break
        results.append(entry)
        matched += 1

    if not has_more:
        for row in _iter_seed_rows():
            if not _row_matches(row, query_terms=query_terms, kind=kind, filters=filters):
                continue
            if matched < offset:
                matched += 1
                continue
            if len(results) >= limit:
                has_more = True
                break
            results.append(_entry_from_row(row))
            matched += 1

    return {
        "schema": CAPABILITY_SCHEMA,
        "query": query,
        "filters": {key: value for key, value in filters.items() if value is not None}
        | ({"kind": kind} if kind else {}),
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "seed_count": SEED_COUNT,
        "generated_entries_are_recipes": True,
        "market_evidence": False,
        "results": results,
    }
