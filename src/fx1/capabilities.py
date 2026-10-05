"""Searchable cards for individually loadable fx-1 extension modules.

The bundled seed contains one million deterministic discovery cards split
across a source file for every registered harness skill, datasource plugin,
and public point-in-time feature. A card resolves through its owning module to
the real fail-closed command, adapter, or feature builder; it does not claim
to be an independent market result or a new underlying algorithm. Operators
can add reviewed JSONL metadata packs through ``FX1_CAPABILITY_ROOTS``. Packs
may only point at already registered commands or datasource adapters.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal, overload

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from fx1.extensions.naming import module_basename, module_directory, module_path
from fx1.harness import HARNESS_REGISTRY, HarnessCommand

CAPABILITY_SCHEMA = "fx1.capabilities/v1"
# Final (not a bare str) so the literal type survives to the Field default below,
# which is annotated Literal["fx1.capability-pack-entry/v1"].
PACK_SCHEMA: Final = "fx1.capability-pack-entry/v1"
SEED_COUNT = 1_000_001
SEED_VERSION = 2
SEED_MAGIC = b"FX1CAP2\0"
SEED_PATH = Path(__file__).with_name("capability_seed_v1.bin.gz")
EXTENSIONS_PATH = Path(__file__).with_name("extensions")
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
FEATURE_WORKFLOWS = (
    "inspect-feature-contract",
    "build-point-in-time-feature",
    "validate-feature-availability",
    "materialize-feature-panel",
    "review-feature-lineage",
)
WORKFLOWS = tuple(
    dict.fromkeys((*sum(WORKFLOWS_BY_ROLE.values(), ()), *PLUGIN_WORKFLOWS, *FEATURE_WORKFLOWS))
)
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
    feature: str | None = None
    market: str | None = None
    asset: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=MAX_PAGE_SIZE)


class ExtensionManifestArguments(BaseModel):
    """Validated request for one separately loadable extension manifest."""

    model_config = ConfigDict(extra="forbid")

    kind: CapabilityKind
    owner: str = Field(min_length=1, max_length=128)


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

EXTENSION_MANIFEST_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_extension_manifest",
        "description": (
            "Load the manifest for one approved, separately loadable fx-1 skill, "
            "datasource plugin, or feature. The manifest is operational metadata, "
            "not market evidence."
        ),
        "parameters": ExtensionManifestArguments.model_json_schema(),
    },
}


@lru_cache(maxsize=1)
def _commands() -> tuple[HarnessCommand, ...]:
    return tuple(command for command in HARNESS_REGISTRY if command.name != "capability-search")


@lru_cache(maxsize=1)
def _source_specs() -> list[Any]:
    from fx1.data.sources.registry import list_sources

    return list_sources()


@lru_cache(maxsize=1)
def _feature_specs() -> list[Any]:
    from fx1.extensions.feature_catalog import list_feature_definitions

    return list(list_feature_definitions())


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
        "features": [
            (
                feature.name,
                feature.lookback,
                feature.point_in_time_safe,
                feature.family,
                feature.synthetic_only,
            )
            for feature in _feature_specs()
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
    features = _feature_specs()
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
    elif KINDS[kind_index] == "skill":
        command_index = sequence % len(commands)
        command = commands[command_index]
        role_workflows = WORKFLOWS_BY_ROLE[command.role.value]
        workflow = role_workflows[(sequence // len(commands)) % len(role_workflows)]
        source_index = _NONE
        market = MARKETS[(sequence // (len(commands) * len(role_workflows))) % len(MARKETS)]
        asset = ASSETS[
            (sequence // (len(commands) * len(role_workflows) * len(MARKETS))) % len(ASSETS)
        ]
    else:
        feature_index = sequence % len(features)
        feature = features[feature_index]
        command_index = next(
            index for index, command in enumerate(commands) if command.name == "build-features"
        )
        market = MARKETS[(sequence // len(features)) % len(MARKETS)]
        asset = ASSETS[(sequence // (len(features) * len(MARKETS))) % len(ASSETS)]
        workflow = FEATURE_WORKFLOWS[
            (sequence // (len(features) * len(MARKETS) * len(ASSETS))) % len(FEATURE_WORKFLOWS)
        ]
        source_index = feature_index

    discipline = (
        "synthetic-correctness-only"
        if KINDS[kind_index] == "feature" and feature.family == "synthetic_oracle"
        else DATA_DISCIPLINES[(sequence // 13) % len(DATA_DISCIPLINES)]
    )
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


def _extension_owners(kind: CapabilityKind) -> tuple[str, ...]:
    if kind == "skill":
        return tuple(command.name for command in _commands())
    if kind == "plugin":
        return tuple(source.name for source in _source_specs())
    if kind == "feature":
        return tuple(feature.name for feature in _feature_specs())
    raise ValueError(f"unknown extension kind {kind!r}")


def _owner_from_row(row: tuple[int, ...]) -> tuple[CapabilityKind, str]:
    kind = KINDS[row[1]]
    if kind == "skill":
        return kind, _commands()[row[2]].name
    if kind == "plugin":
        return kind, _source_specs()[row[3]].name
    return kind, _feature_specs()[row[3]].name


@dataclass(frozen=True)
class SeedReferences(Sequence[int]):
    """A direct, compact sequence of seed ids owned by one extension module."""

    kind: CapabilityKind
    owner: str
    first_id: int
    stride: int
    # Renamed from `count` to avoid shadowing Sequence.count(value); the Liskov
    # violation was harmless but mypy flagged it and the shadow would silently
    # break any caller relying on the inherited count method.
    size: int

    def __len__(self) -> int:
        return self.size

    @overload
    def __getitem__(self, index: int, /) -> int: ...
    @overload
    def __getitem__(self, index: slice, /) -> tuple[int, ...]: ...
    def __getitem__(self, index: int | slice, /) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(self.size)))
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError("capability reference index out of range")
        return self.first_id + index * self.stride


def owner_references(
    kind: CapabilityKind,
    owner: str,
    *,
    count: int = SEED_COUNT,
) -> SeedReferences:
    """Return direct seed references for one approved extension owner."""
    if count < 1 or count > 0xFFFFFFFF:
        raise ValueError("count must be in the uint32 range")
    owners = _extension_owners(kind)
    try:
        owner_index = owners.index(owner)
    except ValueError:
        raise KeyError(f"unknown {kind} extension owner {owner!r}; known: {list(owners)}") from None
    first_id = KINDS.index(kind) + len(KINDS) * owner_index
    stride = len(KINDS) * len(owners)
    available = 0 if first_id >= count else 1 + (count - 1 - first_id) // stride
    return SeedReferences(
        kind=kind,
        owner=owner,
        first_id=first_id,
        stride=stride,
        size=available,
    )


def resolve_seed_id(entry_id: int) -> dict[str, Any]:
    """Resolve one deterministic seed id without scanning the catalog index."""
    if not 0 <= entry_id < SEED_COUNT:
        raise ValueError("capability seed id is outside the seeded catalog")
    return _entry_from_row((entry_id, *_seed_row(entry_id)))


def write_seed_catalog(path: Path = SEED_PATH, *, count: int = SEED_COUNT) -> int:
    """Write the deterministic binary seed as gzip-compressed dimension codes."""
    if count < 1 or count > 0xFFFFFFFF:
        raise ValueError("count must be in the uint32 range")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream,
        ):
            stream.write(_HEADER.pack(SEED_MAGIC, SEED_VERSION, count, _layout_digest()))
            for index in range(count):
                stream.write(_ROW.pack(index, *_seed_row(index)))
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(contents, encoding="utf-8", newline="\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _extension_class_name(kind: CapabilityKind) -> str:
    return {"skill": "SkillExtension", "plugin": "PluginExtension", "feature": "FeatureExtension"}[
        kind
    ]


def write_extension_modules(path: Path = EXTENSIONS_PATH) -> int:
    """Write generated wrappers for existing registered surfaces."""
    written = 0
    for kind in KINDS:
        for owner in _extension_owners(kind):
            class_name = _extension_class_name(kind)
            target = path / module_directory(kind) / f"{module_basename(owner)}.py"
            contents = (
                f'"""Generated {kind} wrapper for {owner!r}; binds to the existing registry."""\n'
                "\n"
                "from fx1.capabilities import owner_references\n"
                f"from fx1.extensions.contracts import {class_name}\n"
                "\n"
                f"MODULE = {class_name}(\n"
                f'    kind="{kind}",\n'
                f'    owner="{owner}",\n'
                f'    references=owner_references("{kind}", "{owner}"),\n'
                "    module=__name__,\n"
                ")\n"
            )
            _atomic_write_text(target, contents)
            written += 1
    return written


def write_declaration_shards(path: Path, *, count: int = SEED_COUNT) -> int:
    """Write executable registration source, split by its real extension owner."""
    if count < 1 or count > 0xFFFFFFFF:
        raise ValueError("count must be in the uint32 range")
    total = 0
    for kind in KINDS:
        for owner in _extension_owners(kind):
            module = module_path(kind, owner)
            target = path / module_directory(kind) / f"{module_basename(owner)}.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp")
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(
                        f'"""Executable generated registrations for {kind}/{owner}.\n\n'
                        f"The generated runtime wrapper is {module}.MODULE. Regenerate; do not edit.\n"
                        '"""\n'
                        "\n"
                        "from array import array\n"
                        f"from {module} import MODULE\n"
                        "\n"
                        'CAPABILITY_REFERENCES = array("I")\n'
                        "\n"
                        "\n"
                        "def _register(seed_id: int) -> None:\n"
                        "    if seed_id < 0:\n"
                        '        raise ValueError("capability seed id must be non-negative")\n'
                        "    CAPABILITY_REFERENCES.append(seed_id)\n"
                        "\n"
                    )
                    references = owner_references(kind, owner, count=count)
                    for entry_id in references:
                        stream.write(f"_register({entry_id})\n")
                    stream.write(
                        "\n"
                        "def extension():\n"
                        '    """Return the separately loadable generated runtime wrapper."""\n'
                        "    return MODULE\n"
                    )
                    total += len(references)
                temporary.replace(target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
    return total


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

    feature_name: str | None = None
    feature_family: str | None = None
    if kind == "plugin":
        source = _source_specs()[source_index]
        command_name = None
        role = "data_engine"
        name = f"plugin/{source.name}/{market}/{asset}/{workflow}/{discipline}/{output}/{unique_suffix}"
        entrypoint = f"{module_path(kind, source.name)}:MODULE"
        description = (
            f"Generated datasource recipe for {source.display} ({source.name}). "
            f"Use {workflow}; this adapter supports {market}/{asset}. Preserve "
            f"{discipline} and produce {output}. The source registry and its "
            "availability probe remain authoritative."
        )
        owner = source.name
    elif kind == "skill":
        command = _commands()[command_index]
        source = None
        command_name = command.name
        role = command.role.value
        name = f"{kind}/{command.name}/{market}/{asset}/{workflow}/{discipline}/{output}/{unique_suffix}"
        entrypoint = f"{module_path(kind, command.name)}:MODULE.run"
        description = (
            f"Generated {kind} recipe backed by registered command {command.name!r}. "
            f"Focus: {workflow} for {market}/{asset}. Respect {discipline}; "
            f"return {output}. Harness command: {command.description}"
        )
        owner = command.name
    else:
        command = _commands()[command_index]
        source = None
        command_name = command.name
        feature = _feature_specs()[source_index]
        feature_name = feature.name
        feature_family = feature.family
        role = command.role.value
        name = f"feature/{feature.name}/{market}/{asset}/{workflow}/{discipline}/{output}/{unique_suffix}"
        entrypoint = f"{module_path(kind, feature.name)}:MODULE.build"
        synthetic_note = (
            " This feature is SYNTHETIC correctness-only and cannot be cited as market evidence."
            if feature.synthetic_only
            else ""
        )
        description = (
            f"Generated feature recipe for the named {feature.family} feature {feature.name!r}. "
            f"Use {workflow} for {market}/{asset}; preserve {discipline} and return {output}. "
            f"The feature runs through registered command {command.name!r}.{synthetic_note}"
        )
        owner = feature.name

    return {
        "id": f"seed:{entry_id:07d}",
        "kind": kind,
        "name": name,
        "description": description,
        "command": command_name,
        "source": source.name if source is not None else None,
        "feature": feature_name,
        "feature_family": feature_family,
        "role": role,
        "market": market,
        "asset": asset,
        "workflow": workflow,
        "data_discipline": discipline,
        "output": output,
        "entrypoint": entrypoint,
        "generated": True,
        "metadata_trust": "repository-generated recipe",
        "origin": "dipcatcher-generated-capability-seed-v2",
        "module": module_path(kind, owner),
        "variant": variant,
    }


def resolve_seed_reference(reference: int) -> dict[str, Any]:
    """Resolve an encoded reference emitted by the generated source program."""
    if reference < 0:
        raise ValueError("capability reference must be non-negative")
    kind_code = reference & 0b11
    entry_id = reference >> 2
    if kind_code >= len(KINDS) or entry_id >= SEED_COUNT:
        raise ValueError("capability reference is outside the seeded catalog")
    entry = resolve_seed_id(entry_id)
    if KINDS.index(entry["kind"]) != kind_code:
        raise ValueError("capability reference kind does not match its seed entry")
    return entry


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
                    "feature": None,
                    "feature_family": None,
                    "role": command.role.value if command is not None else "data_engine",
                    "entrypoint": entrypoint,
                    "tags": entry.tags,
                    "generated": False,
                    "metadata_trust": "operator-provided; review before use",
                    "origin": str(path),
                    "module": None,
                }


def _search_text(entry: dict[str, Any]) -> str:
    values = (
        entry.get("id"),
        entry.get("kind"),
        entry.get("name"),
        entry.get("description"),
        entry.get("command"),
        entry.get("source"),
        entry.get("feature"),
        entry.get("feature_family"),
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
    source = _source_specs()[source_index] if row_kind == "plugin" else None
    feature = _feature_specs()[source_index] if row_kind == "feature" else None
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
    if filters["feature"] is not None and (
        feature is None or feature.name.casefold() != filters["feature"]
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
    if feature is not None:
        searchable_parts.extend(
            (
                "generated feature recipe point in time harness build features",
                feature.name,
                feature.family,
                "synthetic correctness only" if feature.synthetic_only else "",
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
    feature: str | None = None,
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
        "feature": feature.casefold() if feature else None,
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
