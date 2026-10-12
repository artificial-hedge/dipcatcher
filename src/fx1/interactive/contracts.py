"""Capability-contract client for ``/superpower``.

Loads the lab JSON schema ``dipcatcher.capabilities.v1`` and turns it into
typed, argv-safe calls. Unknown optional fields outside an argument object
are ignored. Unknown keys on an argument object are rejected. Missing required
fields, wrong types, and an ``executable`` capability that is not
``qualified`` (or that has no qualification evidence) reject the catalog.
Retired capabilities are never executable.

``bind`` is an optional planner hint, ``dataset`` or ``goal``. ``goal_terms``
maps a goal-bound choice to lowercase phrases and must be a subset of
``choices``. The size bounds match the lab publisher.

``allowed_args``, when present, is a whitelist. A list of names only
whitelists. An object whitelists and may tighten ``min``, ``max``, and
``choices``; a tightening cannot widen the spec. JSON key ``min``/``max``
map to :attr:`ArgSpec.minimum` / :attr:`ArgSpec.maximum`. ``has_default``
is true only when the catalog declares ``default`` (JSON null is rejected).

A non-executable entry may publish JSON null for ``command.timeout_s`` and
for ``limits.max_runtime_s`` / ``max_output_chars``; those nulls are read as
0. An executable entry must publish real numbers. Capability ids may be the
lab's space-separated nested command names. Positional argument names may
contain ``_``. Neither ids nor argument names are interpolated into a shell;
calls stay argv arrays, and shell metacharacters are still rejected.

The catalog digest is the SHA-256 of the canonical JSON (sorted keys,
compact separators) of the parsed document, including ignored fields.
Argument strings reject shell metacharacters ``;|&`$()<>\\`` plus NUL and
newlines. Path values may contain ``\\`` (Windows separators) but not ``..``
segments, and must resolve inside the named roots for their ``path_role``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import fx1
from fx1.interactive.atomicstore import StoreCorruptError, atomic_write_json, read_json
from fx1.interactive.redact import redact_text

SCHEMA_ID = "dipcatcher.capabilities.v1"
SCHEMA_MAJOR = "dipcatcher.capabilities."
VIOLATION_CODES = frozenset(
    {
        "unknown_arg",
        "missing_required",
        "bad_type",
        "out_of_range",
        "bad_choice",
        "path_escape",
        "too_long",
        "not_executable",
    }
)
_STATUSES = frozenset({"qualified", "experimental", "unqualified", "retired"})
_EVIDENCE = frozenset({"SYNTHETIC", "RESEARCH", "VERIFICATION"})
_KINDS = frozenset({"option", "argument"})
_TYPES = frozenset({"flag", "int", "float", "str", "enum", "path", "csv"})
_ITEM_TYPES = frozenset({"int", "float", "str", "enum"})
_PATH_ROLES = frozenset({"config", "input", "output_dir", "any"})
# Bounds are mirrored by quant_fund.cli.capability_cmds (publisher/client lockstep).
_BIND_KINDS = frozenset({"dataset", "goal"})
_ARG_KEYS = frozenset(
    {
        "name",
        "kind",
        "type",
        "required",
        "default",
        "repeatable",
        "help",
        "choices",
        "min",
        "max",
        "item_type",
        "path_role",
        "bind",
        "goal_terms",
    }
)
_MAX_BIND_CHOICES = 8
_MAX_BIND_TERMS = 12
_MAX_BIND_TERM_LEN = 48
_MAX_BIND_TERM_TOTAL = 48
_TERM_RE = re.compile(r"^[a-z0-9]+(?:[ -][a-z0-9]+)*$")
_EMPTY_TERMS: Mapping[str, tuple[str, ...]] = MappingProxyType({})
_NUMERIC_TYPES = frozenset({"int", "float", "csv"})
_CHOICE_TYPES = frozenset({"str", "enum", "csv", "int"})
_SHELL_CHARS = frozenset(";|&`$()<>\\\n\r\x00")
_PATH_META = frozenset(";|&`$()<>\n\r\x00")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
# Lab command ids: one token, or nested tokens separated by single spaces
# (``allocation compare``). Shell metacharacters are not part of this pattern.
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*(?: [a-z0-9][a-z0-9-]*)*$")
_TASK_RE = re.compile(r"^[a-z][a-z0-9_]{0,80}$")
_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9 ._-]{0,80}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
_OPTION_NAME = re.compile(r"^--[a-z][a-z0-9-]{0,63}$")
_ARG_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_INT_TOKEN = re.compile(r"-?\d+")
_FLOAT_TOKEN = re.compile(r"-?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?")
_MAX_STR = 256
_MAX_PATH = 4096
_MAX_CSV = 32
_CACHE_NAME = "capabilities.json"

Runner = Callable[[str, list[str]], tuple[int, str]]


class ContractError(ValueError):
    """The catalog document itself is unusable."""


class CatalogUnavailable(ContractError):
    """The capabilities command did not yield a schema-valid catalog."""


class ContractViolation(ValueError):
    """A caller argument does not satisfy the capability contract."""

    def __init__(self, code: str, arg: str, detail: str) -> None:
        if code not in VIOLATION_CODES:
            raise ValueError(f"unknown violation code {code}")
        self.code = code
        self.arg = arg
        message = f"{arg}: {detail}" if arg else detail
        super().__init__(message)


@dataclass(frozen=True)
class ArgSpec:
    """One CLI option or positional argument.

    ``bind`` is ``dataset`` or ``goal`` when the planner may fill the value.
    ``goal_terms`` maps a goal-bound choice to the phrases that select it.
    """

    name: str
    kind: str
    type: str
    required: bool
    default: Any
    choices: tuple[str, ...] | None
    minimum: float | None
    maximum: float | None
    item_type: str | None
    path_role: str | None
    repeatable: bool
    help: str
    has_default: bool
    bind: str | None = None
    goal_terms: Mapping[str, tuple[str, ...]] = _EMPTY_TERMS


@dataclass(frozen=True)
class ArgTightening:
    """Optional per-argument narrowing from ``allowed_args``."""

    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Capability:
    """One lab capability flattened from the nested JSON contract."""

    id: str
    version: str
    purpose: str
    tasks: tuple[str, ...]
    aliases: tuple[str, ...]
    argv: tuple[str, ...]
    role: str
    read_only: bool
    writes: tuple[str, ...]
    consequential: bool
    timeout_s: float
    args: tuple[ArgSpec, ...]
    data_labels: tuple[str, ...]
    requires_dataset: bool
    availability_timing: str
    scores: tuple[str, ...]
    receipt_schema: str
    receipt_flag: tuple[str, ...]
    receipt_stdout_marker: str
    dependencies: tuple[str, ...]
    post_checks: tuple[Mapping[str, Any], ...]
    max_runtime_s: float
    max_output_chars: int
    qualification_status: str
    qualification_evidence: tuple[str, ...]
    evidence_class: str
    executable: bool
    executable_reason: str
    allowed_args: Mapping[str, ArgTightening] | None = None


@dataclass(frozen=True)
class CapabilityCatalog:
    """Parsed catalog plus the digest of the JSON that produced it."""

    capabilities: Mapping[str, Capability]
    digest: str
    document: Mapping[str, Any]

    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | str) -> CapabilityCatalog:
        """Parse a catalog document. ``doc`` may be a mapping or JSON text."""
        parsed = _load_document(doc)
        _check_schema(parsed)
        _check_generated_by(parsed)
        caps = _capabilities_from_document(parsed)
        digest = catalog_digest(parsed)
        return cls(capabilities=MappingProxyType(caps), digest=digest, document=parsed)

    def executable(self) -> list[Capability]:
        """Executable capabilities, sorted by id."""
        return [
            self.capabilities[key]
            for key in sorted(self.capabilities)
            if self.capabilities[key].executable
        ]

    def get(self, cid: str) -> Capability | None:
        """Return one capability, or ``None`` when the id is unknown."""
        return self.capabilities.get(cid)


def catalog_digest(document: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON form of ``document``."""
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def contains_shell_metacharacters(text: str) -> bool:
    """True when ``text`` contains a character that must never reach a shell."""
    return any(char in _SHELL_CHARS for char in text)


def fetch_catalog(runner: Runner) -> CapabilityCatalog:
    """Run ``capabilities list --json`` and parse stdout.

    Non-zero exits, invalid JSON, and schema failures raise
    :class:`CatalogUnavailable`. The message is redacted and truncated.
    """
    code, stdout = _invoke(runner)
    if code != 0:
        raise CatalogUnavailable(_short_reason(f"capabilities list exited {code}"))
    try:
        return CapabilityCatalog.from_json(stdout)
    except ContractError as exc:
        raise CatalogUnavailable(_short_reason(str(exc))) from exc


class CatalogCache:
    """TTL cache of a schema-valid catalog under ``$FX1_CONFIG_DIR``.

    The cache key is ``(fx1.__version__, catalog digest)``. A hit is parsed
    again with :meth:`CapabilityCatalog.from_json`. Stale, corrupt, or
    mismatched entries are refetched. A successful fetch is what gets
    returned, even when its digest differs from the file that was there.
    """

    def __init__(
        self,
        runner: Runner,
        *,
        config_dir: Path | None = None,
        ttl_s: float = 300.0,
        clock: Callable[[], float] | None = None,
        version: str | None = None,
    ) -> None:
        if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)) or ttl_s < 0:
            raise ValueError("ttl_s must be a nonnegative number")
        self._runner = runner
        self._config_dir = config_dir
        self._ttl_s = float(ttl_s)
        self._clock = clock or _wall_clock
        self._version = fx1.__version__ if version is None else version

    def path(self) -> Path:
        """Filesystem path of the cache file."""
        return _resolve_config_dir(self._config_dir) / "cache" / _CACHE_NAME

    def get(self) -> CapabilityCatalog:
        """Return a fresh catalog, refetching when the cache cannot be used."""
        cached = self._load_fresh()
        if cached is not None:
            return cached
        catalog = fetch_catalog(self._runner)
        self._store(catalog)
        return catalog

    def _load_fresh(self) -> CapabilityCatalog | None:
        payload = _read_payload(self.path())
        if payload is None:
            return None
        return _fresh_catalog(
            payload,
            version=self._version,
            now=float(self._clock()),
            ttl_s=self._ttl_s,
        )

    def _store(self, catalog: CapabilityCatalog) -> None:
        payload = {
            "digest": catalog.digest,
            "document": catalog.document,
            "fetched_at": float(self._clock()),
            "fx1_version": self._version,
        }
        try:
            atomic_write_json(self.path(), payload)
        except OSError:
            return


def validate_args(
    cap: Capability,
    args: Mapping[str, Any],
    *,
    roots: Mapping[str, Sequence[Path]] | None = None,
) -> dict[str, Any]:
    """Validate ``args`` and apply declared defaults.

    Unknown names are rejected. ``executable`` false raises ``not_executable``.
    Returned keys are spec names in spec order.
    """
    if not cap.executable:
        detail = cap.executable_reason or "capability is not executable"
        raise ContractViolation("not_executable", "", detail)
    if not isinstance(args, Mapping):
        raise ContractViolation("bad_type", "", "args must be an object")
    provided = _canonicalize_keys(cap, args)
    return _collect_values(cap, provided, {} if roots is None else roots)


def parse_cli_tokens(cap: Capability, tokens: Sequence[str]) -> dict[str, Any]:
    """Parse an argv tail into the same dict :func:`validate_args` returns."""
    raw = _parse_tokens(cap, tokens)
    return validate_args(cap, raw)


def build_argv(
    cap: Capability,
    validated: Mapping[str, Any],
    *,
    roots: Mapping[str, Sequence[Path]] | None = None,
) -> list[str]:
    """Build a deterministic argv, including the receipt flag when required.

    Values are re-validated. Pass ``roots`` when a value is a path so the
    same containment check that accepted the plan accepts the argv. Spec
    order is preserved. False flags are omitted. Shell metacharacters never
    appear in the result.
    """
    if not isinstance(validated, Mapping):
        raise ContractViolation("bad_type", "", "validated args must be an object")
    confirmed = validate_args(cap, validated, roots=roots)
    argv = list(cap.argv)
    for spec in cap.args:
        if spec.name not in confirmed:
            continue
        argv.extend(_emit(spec, confirmed[spec.name]))
    argv.extend(_receipt_suffix(cap, argv))
    _ensure_clean(argv)
    return argv


def _wall_clock() -> float:
    return time.time()


def _load_document(doc: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(doc, str):
        parsed = _parse_json_text(doc)
    elif isinstance(doc, Mapping):
        parsed = _parse_json_mapping(doc)
    else:
        raise ContractError("catalog must be a JSON object or text")
    if not isinstance(parsed, dict):
        raise ContractError("catalog must be a JSON object")
    return parsed


def _parse_json_text(doc: str) -> Any:
    try:
        return json.loads(doc)
    except json.JSONDecodeError as exc:
        raise ContractError("catalog is not valid JSON") from exc


def _parse_json_mapping(doc: Mapping[str, Any]) -> Any:
    try:
        return json.loads(json.dumps(doc, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ContractError("catalog is not JSON-compatible") from exc


def _check_schema(doc: Mapping[str, Any]) -> None:
    schema = doc.get("schema")
    if schema == SCHEMA_ID:
        return
    if isinstance(schema, str) and schema.startswith(SCHEMA_MAJOR):
        raise ContractError(f"unsupported schema major {schema!r}")
    raise ContractError(f"unsupported schema {schema!r}; expected {SCHEMA_ID}")


def _check_generated_by(doc: Mapping[str, Any]) -> None:
    raw = doc.get("generated_by")
    if not isinstance(raw, Mapping):
        raise ContractError("generated_by must be an object")
    digest = raw.get("overlay_sha256")
    count = raw.get("n_lab_commands")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ContractError("generated_by.overlay_sha256 must be 64 hex characters")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ContractError("generated_by.n_lab_commands must be a nonnegative integer")


def _capabilities_from_document(doc: Mapping[str, Any]) -> dict[str, Capability]:
    if doc.get("view") == "describe":
        return _parse_describe(doc)
    return _parse_capabilities(doc)


def _parse_describe(doc: Mapping[str, Any]) -> dict[str, Capability]:
    error = doc.get("error")
    if error is not None:
        raise ContractError(_describe_error_text(error))
    capability = doc.get("capability")
    if not isinstance(capability, dict):
        raise ContractError("describe capability must be an object")
    cap = _parse_capability(capability, 0)
    return {cap.id: cap}


def _describe_error_text(error: object) -> str:
    if isinstance(error, str) and error.strip():
        return error.strip()
    return "describe failed"


def _parse_capabilities(doc: Mapping[str, Any]) -> dict[str, Capability]:
    raw = doc.get("capabilities")
    if not isinstance(raw, list):
        raise ContractError("capabilities must be a list")
    found: dict[str, Capability] = {}
    for index, item in enumerate(raw):
        cap = _parse_capability(item, index)
        if cap.id in found:
            raise ContractError(f"duplicate capability id {cap.id}")
        found[cap.id] = cap
    _check_dependencies(found)
    return found


def _parse_capability(item: Any, index: int) -> Capability:
    obj = _mapping(item, f"capabilities[{index}]")
    ctx = f"capabilities[{index}]"
    identity = _parse_identity(obj, ctx)
    executable = _bool(obj.get("executable"), f"{ctx}.executable")
    command = _parse_command(obj, ctx, allow_null_budget=not executable)
    inputs = _parse_inputs(obj, ctx)
    outputs = _parse_outputs(obj, ctx)
    limits = _parse_limits(obj, ctx, allow_null_budget=not executable)
    status, evidence = _parse_qualification(obj, ctx)
    args = _parse_args(obj.get("args"), ctx)
    allowed = _parse_allowed(obj, args, ctx)
    _check_allowed_covers_required(args, allowed, ctx)
    _check_defaults(args, allowed, ctx)
    cap = Capability(
        id=identity[0],
        version=identity[1],
        purpose=identity[2],
        tasks=identity[3],
        aliases=identity[4],
        argv=command[0],
        role=command[1],
        read_only=command[2],
        writes=command[3],
        consequential=command[4],
        timeout_s=command[5],
        args=args,
        data_labels=inputs[0],
        requires_dataset=inputs[1],
        availability_timing=inputs[2],
        scores=outputs[0],
        receipt_schema=outputs[1],
        receipt_flag=outputs[2],
        receipt_stdout_marker=outputs[3],
        dependencies=_id_tuple(obj.get("dependencies"), f"{ctx}.dependencies"),
        post_checks=_parse_post_checks(obj.get("post_checks"), ctx),
        max_runtime_s=limits[0],
        max_output_chars=limits[1],
        qualification_status=status,
        qualification_evidence=evidence,
        evidence_class=_member(obj.get("evidence_class"), _EVIDENCE, f"{ctx}.evidence_class"),
        executable=executable,
        executable_reason=_text(
            obj.get("executable_reason"), f"{ctx}.executable_reason", limit=400
        ),
        allowed_args=None if allowed is None else MappingProxyType(allowed),
    )
    _check_executable_policy(cap)
    return cap


def _parse_identity(
    obj: Mapping[str, Any], ctx: str
) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
    cap_id = _text(obj.get("id"), f"{ctx}.id", limit=64)
    if _ID_RE.fullmatch(cap_id) is None:
        raise ContractError(f"{ctx}.id is invalid")
    version = _text(obj.get("version"), f"{ctx}.version", limit=32)
    if _VERSION_RE.fullmatch(version) is None:
        raise ContractError(f"{ctx}.version is invalid")
    purpose = _text(obj.get("purpose"), f"{ctx}.purpose", limit=500).strip()
    if not purpose:
        raise ContractError(f"{ctx}.purpose must be non-empty")
    tasks = _pattern_tuple(obj.get("tasks"), f"{ctx}.tasks", _TASK_RE)
    aliases = _pattern_tuple(obj.get("aliases"), f"{ctx}.aliases", _ALIAS_RE)
    return cap_id, version, purpose, tasks, aliases


def _parse_command(
    obj: Mapping[str, Any],
    ctx: str,
    *,
    allow_null_budget: bool,
) -> tuple[tuple[str, ...], str, bool, tuple[str, ...], bool, float]:
    raw = _mapping(obj.get("command"), f"{ctx}.command")
    argv = _str_tuple(raw.get("argv"), f"{ctx}.command.argv", allow_empty=False)
    _reject_shell_tokens(argv, f"{ctx}.command.argv")
    role = _text(raw.get("role"), f"{ctx}.command.role", limit=64).strip()
    if not role:
        raise ContractError(f"{ctx}.command.role must be non-empty")
    timeout = _optional_budget(
        raw.get("timeout_s"), f"{ctx}.command.timeout_s", allow_null=allow_null_budget
    )
    return (
        argv,
        role,
        _bool(raw.get("read_only"), f"{ctx}.command.read_only"),
        _str_tuple(raw.get("writes"), f"{ctx}.command.writes", allow_empty=True),
        _bool(raw.get("consequential"), f"{ctx}.command.consequential"),
        timeout,
    )


def _parse_inputs(obj: Mapping[str, Any], ctx: str) -> tuple[tuple[str, ...], bool, str]:
    raw = _mapping(obj.get("inputs"), f"{ctx}.inputs")
    timing = _text(raw.get("availability_timing"), f"{ctx}.inputs.availability_timing", limit=400)
    if not timing.strip():
        raise ContractError(f"{ctx}.inputs.availability_timing must be non-empty")
    return (
        _str_tuple(raw.get("data_labels"), f"{ctx}.inputs.data_labels", allow_empty=True),
        _bool(raw.get("requires_dataset"), f"{ctx}.inputs.requires_dataset"),
        timing.strip(),
    )


def _parse_outputs(
    obj: Mapping[str, Any], ctx: str
) -> tuple[tuple[str, ...], str, tuple[str, ...], str]:
    raw = _mapping(obj.get("outputs"), f"{ctx}.outputs")
    flag = _str_tuple(raw.get("receipt_flag"), f"{ctx}.outputs.receipt_flag", allow_empty=True)
    _reject_shell_tokens(flag, f"{ctx}.outputs.receipt_flag")
    marker = _text(
        raw.get("receipt_stdout_marker"), f"{ctx}.outputs.receipt_stdout_marker", limit=80
    )
    if _marker_rejected(marker):
        raise ContractError(f"{ctx}.outputs.receipt_stdout_marker contains unsupported characters")
    schema = _text(raw.get("receipt_schema"), f"{ctx}.outputs.receipt_schema", limit=80)
    return (
        _str_tuple(raw.get("scores"), f"{ctx}.outputs.scores", allow_empty=True),
        schema,
        flag,
        marker,
    )


def _parse_limits(
    obj: Mapping[str, Any], ctx: str, *, allow_null_budget: bool
) -> tuple[float, int]:
    raw = _mapping(obj.get("limits"), f"{ctx}.limits")
    runtime = _optional_budget(
        raw.get("max_runtime_s"),
        f"{ctx}.limits.max_runtime_s",
        allow_null=allow_null_budget,
    )
    output_chars = _optional_budget_int(
        raw.get("max_output_chars"),
        f"{ctx}.limits.max_output_chars",
        allow_null=allow_null_budget,
    )
    return runtime, output_chars


def _parse_qualification(obj: Mapping[str, Any], ctx: str) -> tuple[str, tuple[str, ...]]:
    raw = _mapping(obj.get("qualification"), f"{ctx}.qualification")
    status = _member(raw.get("status"), _STATUSES, f"{ctx}.qualification.status")
    evidence = _str_tuple(raw.get("evidence"), f"{ctx}.qualification.evidence", allow_empty=True)
    return status, evidence


def _parse_args(raw: Any, ctx: str) -> tuple[ArgSpec, ...]:
    if not isinstance(raw, list):
        raise ContractError(f"{ctx}.args must be a list")
    args = tuple(_parse_arg(item, f"{ctx}.args[{index}]") for index, item in enumerate(raw))
    _check_unique_names(args)
    return args


def _parse_arg(raw: Any, ctx: str) -> ArgSpec:
    obj = _mapping(raw, ctx)
    _reject_unknown_keys(obj, _ARG_KEYS, ctx)
    name = _text(obj.get("name"), f"{ctx}.name", limit=68)
    kind = _member(obj.get("kind"), _KINDS, f"{ctx}.kind")
    typ = _member(obj.get("type"), _TYPES, f"{ctx}.type")
    _check_name_kind(name, kind, ctx)
    # The lab publishes ``"default": null`` for an option with no default, and
    # every other default as text; an absent key and a null both mean "no
    # default", and text is coerced to the declared type once, here.
    default = _coerce_default(typ, obj.get("default"), ctx)
    has_default = default is not None
    spec = ArgSpec(
        name=name,
        kind=kind,
        type=typ,
        required=_bool(obj.get("required"), f"{ctx}.required"),
        default=default,
        choices=_optional_choices(obj, ctx),
        minimum=_optional_number(obj, "min", ctx),
        maximum=_optional_number(obj, "max", ctx),
        item_type=_optional_member(obj, "item_type", _ITEM_TYPES, ctx),
        path_role=_optional_member(obj, "path_role", _PATH_ROLES, ctx),
        repeatable=_optional_bool(obj, "repeatable", False, ctx),
        help=_help_text(obj, ctx),
        has_default=has_default,
        bind=_optional_member(obj, "bind", _BIND_KINDS, ctx),
        goal_terms=_parse_goal_terms(obj, ctx),
    )
    _check_arg_shape(spec, ctx)
    return spec


def _coerce_default(kind: str, raw: Any, ctx: str) -> Any:
    """Typed form of a published default; non-text values pass through."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if kind == "int":
        if re.fullmatch(r"[+-]?\d+", text) is None:
            raise ContractError(f"{ctx}.default is not an integer")
        return int(text)
    if kind == "float":
        try:
            return float(text)
        except ValueError:
            raise ContractError(f"{ctx}.default is not a number") from None
    if kind == "flag":
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        raise ContractError(f"{ctx}.default is not a boolean")
    # Typer publishes ``""`` for an optional path with no default. That is not
    # a usable path, and it must not become a default the planner can emit.
    if kind == "path" and text == "":
        return None
    return raw


def _check_arg_shape(spec: ArgSpec, ctx: str) -> None:
    _check_bounds_pair(spec.minimum, spec.maximum, ctx)
    _check_type_rules(spec, ctx)
    _check_binding(spec, ctx)
    if spec.has_default:
        _reject_bad_default(spec, None, ctx)


def _reject_unknown_keys(obj: Mapping[str, Any], allowed: frozenset[str], ctx: str) -> None:
    extra = sorted(key for key in obj if key not in allowed)
    if extra:
        raise ContractError(f"{ctx} unknown key {extra[0]}")


def _parse_goal_terms(obj: Mapping[str, Any], ctx: str) -> Mapping[str, tuple[str, ...]]:
    if "goal_terms" not in obj or obj["goal_terms"] is None:
        return _EMPTY_TERMS
    raw = obj["goal_terms"]
    _check_goal_terms_shape(raw, ctx)
    if not isinstance(raw, Mapping):
        return _EMPTY_TERMS
    return _goal_terms_mapping(raw, ctx)


def _check_goal_terms_shape(raw: Any, ctx: str) -> None:
    if not isinstance(raw, Mapping) or not raw:
        raise ContractError(f"{ctx}.goal_terms must be an object")
    if len(raw) > _MAX_BIND_CHOICES:
        raise ContractError(f"{ctx}.goal_terms exceeds {_MAX_BIND_CHOICES} choices")


def _goal_terms_mapping(raw: Mapping[str, Any], ctx: str) -> Mapping[str, tuple[str, ...]]:
    built: dict[str, tuple[str, ...]] = {}
    total = 0
    for key, value in raw.items():
        terms = _parse_term_list(key, value, ctx)
        total += len(terms)
        if total > _MAX_BIND_TERM_TOTAL:
            raise ContractError(f"{ctx}.goal_terms exceeds {_MAX_BIND_TERM_TOTAL} terms")
        built[str(key)] = terms
    return MappingProxyType(built)


def _parse_term_list(key: Any, value: Any, ctx: str) -> tuple[str, ...]:
    if not isinstance(key, str) or not key:
        raise ContractError(f"{ctx}.goal_terms has an invalid choice")
    where = f"{ctx}.goal_terms.{key}"
    items = _term_items(value, where)
    terms = tuple(_one_term(item, where) for item in items)
    if len(set(terms)) != len(terms):
        raise ContractError(f"{where} has a duplicate term")
    return terms


def _term_items(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{where} must be a non-empty list")
    if len(value) > _MAX_BIND_TERMS:
        raise ContractError(f"{where} exceeds {_MAX_BIND_TERMS} terms")
    return list(value)


def _one_term(item: Any, where: str) -> str:
    if isinstance(item, str) and len(item) <= _MAX_BIND_TERM_LEN and _TERM_RE.fullmatch(item):
        return item
    raise ContractError(f"{where} has an invalid term")


def _check_binding(spec: ArgSpec, ctx: str) -> None:
    if spec.bind is None:
        _check_unbound_terms(spec, ctx)
        return
    if spec.bind == "dataset":
        _check_dataset_bind(spec, ctx)
        return
    _check_goal_bind(spec, ctx)


def _check_unbound_terms(spec: ArgSpec, ctx: str) -> None:
    if spec.goal_terms:
        raise ContractError(f"{ctx} goal_terms requires bind goal")


def _check_dataset_bind(spec: ArgSpec, ctx: str) -> None:
    if spec.type != "path" or spec.path_role != "input":
        raise ContractError(f"{ctx} bind dataset requires an input path")
    if spec.goal_terms:
        raise ContractError(f"{ctx} goal_terms is only valid for bind goal")


def _check_goal_bind(spec: ArgSpec, ctx: str) -> None:
    if spec.type not in {"str", "enum"} or not spec.choices:
        raise ContractError(f"{ctx} bind goal requires choices")
    if not spec.goal_terms:
        raise ContractError(f"{ctx} bind goal requires goal_terms")
    extra = [key for key in spec.goal_terms if key not in spec.choices]
    if extra:
        raise ContractError(f"{ctx} goal_terms keys must be a subset of choices")


def _check_type_rules(spec: ArgSpec, ctx: str) -> None:
    if spec.type == "enum" and not spec.choices:
        raise ContractError(f"{ctx} enum requires choices")
    if spec.type == "path" and spec.path_role not in _PATH_ROLES:
        raise ContractError(f"{ctx} path requires path_role")
    if spec.type == "csv" and spec.item_type not in _ITEM_TYPES:
        raise ContractError(f"{ctx} csv requires item_type")
    if spec.type == "csv" and spec.item_type == "enum" and not spec.choices:
        raise ContractError(f"{ctx} csv enum requires choices")
    if spec.type == "flag" and spec.repeatable:
        raise ContractError(f"{ctx} flag args cannot be repeatable")
    if spec.type == "csv" and spec.repeatable:
        raise ContractError(f"{ctx} csv args cannot be repeatable")
    _check_placement(spec, ctx)


def _check_placement(spec: ArgSpec, ctx: str) -> None:
    if spec.choices is not None and spec.type not in _CHOICE_TYPES:
        raise ContractError(f"{ctx} choices are not valid for {spec.type}")
    if spec.path_role is not None and spec.type != "path":
        raise ContractError(f"{ctx} path_role is only valid for path")
    if spec.item_type is not None and spec.type != "csv":
        raise ContractError(f"{ctx} item_type is only valid for csv")
    if spec.minimum is None and spec.maximum is None:
        return
    if spec.type not in _NUMERIC_TYPES:
        raise ContractError(f"{ctx} min/max are only valid for numeric args")
    if spec.type == "csv" and spec.item_type not in {"int", "float"}:
        raise ContractError(f"{ctx} min/max require numeric csv items")


def _parse_allowed(
    obj: Mapping[str, Any], args: tuple[ArgSpec, ...], ctx: str
) -> dict[str, ArgTightening] | None:
    if "allowed_args" not in obj:
        return None
    raw = obj["allowed_args"]
    names = {spec.name: spec for spec in args}
    if isinstance(raw, list):
        return _allowed_list(raw, names, ctx)
    if isinstance(raw, Mapping):
        return _allowed_map(raw, names, ctx)
    raise ContractError(f"{ctx}.allowed_args must be a list or object")


def _allowed_list(
    raw: list[Any], names: Mapping[str, ArgSpec], ctx: str
) -> dict[str, ArgTightening]:
    if not all(isinstance(item, str) for item in raw):
        raise ContractError(f"{ctx}.allowed_args must list argument names")
    unknown = [item for item in raw if item not in names]
    if unknown:
        raise ContractError(f"{ctx}.allowed_args references unknown arg {unknown[0]}")
    return {item: ArgTightening() for item in raw}


def _allowed_map(
    raw: Mapping[str, Any], names: Mapping[str, ArgSpec], ctx: str
) -> dict[str, ArgTightening]:
    found: dict[str, ArgTightening] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or name not in names:
            raise ContractError(f"{ctx}.allowed_args references unknown arg {name}")
        found[name] = _tightening(value, names[name], f"{ctx}.allowed_args.{name}")
    return found


def _tightening(value: Any, spec: ArgSpec, ctx: str) -> ArgTightening:
    if value is None:
        return ArgTightening()
    obj = _mapping(value, ctx)
    minimum = _optional_number(obj, "min", ctx)
    maximum = _optional_number(obj, "max", ctx)
    choices = _optional_choices(obj, ctx)
    _check_bounds_pair(minimum, maximum, ctx)
    _check_choice_tightening(spec, choices, ctx)
    _check_effective_bounds(spec, minimum, maximum, ctx)
    return ArgTightening(minimum=minimum, maximum=maximum, choices=choices)


def _check_choice_tightening(spec: ArgSpec, choices: tuple[str, ...] | None, ctx: str) -> None:
    if choices is None:
        return
    if spec.type not in _CHOICE_TYPES:
        raise ContractError(f"{ctx} cannot tighten choices for {spec.type}")
    if spec.choices is None:
        return
    extra = [item for item in choices if item not in spec.choices]
    if extra:
        raise ContractError(f"{ctx} widens choices")
    if not any(item in spec.choices for item in choices):
        raise ContractError(f"{ctx} excludes every choice")


def _check_effective_bounds(
    spec: ArgSpec, minimum: float | None, maximum: float | None, ctx: str
) -> None:
    lo = _merge_min(spec.minimum, minimum)
    hi = _merge_max(spec.maximum, maximum)
    if lo is not None and hi is not None and lo > hi:
        raise ContractError(f"{ctx} tightening leaves an empty range")


def _check_allowed_covers_required(
    args: tuple[ArgSpec, ...],
    allowed: Mapping[str, ArgTightening] | None,
    ctx: str,
) -> None:
    if allowed is None:
        return
    for spec in args:
        if spec.required and spec.name not in allowed:
            raise ContractError(f"{ctx} required arg {spec.name} is outside allowed_args")


def _check_defaults(
    args: tuple[ArgSpec, ...],
    allowed: Mapping[str, ArgTightening] | None,
    ctx: str,
) -> None:
    for spec in args:
        if not spec.has_default:
            continue
        if allowed is not None and spec.name not in allowed:
            continue
        tight = None if allowed is None else allowed.get(spec.name)
        _reject_bad_default(spec, tight, ctx)


def _reject_bad_default(spec: ArgSpec, tight: ArgTightening | None, ctx: str) -> None:
    try:
        _validate_value(spec, spec.default, tight, {}, for_default=True)
    except ContractViolation as exc:
        raise ContractError(f"{ctx} invalid default for {spec.name}: {exc.code}") from exc


def _check_executable_policy(cap: Capability) -> None:
    if cap.executable:
        _check_runnable(cap)
        return
    if not cap.executable_reason.strip():
        raise ContractError(f"{cap.id} is not executable and has no executable_reason")


def _check_runnable(cap: Capability) -> None:
    if cap.qualification_status != "qualified":
        raise ContractError(
            f"{cap.id} is executable but qualification status is {cap.qualification_status}"
        )
    if not cap.qualification_evidence:
        raise ContractError(f"{cap.id} is executable without qualification evidence")


def _check_dependencies(caps: Mapping[str, Capability]) -> None:
    for cap in caps.values():
        for dep in cap.dependencies:
            if dep == cap.id:
                raise ContractError(f"{cap.id} depends on itself")
            if dep not in caps:
                raise ContractError(f"{cap.id} depends on unknown id {dep}")


def _parse_post_checks(raw: Any, ctx: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(raw, list):
        raise ContractError(f"{ctx}.post_checks must be a list")
    return tuple(_one_check(item, f"{ctx}.post_checks[{index}]") for index, item in enumerate(raw))


def _one_check(raw: Any, ctx: str) -> Mapping[str, Any]:
    obj = _mapping(raw, ctx)
    kind = obj.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise ContractError(f"{ctx}.kind must be a non-empty string")
    cleaned: dict[str, Any] = {}
    for key, value in obj.items():
        if not isinstance(key, str):
            raise ContractError(f"{ctx} has a non-string key")
        if not _json_scalar(value):
            raise ContractError(f"{ctx}.{key} must be a scalar")
        cleaned[key] = value
    return MappingProxyType(cleaned)


def _check_unique_names(args: Sequence[ArgSpec]) -> None:
    seen: set[str] = set()
    for spec in args:
        keys = [spec.name]
        stripped = spec.name.lstrip("-")
        if stripped != spec.name:
            keys.append(stripped)
        for key in keys:
            if key in seen:
                raise ContractError(f"ambiguous argument name {key}")
            seen.add(key)


def _check_name_kind(name: str, kind: str, ctx: str) -> None:
    pattern = _OPTION_NAME if kind == "option" else _ARG_NAME
    if pattern.fullmatch(name) is None:
        raise ContractError(f"{ctx}.name is invalid")


def _canonicalize_keys(cap: Capability, args: Mapping[str, Any]) -> dict[str, Any]:
    index = _name_index(cap)
    allowed = None if cap.allowed_args is None else set(cap.allowed_args)
    found: dict[str, Any] = {}
    for key, value in args.items():
        canonical = _canonical_name(key, index)
        if allowed is not None and canonical not in allowed:
            raise ContractViolation("unknown_arg", canonical, "argument is not allowed")
        if canonical in found:
            raise ContractViolation("unknown_arg", canonical, "duplicate argument")
        found[canonical] = value
    return found


def _canonical_name(key: Any, index: Mapping[str, str]) -> str:
    if not isinstance(key, str):
        raise ContractViolation("unknown_arg", "", "argument name must be a string")
    canonical = index.get(key)
    if canonical is None:
        raise ContractViolation("unknown_arg", key, "unknown argument")
    return canonical


def _name_index(cap: Capability) -> dict[str, str]:
    index: dict[str, str] = {}
    for spec in cap.args:
        index[spec.name] = spec.name
        index[spec.name.lstrip("-")] = spec.name
    return index


def _collect_values(
    cap: Capability, provided: Mapping[str, Any], roots: Mapping[str, Sequence[Path]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    allowed = None if cap.allowed_args is None else set(cap.allowed_args)
    for spec in cap.args:
        if allowed is not None and spec.name not in allowed:
            continue
        if spec.name not in provided and not spec.has_default and not spec.required:
            continue
        out[spec.name] = _value_for(spec, provided, cap, roots)
    return out


def _value_for(
    spec: ArgSpec,
    provided: Mapping[str, Any],
    cap: Capability,
    roots: Mapping[str, Sequence[Path]],
) -> Any:
    if spec.name in provided:
        raw: Any = provided[spec.name]
    elif spec.has_default:
        raw = spec.default
    else:
        raise ContractViolation("missing_required", spec.name, "required argument is missing")
    tight = None if cap.allowed_args is None else cap.allowed_args.get(spec.name)
    return _validate_value(spec, raw, tight, roots, for_default=False)


def _validate_value(
    spec: ArgSpec,
    raw: Any,
    tight: ArgTightening | None,
    roots: Mapping[str, Sequence[Path]],
    *,
    for_default: bool,
) -> Any:
    if spec.repeatable:
        return _validate_repeatable(spec, raw, tight, roots, for_default=for_default)
    return _validate_scalar(spec, raw, tight, roots, for_default=for_default)


def _validate_repeatable(
    spec: ArgSpec,
    raw: Any,
    tight: ArgTightening | None,
    roots: Mapping[str, Sequence[Path]],
    *,
    for_default: bool,
) -> list[Any]:
    items = raw if isinstance(raw, list) else [raw]
    if not items:
        raise ContractViolation("bad_type", spec.name, "empty list")
    if len(items) > _MAX_CSV:
        raise ContractViolation("too_long", spec.name, "too many values")
    return [_validate_scalar(spec, item, tight, roots, for_default=for_default) for item in items]


def _validate_scalar(
    spec: ArgSpec,
    raw: Any,
    tight: ArgTightening | None,
    roots: Mapping[str, Sequence[Path]],
    *,
    for_default: bool,
) -> Any:
    kind = spec.type
    if kind == "flag":
        return _val_flag(spec, raw)
    if kind == "int":
        return _val_int(spec, raw, tight)
    if kind == "float":
        return _val_float(spec, raw, tight)
    if kind == "str":
        return _val_str(spec, raw, tight)
    if kind == "enum":
        return _val_enum(spec, raw, tight)
    if kind == "path":
        return _val_path(spec, raw, roots, for_default=for_default)
    if kind == "csv":
        return _val_csv(spec, raw, tight)
    raise ContractError(f"unsupported arg type {kind}")


def _val_flag(spec: ArgSpec, raw: Any) -> bool:
    if not isinstance(raw, bool):
        raise ContractViolation("bad_type", spec.name, "flag must be a boolean")
    return raw


def _val_int(spec: ArgSpec, raw: Any, tight: ArgTightening | None) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ContractViolation("bad_type", spec.name, "expected an integer")
    _check_range(spec, float(raw), tight)
    choices = _effective_choices(spec, tight)
    if choices is not None and str(raw) not in choices:
        raise ContractViolation("bad_choice", spec.name, "value is not an allowed choice")
    return raw


def _val_float(spec: ArgSpec, raw: Any, tight: ArgTightening | None) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ContractViolation("bad_type", spec.name, "expected a number")
    value = float(raw)
    if not math.isfinite(value):
        raise ContractViolation("bad_type", spec.name, "number must be finite")
    _check_range(spec, value, tight)
    return value


def _val_str(spec: ArgSpec, raw: Any, tight: ArgTightening | None) -> str:
    return _check_text(spec, raw, choices=_effective_choices(spec, tight))


def _val_enum(spec: ArgSpec, raw: Any, tight: ArgTightening | None) -> str:
    return _check_text(spec, raw, choices=_effective_choices(spec, tight))


def _val_path(
    spec: ArgSpec,
    raw: Any,
    roots: Mapping[str, Sequence[Path]],
    *,
    for_default: bool,
) -> str:
    if not isinstance(raw, str):
        raise ContractViolation("bad_type", spec.name, "expected a path string")
    if len(raw) > _MAX_PATH:
        raise ContractViolation("too_long", spec.name, "path exceeds 4096 characters")
    if _path_rejected(raw):
        raise ContractViolation("path_escape", spec.name, "path is not confined to an allowed root")
    if for_default:
        return _portable_default(spec, raw)
    role = spec.path_role or ""
    matched = _resolve_in_roots(raw, role, roots)
    if matched is None:
        raise ContractViolation("path_escape", spec.name, "path is outside the allowed roots")
    return str(matched)


def _portable_default(spec: ArgSpec, raw: str) -> str:
    if Path(raw).is_absolute():
        raise ContractViolation("path_escape", spec.name, "default path must be relative")
    return raw


def _val_csv(spec: ArgSpec, raw: Any, tight: ArgTightening | None) -> list[Any]:
    items = _csv_items(spec, raw)
    if len(items) > _MAX_CSV:
        raise ContractViolation("too_long", spec.name, "csv exceeds 32 items")
    return [_parse_csv_item(spec, item, tight) for item in items]


def _csv_items(spec: ArgSpec, raw: Any) -> list[Any]:
    if isinstance(raw, str):
        return _split_csv(spec, raw)
    if isinstance(raw, list):
        if not raw:
            raise ContractViolation("bad_type", spec.name, "csv is empty")
        return list(raw)
    raise ContractViolation("bad_type", spec.name, "csv must be a string or list")


def _split_csv(spec: ArgSpec, raw: str) -> list[str]:
    if contains_shell_metacharacters(raw):
        raise ContractViolation("bad_type", spec.name, "csv contains shell metacharacters")
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(part == "" for part in parts):
        raise ContractViolation("bad_type", spec.name, "csv contains an empty item")
    return parts


def _parse_csv_item(spec: ArgSpec, item: Any, tight: ArgTightening | None) -> Any:
    item_type = spec.item_type or "str"
    if item_type == "int":
        return _csv_int(spec, item, tight)
    if item_type == "float":
        return _csv_float(spec, item, tight)
    if not isinstance(item, str):
        raise ContractViolation("bad_type", spec.name, "csv item must be a string")
    _check_text(spec, item, choices=_effective_choices(spec, tight))
    return item


def _csv_int(spec: ArgSpec, item: Any, tight: ArgTightening | None) -> int:
    if isinstance(item, bool):
        raise ContractViolation("bad_type", spec.name, "csv item must be an integer")
    if isinstance(item, int):
        value = item
    elif isinstance(item, str) and _INT_TOKEN.fullmatch(item):
        value = int(item)
    else:
        raise ContractViolation("bad_type", spec.name, "csv item must be an integer")
    _check_range(spec, float(value), tight)
    _check_csv_choice(spec, str(value), tight)
    return value


def _csv_float(spec: ArgSpec, item: Any, tight: ArgTightening | None) -> float:
    if isinstance(item, bool) or not isinstance(item, (int, float, str)):
        raise ContractViolation("bad_type", spec.name, "csv item must be a number")
    value = _float_item(spec, item)
    _check_range(spec, value, tight)
    _check_csv_choice(spec, _format_float(value), tight)
    return value


def _float_item(spec: ArgSpec, item: Any) -> float:
    if isinstance(item, str):
        if _FLOAT_TOKEN.fullmatch(item) is None:
            raise ContractViolation("bad_type", spec.name, "csv item must be a number")
        value = float(item)
    else:
        value = float(item)
    if not math.isfinite(value):
        raise ContractViolation("bad_type", spec.name, "number must be finite")
    return value


def _check_csv_choice(spec: ArgSpec, rendered: str, tight: ArgTightening | None) -> None:
    choices = _effective_choices(spec, tight)
    if choices is not None and rendered not in choices:
        raise ContractViolation("bad_choice", spec.name, "value is not an allowed choice")


def _check_text(spec: ArgSpec, raw: Any, *, choices: tuple[str, ...] | None) -> str:
    if not isinstance(raw, str):
        raise ContractViolation("bad_type", spec.name, "expected a string")
    if not raw.isprintable() or "\n" in raw or "\r" in raw or "\x00" in raw:
        raise ContractViolation("bad_type", spec.name, "string must be printable")
    if len(raw) > _MAX_STR:
        raise ContractViolation("too_long", spec.name, "string exceeds 256 characters")
    if contains_shell_metacharacters(raw):
        raise ContractViolation("bad_type", spec.name, "string contains shell metacharacters")
    if choices is not None and raw not in choices:
        raise ContractViolation("bad_choice", spec.name, "value is not an allowed choice")
    return raw


def _check_range(spec: ArgSpec, value: float, tight: ArgTightening | None) -> None:
    lo = _merge_min(spec.minimum, None if tight is None else tight.minimum)
    hi = _merge_max(spec.maximum, None if tight is None else tight.maximum)
    if lo is not None and value < lo:
        raise ContractViolation("out_of_range", spec.name, "value is below the minimum")
    if hi is not None and value > hi:
        raise ContractViolation("out_of_range", spec.name, "value is above the maximum")


def _effective_choices(spec: ArgSpec, tight: ArgTightening | None) -> tuple[str, ...] | None:
    extra = None if tight is None else tight.choices
    if spec.choices is None:
        return extra
    if extra is None:
        return spec.choices
    return tuple(item for item in spec.choices if item in set(extra))


def _merge_min(left: float | None, right: float | None) -> float | None:
    values = [item for item in (left, right) if item is not None]
    if not values:
        return None
    return max(values)


def _merge_max(left: float | None, right: float | None) -> float | None:
    values = [item for item in (left, right) if item is not None]
    if not values:
        return None
    return min(values)


def _path_rejected(text: str) -> bool:
    if not text or text.strip() != text:
        return True
    if any(char in _PATH_META for char in text):
        return True
    normalized = text.replace("\\", "/")
    return ".." in Path(normalized).parts


def _resolve_in_roots(text: str, role: str, roots: Mapping[str, Sequence[Path]]) -> Path | None:
    raw = Path(text)
    for root in _roots_for(role, roots):
        matched = _inside_root(raw, root)
        if matched is not None:
            return matched
    return None


def _roots_for(role: str, roots: Mapping[str, Sequence[Path]]) -> list[Path]:
    if role == "any":
        found: list[Path] = []
        for values in roots.values():
            found.extend(values)
        return found
    return list(roots.get(role, ()))


def _inside_root(raw: Path, root: Path) -> Path | None:
    try:
        base = Path(root).resolve()
        candidate = raw if raw.is_absolute() else base / raw
        resolved = candidate.resolve()
    except OSError:
        return None
    if resolved == base or resolved.is_relative_to(base):
        return resolved
    return None


def _parse_tokens(cap: Capability, tokens: Sequence[str]) -> dict[str, Any]:
    by_flag = {spec.name: spec for spec in cap.args if spec.kind == "option"}
    positionals = [spec for spec in cap.args if spec.kind == "argument"]
    raw: dict[str, Any] = {}
    pos_index = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            _consume_positionals(positionals, tokens[index + 1 :], raw, pos_index)
            break
        if token.startswith("-"):
            index = _consume_option(by_flag, tokens, index, raw)
            continue
        pos_index = _assign_positional(positionals, token, raw, pos_index)
        index += 1
    return raw


def _consume_positionals(
    positionals: Sequence[ArgSpec],
    tokens: Sequence[str],
    raw: dict[str, Any],
    pos_index: int,
) -> None:
    for token in tokens:
        pos_index = _assign_positional(positionals, token, raw, pos_index)


def _consume_option(
    by_flag: Mapping[str, ArgSpec],
    tokens: Sequence[str],
    index: int,
    raw: dict[str, Any],
) -> int:
    name, sep, embedded = tokens[index].partition("=")
    spec = by_flag.get(name)
    if spec is None:
        raise ContractViolation("unknown_arg", name, "unknown flag")
    if spec.type == "flag":
        _store(raw, spec, _flag_value(spec, embedded) if sep else True)
        return index + 1
    if sep:
        _store(raw, spec, _coerce_token(spec, embedded))
        return index + 1
    return _consume_separate_value(spec, tokens, index, raw)


def _consume_separate_value(
    spec: ArgSpec, tokens: Sequence[str], index: int, raw: dict[str, Any]
) -> int:
    if index + 1 >= len(tokens) or _looks_like_flag(spec, tokens[index + 1]):
        raise ContractViolation("missing_required", spec.name, "flag is missing a value")
    _store(raw, spec, _coerce_token(spec, tokens[index + 1]))
    return index + 2


def _looks_like_flag(spec: ArgSpec, token: str) -> bool:
    if not token.startswith("-"):
        return False
    return not _numeric_token(spec, token)


def _assign_positional(
    positionals: Sequence[ArgSpec], token: str, raw: dict[str, Any], pos_index: int
) -> int:
    if pos_index >= len(positionals):
        raise ContractViolation("unknown_arg", token, "unexpected positional argument")
    spec = positionals[pos_index]
    _store(raw, spec, _coerce_token(spec, token))
    if spec.repeatable:
        return pos_index
    return pos_index + 1


def _store(raw: dict[str, Any], spec: ArgSpec, value: Any) -> None:
    if spec.repeatable:
        _append_repeatable(raw, spec, value)
        return
    if spec.name in raw:
        raise ContractViolation("unknown_arg", spec.name, "duplicate flag")
    raw[spec.name] = value


def _append_repeatable(raw: dict[str, Any], spec: ArgSpec, value: Any) -> None:
    bucket = raw.setdefault(spec.name, [])
    if not isinstance(bucket, list):
        raise ContractViolation("bad_type", spec.name, "repeatable value clash")
    bucket.append(value)


def _coerce_token(spec: ArgSpec, token: str) -> Any:
    if spec.type == "int":
        return _coerce_int(spec, token)
    if spec.type == "float":
        return _coerce_float(spec, token)
    if spec.type in {"str", "enum", "path", "csv"}:
        return token
    raise ContractViolation("bad_type", spec.name, "unsupported type")


def _coerce_int(spec: ArgSpec, token: str) -> int:
    if _INT_TOKEN.fullmatch(token) is None:
        raise ContractViolation("bad_type", spec.name, "expected an integer")
    return int(token)


def _coerce_float(spec: ArgSpec, token: str) -> float:
    if _FLOAT_TOKEN.fullmatch(token) is None:
        raise ContractViolation("bad_type", spec.name, "expected a number")
    value = float(token)
    if not math.isfinite(value):
        raise ContractViolation("bad_type", spec.name, "number must be finite")
    return value


def _flag_value(spec: ArgSpec, text: str) -> bool:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ContractViolation("bad_type", spec.name, "flag value must be true or false")


def _numeric_token(spec: ArgSpec, token: str) -> bool:
    if spec.type == "int":
        return _INT_TOKEN.fullmatch(token) is not None
    if spec.type == "float":
        return _FLOAT_TOKEN.fullmatch(token) is not None
    return False


def _emit(spec: ArgSpec, value: Any) -> list[str]:
    if spec.repeatable:
        return _emit_repeatable(spec, value)
    if spec.type == "csv":
        return _emit_csv(spec, value)
    return _emit_one(spec, value)


def _emit_repeatable(spec: ArgSpec, value: Any) -> list[str]:
    rendered: list[str] = []
    for item in value:
        rendered.extend(_emit_one(spec, item))
    return rendered


def _emit_one(spec: ArgSpec, value: Any) -> list[str]:
    if spec.type == "flag":
        return [_flag(spec)] if value else []
    rendered = _render(value)
    if spec.kind == "argument":
        return [rendered]
    return [_flag(spec), rendered]


def _emit_csv(spec: ArgSpec, value: Any) -> list[str]:
    rendered = ",".join(_render(item) for item in value)
    if spec.kind == "argument":
        return [rendered]
    return [_flag(spec), rendered]


def _flag(spec: ArgSpec) -> str:
    if spec.name.startswith("-"):
        return spec.name
    return f"--{spec.name}"


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def _format_float(value: float) -> str:
    text = format(value, ".12g")
    if text in {"inf", "-inf", "nan"}:
        raise ContractViolation("bad_type", "", "non-finite number")
    return text


def _receipt_suffix(cap: Capability, argv: Sequence[str]) -> list[str]:
    flag = list(cap.receipt_flag)
    if not flag or "receipts" not in cap.writes:
        return []
    if _contains_sequence(argv, flag):
        return []
    return flag


def _contains_sequence(argv: Sequence[str], flag: Sequence[str]) -> bool:
    width = len(flag)
    if width == 0:
        return False
    return any(
        list(argv[index : index + width]) == list(flag) for index in range(len(argv) - width + 1)
    )


def _ensure_clean(argv: Sequence[str]) -> None:
    for token in argv:
        if contains_shell_metacharacters(token):
            raise ContractViolation("bad_type", "", "argv contains shell metacharacters")


def _invoke(runner: Runner) -> tuple[int, str]:
    try:
        outcome = runner("capabilities", ["list", "--json"])
    except Exception as exc:  # noqa: BLE001 — runner failures are untyped and must not leak
        raise CatalogUnavailable(_short_reason(f"capabilities list failed: {exc}")) from exc
    return _split_outcome(outcome)


def _split_outcome(outcome: Any) -> tuple[int, str]:
    if not isinstance(outcome, tuple) or len(outcome) != 2:
        raise CatalogUnavailable("capabilities runner returned an unexpected result")
    code, stdout = outcome
    if isinstance(code, bool) or not isinstance(code, int):
        raise CatalogUnavailable("capabilities runner returned a non-integer exit code")
    if not isinstance(stdout, str):
        raise CatalogUnavailable("capabilities runner returned non-text output")
    return code, stdout


def _short_reason(text: str) -> str:
    cleaned = redact_text(" ".join(text.split()))
    if len(cleaned) > 180:
        return cleaned[:177] + "..."
    return cleaned or "capabilities catalog unavailable"


def _read_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = read_json(path)
    except (OSError, StoreCorruptError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _fresh_catalog(
    payload: Mapping[str, Any], *, version: str, now: float, ttl_s: float
) -> CapabilityCatalog | None:
    if payload.get("fx1_version") != version:
        return None
    fetched = _fetched_at(payload.get("fetched_at"))
    if fetched is None or now < fetched or now - fetched >= ttl_s:
        return None
    document = payload.get("document")
    if not isinstance(document, Mapping):
        return None
    if payload.get("digest") != catalog_digest(document):
        return None
    return _validated_document(document)


def _fetched_at(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _validated_document(document: Mapping[str, Any]) -> CapabilityCatalog | None:
    try:
        catalog = CapabilityCatalog.from_json(document)
    except ContractError:
        return None
    if catalog.digest != catalog_digest(document):
        return None
    return catalog


def _resolve_config_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    override = os.environ.get("FX1_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".fx1"


def _mapping(value: Any, ctx: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{ctx} must be an object")
    return value


def _bool(value: Any, ctx: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{ctx} must be a boolean")
    return value


def _optional_bool(obj: Mapping[str, Any], key: str, default: bool, ctx: str) -> bool:
    if key not in obj:
        return default
    return _bool(obj[key], f"{ctx}.{key}")


def _text(value: Any, ctx: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{ctx} must be a string")
    if len(value) > limit:
        raise ContractError(f"{ctx} exceeds {limit} characters")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ContractError(f"{ctx} contains a control character")
    return value


def _member(value: Any, allowed: frozenset[str], ctx: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ContractError(f"{ctx} is invalid")
    return value


def _optional_member(
    obj: Mapping[str, Any], key: str, allowed: frozenset[str], ctx: str
) -> str | None:
    if key not in obj or obj[key] is None:
        return None
    return _member(obj[key], allowed, f"{ctx}.{key}")


def _optional_budget(value: Any, ctx: str, *, allow_null: bool) -> float:
    """Null budgets are 0 only for entries that cannot be executed."""
    if value is None and allow_null:
        return 0.0
    return _nonneg_number(value, ctx)


def _optional_budget_int(value: Any, ctx: str, *, allow_null: bool) -> int:
    if value is None and allow_null:
        return 0
    return _nonneg_int(value, ctx)


def _nonneg_number(value: Any, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{ctx} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ContractError(f"{ctx} must be a nonnegative finite number")
    return number


def _nonneg_int(value: Any, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{ctx} must be a nonnegative integer")
    return value


def _optional_number(obj: Mapping[str, Any], key: str, ctx: str) -> float | None:
    if key not in obj or obj[key] is None:
        return None
    return _any_number(obj[key], f"{ctx}.{key}")


def _any_number(value: Any, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{ctx} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{ctx} must be finite")
    return number


def _check_bounds_pair(minimum: float | None, maximum: float | None, ctx: str) -> None:
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ContractError(f"{ctx} min exceeds max")


def _str_tuple(value: Any, ctx: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContractError(f"{ctx} must be a list of strings")
    if not allow_empty and not value:
        raise ContractError(f"{ctx} must be non-empty")
    return tuple(_nonempty_item(item, ctx) for item in value)


def _nonempty_item(item: Any, ctx: str) -> str:
    if not isinstance(item, str) or item == "":
        raise ContractError(f"{ctx} must be a list of non-empty strings")
    return item


def _pattern_tuple(value: Any, ctx: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    items = _str_tuple(value, ctx, allow_empty=True)
    for item in items:
        if pattern.fullmatch(item) is None:
            raise ContractError(f"{ctx} contains an invalid entry")
    return items


def _id_tuple(value: Any, ctx: str) -> tuple[str, ...]:
    items = _str_tuple(value, ctx, allow_empty=True)
    for item in items:
        if _ID_RE.fullmatch(item) is None:
            raise ContractError(f"{ctx} contains an invalid id")
    return items


def _optional_choices(obj: Mapping[str, Any], ctx: str) -> tuple[str, ...] | None:
    if "choices" not in obj or obj["choices"] is None:
        return None
    return _str_tuple(obj["choices"], f"{ctx}.choices", allow_empty=False)


def _help_text(obj: Mapping[str, Any], ctx: str) -> str:
    if "help" not in obj:
        return ""
    text = _text(obj.get("help"), f"{ctx}.help", limit=160)
    if "\n" in text or "\r" in text:
        raise ContractError(f"{ctx}.help must be a single line")
    return text


def _json_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    return isinstance(value, int)


def _marker_rejected(text: str) -> bool:
    return "\n" in text or "\r" in text or contains_shell_metacharacters(text)


def _reject_shell_tokens(tokens: Sequence[str], ctx: str) -> None:
    if any(contains_shell_metacharacters(token) for token in tokens):
        raise ContractError(f"{ctx} contains shell metacharacters")
