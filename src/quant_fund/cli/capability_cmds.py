"""Read-only ``dipcatcher capabilities`` contract.

The research catalog is imported only when a family view is built. Command
introspection loads ``quant_fund.cli.main`` on first use, not at import.
Published defaults drop absolute paths and URL hostnames. ``executable`` is
true only for overlay status ``qualified`` whose command, arg hints, and
evidence files all resolve.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from quant_fund.cli.capability_overlay import (
    AUDITED_ON,
    CURATED_OVERLAY,
    EVIDENCE_CLASSES,
    FAIL_CLOSED_REASON,
    QUALIFICATION_STATUSES,
    RULESET,
    SCHEMA,
    overlay_sha256,
)

NO_RUNNER_REASON = "runs only inside the full `research` scorecard; no per-family runner exists"
_CONSEQUENTIAL_TOKENS = frozenset({"train", "optimize", "paper"})
_WEAKENING_ARGS = frozenset({"--check-index", "--write", "--claim-live", "--dev"})
_ARG_TYPES = frozenset({"flag", "int", "float", "str", "enum", "path", "csv"})
_ARG_KINDS = frozenset({"option", "argument"})
_PATH_ROLES = frozenset({"config", "input", "output_dir", "any"})
_ITEM_TYPES = frozenset({"int", "float", "str"})
_FAMILY_STATUSES = frozenset({"required", "live", "retired", "all"})
_ENTRY_KEYS = frozenset(
    {
        "id",
        "version",
        "purpose",
        "tasks",
        "aliases",
        "command",
        "allowed_args",
        "arg_hints",
        "inputs",
        "outputs",
        "dependencies",
        "post_checks",
        "limits",
        "qualification",
        "evidence_class",
    }
)
_CAP_KEYS = frozenset(
    {
        "id",
        "version",
        "purpose",
        "tasks",
        "aliases",
        "command",
        "args",
        "inputs",
        "outputs",
        "dependencies",
        "post_checks",
        "limits",
        "qualification",
        "evidence_class",
        "executable",
        "executable_reason",
    }
)
_COMMAND_KEYS = frozenset({"argv", "role", "read_only", "writes", "consequential", "timeout_s"})
_INPUT_KEYS = frozenset({"data_labels", "requires_dataset", "availability_timing"})
_OUTPUT_KEYS = frozenset({"scores", "receipt_schema", "receipt_flag", "receipt_stdout_marker"})
_LIMIT_KEYS = frozenset({"max_runtime_s", "max_output_chars"})
_QUAL_KEYS = frozenset({"status", "evidence", "ruleset", "audited_on"})
_ARG_REQUIRED = frozenset({"name", "kind", "type", "required", "default", "repeatable", "help"})
_ARG_OPTIONAL = frozenset({"choices", "min", "max", "item_type", "path_role"})
_POST_REQUIRED = frozenset({"kind"})
_POST_OPTIONAL = frozenset({"from"})
_ARG_ORDER = (
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
)
_STATUS_REASON = {
    "retired": "retired capability is not executable",
    "experimental": "experimental capability is not executable",
    "unqualified": "unqualified capability is not executable",
}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*( [a-z0-9][a-z0-9-]*)*$")
_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_TASK_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*\.py$")
_HOSTPATH = re.compile(r"/home(?:/|\"|$)|/Users/|[A-Za-z]:[\\/]|https?://")
_SECRET = re.compile(
    r"(?i)(api[_-]?key\s*[:=]|secret[_-]?key|BEGIN [A-Z0-9 ]*PRIVATE KEY|"
    r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})"
)
_FQDN = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|dev|ai)\b", re.I)

capabilities_app = typer.Typer(
    help="Read-only capability and benchmark-family contract. Missing qualification fails closed.",
    no_args_is_help=True,
)


@dataclass
class LabCommand:
    """One invocable lab command, already introspected."""

    argv: tuple[str, ...]
    help_text: str
    hidden: bool
    args: tuple[dict[str, Any], ...]

    @property
    def id(self) -> str:
        return " ".join(self.argv)


@dataclass(frozen=True)
class FamilyCensus:
    """Ordered required names, sorted live names, and retired rows."""

    required: tuple[str, ...]
    live: tuple[str, ...]
    retired: tuple[tuple[str, str, int], ...]


@dataclass
class _Built:
    document: dict[str, Any]
    problems: list[str]
    commands: tuple[LabCommand, ...]


def find_repo_root(start: Path | None = None) -> Path | None:
    """Return the checkout root, or None when this file is not inside one."""
    here = (start or Path(__file__)).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "quant_fund").is_dir():
            return parent
    return None


def dumps_document(doc: Mapping[str, Any]) -> str:
    """Serialize one document with stable key order and a trailing newline."""
    return json.dumps(doc, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def commands_from_app(app: Any) -> tuple[LabCommand, ...]:
    """Introspect a Typer app or an already-built Click group."""
    root = typer.main.get_command(app) if isinstance(app, typer.Typer) else app
    return tuple(sorted(_collect(root), key=lambda item: item.id))


def _collect(root: Any) -> list[LabCommand]:
    """Walk groups. A one-command Typer app is the command itself, not a group."""
    found = _walk(root, ())
    if found or isinstance(getattr(root, "commands", None), dict):
        return found
    name = getattr(root, "name", None)
    if not name:
        return []
    return [_leaf(root, (str(name),))]


def list_lab_commands() -> tuple[LabCommand, ...]:
    """Introspect the real lab CLI. Imports ``quant_fund.cli.main`` on call."""
    from quant_fund.cli.main import app as lab_app

    return commands_from_app(lab_app)


def build_capability_document(
    *,
    overlay: Sequence[Mapping[str, Any]] | None = None,
    commands: Sequence[LabCommand] | None = None,
    repo_root: Path | None = None,
    discover_repo: bool = True,
    executable_only: bool = False,
) -> dict[str, Any]:
    """Build the ``dipcatcher.capabilities.v1`` list document."""
    return _assemble(
        overlay=overlay,
        commands=commands,
        repo_root=repo_root,
        discover_repo=discover_repo,
        executable_only=executable_only,
    ).document


def describe_capability(
    capability_id: str,
    *,
    overlay: Sequence[Mapping[str, Any]] | None = None,
    commands: Sequence[LabCommand] | None = None,
    repo_root: Path | None = None,
    discover_repo: bool = True,
) -> dict[str, Any]:
    """Return one capability. ``error`` is set when the id is missing or duplicated."""
    doc = build_capability_document(
        overlay=overlay,
        commands=commands,
        repo_root=repo_root,
        discover_repo=discover_repo,
    )
    matches = [cap for cap in doc["capabilities"] if cap["id"] == capability_id]
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "view": "describe",
        "generated_by": doc["generated_by"],
        "id": capability_id,
        "capability": matches[0] if len(matches) == 1 else None,
    }
    error = _describe_error(matches)
    if error is not None:
        body["error"] = error
    return body


def load_family_census() -> FamilyCensus:
    """Load scorecard family sets. Imports the research registry on call."""
    from quant_fund.research.catalog.registry import (
        BENCHMARK_FAMILY_ORDER,
        LIVE_OPTIONAL_BENCHMARK_FAMILIES,
        REQUIRED_BENCHMARK_FAMILIES,
        RETIRED_BENCHMARK_FAMILIES,
    )

    return _census_from_sets(
        required=set(REQUIRED_BENCHMARK_FAMILIES),
        order=tuple(BENCHMARK_FAMILY_ORDER),
        live=set(LIVE_OPTIONAL_BENCHMARK_FAMILIES),
        retired=RETIRED_BENCHMARK_FAMILIES,
    )


def build_family_document(
    *,
    status: str = "all",
    limit: int = 50,
    offset: int = 0,
    census: FamilyCensus | None = None,
) -> dict[str, Any]:
    """Page family metadata. No family is executable; there is no per-family runner."""
    _validate_family_page(status, limit, offset)
    loaded = census if census is not None else load_family_census()
    rows = _family_rows(loaded, status)
    total = len(rows)
    next_offset = offset + limit if offset + limit < total else None
    return {
        "schema": SCHEMA,
        "view": "families",
        "status": status,
        "counts": {
            "required": len(loaded.required),
            "live": len(loaded.live),
            "retired": len(loaded.retired),
        },
        "total": total,
        "offset": offset,
        "limit": limit,
        "next_offset": next_offset,
        "families": rows[offset : offset + limit],
    }


def audit_family_contract(census: FamilyCensus | None = None) -> list[str]:
    """Audit the family view. Pass ``census`` to avoid importing the registry."""
    loaded = census if census is not None else load_family_census()
    limit = max(_census_total(loaded), 1)
    doc = build_family_document(status="all", limit=limit, offset=0, census=loaded)
    again = build_family_document(status="all", limit=limit, offset=0, census=loaded)
    problems = _family_doc_problems(doc, loaded)
    if dumps_document(doc) != dumps_document(again):
        problems.append("family document is not deterministic")
    problems.extend(_scan_text(dumps_document(doc), "families"))
    return problems


def audit_capabilities(
    *,
    overlay: Sequence[Mapping[str, Any]] | None = None,
    commands: Sequence[LabCommand] | None = None,
    repo_root: Path | None = None,
    discover_repo: bool = True,
    include_families: bool = False,
) -> list[str]:
    """Return sorted problems. An empty list means the contract audits clean.

    ``include_families`` imports the research registry unless a caller is only
    auditing an injected capability overlay.
    """
    built = _assemble(
        overlay=overlay,
        commands=commands,
        repo_root=repo_root,
        discover_repo=discover_repo,
        executable_only=False,
    )
    again = _assemble(
        overlay=overlay,
        commands=commands,
        repo_root=repo_root,
        discover_repo=discover_repo,
        executable_only=False,
    )
    problems = list(built.problems)
    if dumps_document(built.document) != dumps_document(again.document):
        problems.append("capability document is not deterministic")
    problems.extend(_structural_problems(built.document))
    problems.extend(_coverage_problems(built.document, built.commands))
    problems.extend(_scan_text(dumps_document(built.document), "capabilities"))
    if include_families:
        problems.extend(audit_family_contract())
    return sorted(set(problems))


def _assemble(
    *,
    overlay: Sequence[Mapping[str, Any]] | None,
    commands: Sequence[LabCommand] | None,
    repo_root: Path | None,
    discover_repo: bool,
    executable_only: bool,
) -> _Built:
    resolved = _resolve_commands(commands)
    chosen = _resolve_overlay(overlay)
    repo = _resolve_repo(repo_root, discover_repo=discover_repo)
    caps, problems = _publish(resolved, chosen, repo)
    caps.sort(key=_cap_sort_key)
    problems.extend(_duplicate_problems(caps))
    return _Built(
        document=_document(chosen, len(resolved), caps, executable_only=executable_only),
        problems=problems,
        commands=resolved,
    )


def _publish(
    commands: Sequence[LabCommand],
    overlay: Sequence[Mapping[str, Any]],
    repo: Path | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    grouped = _group_overlay(overlay)
    known = {cmd.id: cmd for cmd in commands}
    root_tokens = {cmd.argv[0] for cmd in commands}
    caps: list[dict[str, Any]] = []
    problems: list[str] = []
    caps.extend(_publish_known(commands, grouped, repo, root_tokens, problems))
    caps.extend(_publish_phantoms(grouped, known, repo, root_tokens, problems))
    return caps, problems


def _publish_known(
    commands: Sequence[LabCommand],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    repo: Path | None,
    root_tokens: set[str],
    problems: list[str],
) -> list[dict[str, Any]]:
    caps: list[dict[str, Any]] = []
    for cmd in commands:
        entries = grouped.get(cmd.id, ())
        if not entries:
            caps.append(_unqualified_capability(cmd))
            continue
        for entry in entries:
            cap, extra = _overlay_capability(entry, cmd, repo, root_tokens)
            caps.append(cap)
            problems.extend(extra)
    return caps


def _publish_phantoms(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    known: Mapping[str, LabCommand],
    repo: Path | None,
    root_tokens: set[str],
    problems: list[str],
) -> list[dict[str, Any]]:
    caps: list[dict[str, Any]] = []
    for ident in sorted(grouped):
        if ident in known:
            continue
        for entry in grouped[ident]:
            cap, extra = _overlay_capability(entry, None, repo, root_tokens)
            caps.append(cap)
            problems.extend(extra)
    return caps


def _overlay_capability(
    entry: Mapping[str, Any],
    command: LabCommand | None,
    repo: Path | None,
    root_tokens: set[str],
) -> tuple[dict[str, Any], list[str]]:
    shape = _shape_problems(entry)
    if shape:
        return _malformed_capability(entry), shape
    args, gates = _entry_gates(entry, command, repo, root_tokens)
    status = str(entry["qualification"]["status"])
    executable, reason = _decide(status, gates, in_overlay=True)
    executable, reason = _apply_force(entry, executable, reason)
    cap = _from_overlay(entry, args, executable, reason)
    problems = [*gates, *_force_problems(str(entry["id"]), status, executable)]
    return cap, problems


def _entry_gates(
    entry: Mapping[str, Any],
    command: LabCommand | None,
    repo: Path | None,
    root_tokens: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    problems = _class_problems(entry)
    problems.extend(_budget_problems(entry))
    if command is None:
        return [], [*problems, *_phantom_gates(entry, root_tokens)]
    args, arg_problems = _select_args(command, entry)
    problems.extend(arg_problems)
    problems.extend(_argv_mismatch(entry, command))
    if str(entry["qualification"]["status"]) == "qualified":
        problems.extend(
            _evidence_problems(entry["qualification"]["evidence"], repo, str(entry["id"]))
        )
    return args, problems


def _shape_problems(entry: Mapping[str, Any]) -> list[str]:
    ident = entry.get("id", "<missing>")
    if not isinstance(ident, str) or not ident:
        return ["overlay entry is missing id"]
    missing = [key for key in sorted(_ENTRY_KEYS) if key not in entry]
    if missing:
        return [f"{ident}: overlay entry missing keys: {', '.join(missing)}"]
    if not isinstance(entry["command"], dict) or not isinstance(entry["qualification"], dict):
        return [f"{ident}: overlay entry is malformed"]
    argv = entry["command"].get("argv")
    if not isinstance(argv, list) or not argv:
        return [f"{ident}: overlay command.argv is malformed"]
    if ident != " ".join(str(part) for part in argv):
        return [f"{ident}: id does not match command.argv"]
    return []


def _malformed_capability(entry: Mapping[str, Any]) -> dict[str, Any]:
    raw = entry.get("id")
    ident = raw if isinstance(raw, str) and _ID_RE.fullmatch(raw) else "invalid"
    cap = _unqualified_capability(LabCommand((ident,), "", False, ()))
    cap["purpose"] = "Malformed qualification overlay entry."
    cap["executable_reason"] = "overlay entry is malformed"
    return cap


def _phantom_gates(entry: Mapping[str, Any], root_tokens: set[str]) -> list[str]:
    ident = str(entry["id"])
    argv = [str(part) for part in entry["command"]["argv"]]
    problems = [f"{ident}: command argv is not a real lab command"]
    if not argv or argv[0] not in root_tokens:
        problems.append(f"{ident}: command.argv[0] is not a real lab command")
    return problems


def _class_problems(entry: Mapping[str, Any]) -> list[str]:
    ident = str(entry["id"])
    problems: list[str] = []
    if entry.get("evidence_class") not in EVIDENCE_CLASSES:
        problems.append(f"{ident}: evidence_class is invalid")
    status = (
        entry["qualification"].get("status") if isinstance(entry["qualification"], dict) else None
    )
    if status not in QUALIFICATION_STATUSES:
        problems.append(f"{ident}: qualification status is invalid")
    return problems


def _budget_problems(entry: Mapping[str, Any]) -> list[str]:
    ident = str(entry["id"])
    qual = entry["qualification"]
    if not isinstance(qual, dict) or qual.get("status") != "qualified":
        return []
    problems: list[str] = []
    command = entry["command"] if isinstance(entry["command"], dict) else {}
    if not _positive_int(command.get("timeout_s")):
        problems.append(f"{ident}: qualified command is missing timeout_s")
    limits = entry.get("limits")
    if not isinstance(limits, dict):
        return [*problems, f"{ident}: limits must be an object"]
    if not _positive_int(limits.get("max_runtime_s")):
        problems.append(f"{ident}: qualified command is missing max_runtime_s")
    if not _positive_int(limits.get("max_output_chars")):
        problems.append(f"{ident}: qualified command is missing max_output_chars")
    return problems


def _argv_mismatch(entry: Mapping[str, Any], command: LabCommand) -> list[str]:
    argv = [str(part) for part in entry["command"]["argv"]]
    if argv != list(command.argv):
        return [f"{entry['id']}: command.argv does not match the lab command"]
    return []


def _decide(status: str, gates: Sequence[str], *, in_overlay: bool) -> tuple[bool, str]:
    if not in_overlay:
        return False, FAIL_CLOSED_REASON
    if status == "retired":
        return False, _STATUS_REASON["retired"]
    if status != "qualified" or gates:
        parts: list[str] = []
        if status != "qualified":
            parts.append(_STATUS_REASON.get(status, "capability is not executable"))
        parts.extend(gates)
        return False, "; ".join(parts)
    return True, ""


def _apply_force(entry: Mapping[str, Any], executable: bool, reason: str) -> tuple[bool, str]:
    """Honor an explicit test hook. The curated overlay does not set it."""
    if "force_executable" not in entry:
        return executable, reason
    if bool(entry["force_executable"]):
        return True, ""
    if reason:
        return False, reason
    return False, "forced non-executable"


def _force_problems(ident: str, status: str, executable: bool) -> list[str]:
    problems: list[str] = []
    if executable and status != "qualified":
        problems.append(f"{ident}: executable with non-qualified status")
    if status == "retired" and executable:
        problems.append(f"{ident}: retired capability is executable")
    return problems


def _select_args(
    command: LabCommand, entry: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    ident = str(entry["id"])
    by_name = {str(arg["name"]): arg for arg in command.args}
    hints = entry.get("arg_hints")
    allowed = entry.get("allowed_args")
    problems = _unknown_hint_problems(ident, hints, by_name)
    if not isinstance(allowed, list):
        return [], [*problems, f"{ident}: allowed_args must be a list"]
    if not isinstance(hints, dict):
        return [], [*problems, f"{ident}: arg_hints must be an object"]
    selected = _allowed_args(ident, allowed, hints, by_name, problems)
    problems.extend(_missing_required(ident, command, allowed))
    status = str(entry["qualification"]["status"])
    return _drop_weakening(ident, selected, problems, qualified=status == "qualified"), problems


def _unknown_hint_problems(
    ident: str, hints: object, by_name: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    if not isinstance(hints, dict):
        return []
    return [f"{ident}: unknown arg hint: {name}" for name in hints if name not in by_name]


def _allowed_args(
    ident: str,
    allowed: Sequence[object],
    hints: Mapping[str, Any],
    by_name: Mapping[str, Mapping[str, Any]],
    problems: list[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for name in allowed:
        token = str(name)
        if token not in by_name:
            problems.append(f"{ident}: allowed_args names an unknown parameter: {token}")
            continue
        hint = hints.get(token)
        if hint is not None and not isinstance(hint, dict):
            problems.append(f"{ident}: arg hint {token} must be an object")
            continue
        selected.append(_with_hint(dict(by_name[token]), hint, problems, f"{ident} {token}"))
    return selected


def _missing_required(ident: str, command: LabCommand, allowed: Sequence[object]) -> list[str]:
    published = {str(name) for name in allowed}
    return [
        f"{ident}: allowed_args omits required parameter: {arg['name']}"
        for arg in command.args
        if arg["required"] and str(arg["name"]) not in published
    ]


def _drop_weakening(
    ident: str,
    args: list[dict[str, Any]],
    problems: list[str],
    *,
    qualified: bool,
) -> list[dict[str, Any]]:
    if not qualified:
        return args
    kept: list[dict[str, Any]] = []
    for arg in args:
        if str(arg["name"]) in _WEAKENING_ARGS:
            problems.append(f"{ident}: weakening flag is not exposable: {arg['name']}")
            continue
        kept.append(arg)
    return kept


def _with_hint(
    arg: dict[str, Any],
    hint: Mapping[str, Any] | None,
    problems: list[str],
    where: str,
) -> dict[str, Any]:
    if not hint:
        return _order_arg(arg)
    _apply_type_hint(arg, hint, problems, where)
    _apply_bound_hint(arg, hint, problems, where, "min", higher_is_tighter=True)
    _apply_bound_hint(arg, hint, problems, where, "max", higher_is_tighter=False)
    _apply_choices_hint(arg, hint, problems, where)
    _apply_default_hint(arg, hint, problems, where)
    if "path_role" in hint:
        arg["path_role"] = str(hint["path_role"])
    if "item_type" in hint:
        arg["item_type"] = str(hint["item_type"])
    return _order_arg(arg)


def _apply_type_hint(
    out: dict[str, Any], hint: Mapping[str, Any], problems: list[str], where: str
) -> None:
    if "type" not in hint:
        return
    hinted = str(hint["type"])
    current = str(out.get("type"))
    if hinted in {current, "csv"}:
        out["type"] = hinted
        return
    problems.append(f"{where}: overlay type {hinted} does not match CLI type {current}")


def _apply_bound_hint(
    out: dict[str, Any],
    hint: Mapping[str, Any],
    problems: list[str],
    where: str,
    key: str,
    *,
    higher_is_tighter: bool,
) -> None:
    if key not in hint:
        return
    incoming = hint[key]
    if not _is_number(incoming):
        problems.append(f"{where}: overlay {key} is not a number")
        return
    number = _as_number(incoming)
    if key not in out:
        out[key] = number
        return
    current = out[key]
    if not _is_number(current):
        problems.append(f"{where}: CLI {key} is not a number")
        return
    if _loosens(number, _as_number(current), higher_is_tighter=higher_is_tighter):
        problems.append(f"{where}: overlay loosens {key}")
        return
    out[key] = number


def _apply_choices_hint(
    out: dict[str, Any], hint: Mapping[str, Any], problems: list[str], where: str
) -> None:
    if "choices" not in hint:
        return
    raw = hint["choices"]
    if not isinstance(raw, list) or not raw:
        problems.append(f"{where}: overlay choices must be a non-empty list")
        return
    incoming = [str(item) for item in raw]
    current = out.get("choices")
    if isinstance(current, list) and not set(incoming).issubset({str(item) for item in current}):
        problems.append(f"{where}: overlay choices are not a subset of the CLI choices")
        return
    out["choices"] = incoming
    if out.get("type") == "str":
        out["type"] = "enum"


def _apply_default_hint(
    out: dict[str, Any], hint: Mapping[str, Any], problems: list[str], where: str
) -> None:
    if "default" not in hint:
        return
    value = hint["default"]
    if value is None:
        out["default"] = None
        return
    text = str(value)
    if _nonportable(text):
        problems.append(f"{where}: overlay default is not portable")
        out["default"] = None
        return
    out["default"] = text


def _loosens(incoming: int | float, current: int | float, *, higher_is_tighter: bool) -> bool:
    if higher_is_tighter:
        return incoming < current
    return incoming > current


def _evidence_problems(evidence: object, repo: Path | None, ident: str) -> list[str]:
    if not isinstance(evidence, list) or not evidence:
        return [f"{ident}: qualification evidence is empty"]
    problems: list[str] = []
    for item in evidence:
        problems.extend(_one_evidence(ident, item, repo))
    return problems


def _one_evidence(ident: str, item: object, repo: Path | None) -> list[str]:
    if not isinstance(item, str) or not _evidence_ok(item):
        return [f"{ident}: qualification evidence path is not a relative test file"]
    if repo is not None and not (repo / item).is_file():
        return [f"{ident}: qualification evidence missing: {item}"]
    return []


def _evidence_ok(rel: str) -> bool:
    if _EVIDENCE_RE.fullmatch(rel) is None:
        return False
    return ".." not in rel.split("/")


def _unqualified_capability(command: LabCommand) -> dict[str, Any]:
    return _capability(
        ident=command.id,
        version="0",
        purpose=_unqualified_purpose(command.help_text),
        tasks=[],
        aliases=[],
        argv=command.argv,
        role="unspecified",
        read_only=False,
        writes=[],
        timeout_s=None,
        consequential=False,
        args=list(command.args),
        data_labels=[],
        requires_dataset=False,
        availability="unavailable — unqualified",
        scores=[],
        receipt_schema="",
        receipt_flag=[],
        receipt_marker="",
        dependencies=[],
        post_checks=[],
        max_runtime_s=None,
        max_output_chars=None,
        status="unqualified",
        evidence=[],
        evidence_class=_unqualified_class(command.help_text, command.argv),
        executable=False,
        executable_reason=FAIL_CLOSED_REASON,
    )


def _from_overlay(
    entry: Mapping[str, Any],
    args: Sequence[Mapping[str, Any]],
    executable: bool,
    reason: str,
) -> dict[str, Any]:
    command = entry["command"]
    inputs = entry["inputs"]
    outputs = entry["outputs"]
    limits = entry["limits"]
    qual = entry["qualification"]
    argv = tuple(str(part) for part in command["argv"])
    return _capability(
        ident=str(entry["id"]),
        version=str(entry["version"]),
        purpose=str(entry["purpose"]),
        tasks=[str(item) for item in entry["tasks"]],
        aliases=[str(item) for item in entry["aliases"]],
        argv=argv,
        role=str(command["role"]),
        read_only=bool(command["read_only"]),
        writes=[str(item) for item in command["writes"]],
        timeout_s=_optional_int(command.get("timeout_s")),
        consequential=bool(command["consequential"]),
        args=args,
        data_labels=[str(item) for item in inputs["data_labels"]],
        requires_dataset=bool(inputs["requires_dataset"]),
        availability=str(inputs["availability_timing"]),
        scores=[str(item) for item in outputs["scores"]],
        receipt_schema=str(outputs["receipt_schema"]),
        receipt_flag=[str(item) for item in outputs["receipt_flag"]],
        receipt_marker=str(outputs["receipt_stdout_marker"]),
        dependencies=[str(item) for item in entry["dependencies"]],
        post_checks=[dict(item) for item in entry["post_checks"]],
        max_runtime_s=_optional_int(limits.get("max_runtime_s")),
        max_output_chars=_optional_int(limits.get("max_output_chars")),
        status=str(qual["status"]),
        evidence=[str(item) for item in qual["evidence"]],
        evidence_class=str(entry["evidence_class"]),
        executable=executable,
        executable_reason=reason,
    )


def _capability(
    *,
    ident: str,
    version: str,
    purpose: str,
    tasks: Sequence[str],
    aliases: Sequence[str],
    argv: Sequence[str],
    role: str,
    read_only: bool,
    writes: Sequence[str],
    timeout_s: int | None,
    consequential: bool,
    args: Sequence[Mapping[str, Any]],
    data_labels: Sequence[str],
    requires_dataset: bool,
    availability: str,
    scores: Sequence[str],
    receipt_schema: str,
    receipt_flag: Sequence[str],
    receipt_marker: str,
    dependencies: Sequence[str],
    post_checks: Sequence[Mapping[str, str]],
    max_runtime_s: int | None,
    max_output_chars: int | None,
    status: str,
    evidence: Sequence[str],
    evidence_class: str,
    executable: bool,
    executable_reason: str,
) -> dict[str, Any]:
    return {
        "id": ident,
        "version": version,
        "purpose": purpose,
        "tasks": list(tasks),
        "aliases": list(aliases),
        "command": {
            "argv": list(argv),
            "role": role,
            "read_only": read_only,
            "writes": list(writes),
            "consequential": _consequential(argv, consequential),
            "timeout_s": timeout_s,
        },
        "args": [_order_arg(arg) for arg in args],
        "inputs": {
            "data_labels": list(data_labels),
            "requires_dataset": requires_dataset,
            "availability_timing": availability,
        },
        "outputs": {
            "scores": list(scores),
            "receipt_schema": receipt_schema,
            "receipt_flag": list(receipt_flag),
            "receipt_stdout_marker": receipt_marker,
        },
        "dependencies": list(dependencies),
        "post_checks": [dict(item) for item in post_checks],
        "limits": {"max_runtime_s": max_runtime_s, "max_output_chars": max_output_chars},
        "qualification": {
            "status": status,
            "evidence": list(evidence),
            "ruleset": RULESET,
            "audited_on": AUDITED_ON,
        },
        "evidence_class": evidence_class,
        "executable": executable,
        "executable_reason": executable_reason,
    }


def _consequential(argv: Sequence[str], requested: bool) -> bool:
    return requested or any(part in _CONSEQUENTIAL_TOKENS for part in argv)


def _unqualified_purpose(help_text: str) -> str:
    if help_text:
        return help_text
    return "Listed lab command without qualification metadata."


def _unqualified_class(help_text: str, argv: Sequence[str]) -> str:
    if "SYNTHETIC" in help_text.upper():
        return "SYNTHETIC"
    if "doctor" in argv or any(part.startswith("verify") for part in argv):
        return "VERIFICATION"
    return "RESEARCH"


def _walk(node: Any, prefix: tuple[str, ...]) -> list[LabCommand]:
    found: list[LabCommand] = []
    children = getattr(node, "commands", None)
    if not isinstance(children, dict):
        return found
    for name in children:
        child = children[name]
        argv = prefix + (str(name),)
        nested = getattr(child, "commands", None)
        if isinstance(nested, dict) and nested:
            found.extend(_walk(child, argv))
            continue
        found.append(_leaf(child, argv))
    return found


def _leaf(node: Any, argv: tuple[str, ...]) -> LabCommand:
    params = getattr(node, "params", ()) or ()
    args = tuple(_arg_from_param(param) for param in params if not _skip_param(param))
    return LabCommand(
        argv=argv,
        help_text=_clip_help(getattr(node, "help", None)),
        hidden=bool(getattr(node, "hidden", False)),
        args=args,
    )


def _skip_param(param: Any) -> bool:
    """Ignore Click's help option and Typer's shell-completion installers."""
    name = str(getattr(param, "name", ""))
    return name in {"help", "install_completion", "show_completion"}


def _arg_from_param(param: Any) -> dict[str, Any]:
    kind = _kind(param)
    name = _public_name(param, kind)
    schema_type = _schema_type(param)
    arg: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "type": schema_type,
        "required": bool(getattr(param, "required", False)),
        "default": _default_text(getattr(param, "default", None)),
        "repeatable": bool(getattr(param, "multiple", False)),
        "help": _clip_help(getattr(param, "help", None)),
    }
    choices = _choice_list(param)
    if choices is not None:
        arg["choices"] = choices
    lower, upper = _range_bounds(param)
    if lower is not None:
        arg["min"] = lower
    if upper is not None:
        arg["max"] = upper
    if schema_type == "path":
        arg["path_role"] = _path_role(name)
    return _order_arg(arg)


def _kind(param: Any) -> str:
    if "argument" in type(param).__name__.lower():
        return "argument"
    opts = [str(opt) for opt in (getattr(param, "opts", ()) or ())]
    if opts and not opts[0].startswith("-"):
        return "argument"
    return "option"


def _public_name(param: Any, kind: str) -> str:
    opts = [str(opt) for opt in (getattr(param, "opts", ()) or ())]
    if kind == "option":
        longs = [opt for opt in opts if opt.startswith("--")]
        if longs:
            return max(longs, key=len)
        shorts = [opt for opt in opts if opt.startswith("-")]
        if shorts:
            return max(shorts, key=len)
    if opts:
        return opts[0]
    return str(getattr(param, "name", "arg"))


def _schema_type(param: Any) -> str:
    if bool(getattr(param, "is_flag", False)):
        return "flag"
    type_name = type(getattr(param, "type", object())).__name__.lower()
    if "choice" in type_name:
        return "enum"
    if "path" in type_name:
        return "path"
    if "float" in type_name:
        return "float"
    if "int" in type_name:
        return "int"
    return "str"


def _choice_list(param: Any) -> list[str] | None:
    annotation = getattr(param, "type", None)
    if annotation is None or "choice" not in type(annotation).__name__.lower():
        return None
    raw = getattr(annotation, "choices", None)
    if not raw:
        return None
    return [str(item) for item in raw]


def _range_bounds(param: Any) -> tuple[int | float | None, int | float | None]:
    annotation = getattr(param, "type", None)
    if annotation is None or "range" not in type(annotation).__name__.lower():
        return None, None
    return (
        _number_or_none(getattr(annotation, "min", None)),
        _number_or_none(getattr(annotation, "max", None)),
    )


def _path_role(name: str) -> str:
    token = name.lstrip("-").lower().replace("_", "-")
    parts = token.split("-")
    if "config" in parts:
        return "config"
    if token in {"out", "output", "out-dir", "output-dir"} or token.endswith("-out"):
        return "output_dir"
    if "out-dir" in token or token.endswith("-output"):
        return "output_dir"
    return "input"


def _default_text(value: object) -> str | None:
    if value is None or type(value).__name__ in {"Sentinel", "UNSET"}:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, ".12g")
    text = _path_text(value) if isinstance(value, os.PathLike) else _plain_text(value)
    if text is None or _nonportable(text):
        return None
    return text


def _plain_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [
            _plain_text(item) if not isinstance(item, os.PathLike) else _path_text(item)
            for item in value
        ]
        if any(part is None for part in parts):
            return None
        return ",".join(str(part) for part in parts)
    return None


def _path_text(value: os.PathLike[str] | os.PathLike[bytes]) -> str:
    raw = os.fspath(value)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return Path(raw).as_posix()


def _nonportable(text: str) -> bool:
    if "://" in text:
        return True
    normalized = text.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    return len(normalized) >= 3 and normalized[0].isalpha() and normalized[1:3] == ":/"


def _clip_help(text: object) -> str:
    raw = "" if text is None else str(text)
    stripped = raw.strip()
    if not stripped:
        return ""
    collapsed = " ".join(stripped.splitlines()[0].split())
    if len(collapsed) <= 160:
        return collapsed
    return collapsed[:157] + "..."


def _order_arg(arg: Mapping[str, Any]) -> dict[str, Any]:
    return {key: arg[key] for key in _ARG_ORDER if key in arg}


def _census_from_sets(
    *,
    required: set[str],
    order: tuple[str, ...],
    live: set[str],
    retired: Mapping[str, Mapping[str, Any]],
) -> FamilyCensus:
    ordered = tuple(name for name in order if name in required)
    ordered += tuple(sorted(required.difference(ordered)))
    live_names = tuple(sorted(live - required))
    return FamilyCensus(
        required=ordered,
        live=live_names,
        retired=_retired_rows(retired, required),
    )


def _retired_rows(
    retired: Mapping[str, Mapping[str, Any]],
    required: set[str],
) -> tuple[tuple[str, str, int], ...]:
    rows: list[tuple[str, str, int]] = []
    for name in sorted(retired):
        if name in required:
            continue
        record = retired[name]
        rows.append(
            (str(name), str(record["reason"]), _as_version(record["last_receipt_schema_version"]))
        )
    return tuple(rows)


def _as_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("last_receipt_schema_version is not an int")
    return value


def _family_rows(census: FamilyCensus, status: str) -> list[dict[str, Any]]:
    required = [_family_entry(name, "required", None, None) for name in census.required]
    live = [_family_entry(name, "live", None, None) for name in census.live]
    retired = [
        _family_entry(name, "retired", reason, version) for name, reason, version in census.retired
    ]
    if status == "required":
        return required
    if status == "live":
        return live
    if status == "retired":
        return retired
    return [*required, *live, *retired]


def _family_entry(
    ident: str, status: str, reason: str | None, version: int | None
) -> dict[str, Any]:
    return {
        "id": ident,
        "status": status,
        "retired_reason": reason,
        "last_receipt_schema_version": version,
        "runner": None,
        "executable": False,
        "executable_reason": NO_RUNNER_REASON,
    }


def _validate_family_page(status: str, limit: int, offset: int) -> None:
    if status not in _FAMILY_STATUSES:
        raise ValueError("status must be required, live, retired, or all")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    if offset < 0:
        raise ValueError("offset must be >= 0")


def _census_total(census: FamilyCensus) -> int:
    return len(census.required) + len(census.live) + len(census.retired)


def _resolve_commands(commands: Sequence[LabCommand] | None) -> tuple[LabCommand, ...]:
    if commands is None:
        return list_lab_commands()
    return tuple(sorted(commands, key=lambda item: item.id))


def _resolve_overlay(
    overlay: Sequence[Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    if overlay is None:
        return CURATED_OVERLAY
    return tuple(overlay)


def _resolve_repo(repo_root: Path | None, *, discover_repo: bool) -> Path | None:
    if repo_root is not None:
        return repo_root
    if not discover_repo:
        return None
    return find_repo_root()


def _group_overlay(
    overlay: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for entry in overlay:
        grouped.setdefault(str(entry.get("id", "")), []).append(entry)
    return grouped


def _document(
    overlay: Sequence[Mapping[str, Any]],
    n_commands: int,
    caps: Sequence[Mapping[str, Any]],
    *,
    executable_only: bool,
) -> dict[str, Any]:
    published = (
        [dict(cap) for cap in caps if cap["executable"]]
        if executable_only
        else [dict(cap) for cap in caps]
    )
    return {
        "schema": SCHEMA,
        "generated_by": {"overlay_sha256": overlay_sha256(overlay), "n_lab_commands": n_commands},
        "capabilities": published,
    }


def _cap_sort_key(cap: Mapping[str, Any]) -> tuple[str, str]:
    blob = json.dumps(cap, sort_keys=True, ensure_ascii=True, default=str)
    return str(cap["id"]), blob


def _duplicate_problems(caps: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    problems: list[str] = []
    for cap in caps:
        ident = str(cap["id"])
        if ident in seen:
            problems.append(f"duplicate capability id: {ident}")
        seen.add(ident)
    return problems


def _describe_error(matches: Sequence[Mapping[str, Any]]) -> str | None:
    if len(matches) == 1:
        return None
    if not matches:
        return "unknown capability"
    return "duplicate capability"


def _coverage_problems(doc: Mapping[str, Any], commands: Sequence[LabCommand]) -> list[str]:
    published = {str(cap["id"]) for cap in doc["capabilities"]}
    problems = [
        f"lab command missing from capabilities: {cmd.id}"
        for cmd in commands
        if cmd.id not in published
    ]
    roots = {cmd.argv[0] for cmd in commands}
    known = {cmd.id for cmd in commands}
    for cap in doc["capabilities"]:
        problems.extend(_argv_coverage(cap, roots, known))
    n_lab = doc["generated_by"]["n_lab_commands"]
    if n_lab != len(tuple(commands)):
        problems.append("n_lab_commands does not match the lab command count")
    return problems


def _argv_coverage(cap: Mapping[str, Any], roots: set[str], known: set[str]) -> list[str]:
    argv = cap["command"]["argv"]
    ident = str(cap["id"])
    problems: list[str] = []
    if not argv or argv[0] not in roots:
        problems.append(f"{ident}: command.argv[0] is not a real lab command")
    if " ".join(str(part) for part in argv) not in known:
        problems.append(f"{ident}: command argv is not a real lab command")
    return problems


def _structural_problems(doc: Mapping[str, Any]) -> list[str]:
    problems = _top_problems(doc)
    caps = doc.get("capabilities")
    if not isinstance(caps, list):
        return [*problems, "capabilities must be a list"]
    for index, cap in enumerate(caps):
        problems.extend(_one_capability_problems(cap, index))
    return problems


def _top_problems(doc: Mapping[str, Any]) -> list[str]:
    problems = _key_problems(
        doc, frozenset({"schema", "generated_by", "capabilities"}), frozenset(), "document"
    )
    if doc.get("schema") != SCHEMA:
        problems.append("schema is not dipcatcher.capabilities.v1")
    generated = doc.get("generated_by")
    problems.extend(
        _key_problems(
            generated, frozenset({"overlay_sha256", "n_lab_commands"}), frozenset(), "generated_by"
        )
    )
    if isinstance(generated, dict):
        problems.extend(_generated_problems(generated))
    return problems


def _generated_problems(generated: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    digest = generated.get("overlay_sha256")
    if not isinstance(digest, str) or _SHA_RE.fullmatch(digest) is None:
        problems.append("generated_by.overlay_sha256 is invalid")
    count = generated.get("n_lab_commands")
    if not _nonnegative_int(count):
        problems.append("generated_by.n_lab_commands is invalid")
    return problems


def _one_capability_problems(cap: object, index: int) -> list[str]:
    where = f"capabilities[{index}]"
    problems = _key_problems(cap, _CAP_KEYS, frozenset(), where)
    if not isinstance(cap, dict):
        return problems
    problems.extend(_identity_problems(cap, where))
    problems.extend(_command_block_problems(cap.get("command"), f"{where}.command"))
    problems.extend(_args_problems(cap.get("args"), f"{where}.args"))
    problems.extend(_io_problems(cap, where))
    problems.extend(_qual_block_problems(cap, where))
    return problems


def _identity_problems(cap: Mapping[str, Any], where: str) -> list[str]:
    problems = _id_problems(cap, where)
    problems.extend(_text_problems(cap.get("version"), f"{where}.version"))
    problems.extend(_text_problems(cap.get("purpose"), f"{where}.purpose"))
    problems.extend(_task_problems(cap.get("tasks"), f"{where}.tasks"))
    problems.extend(_alias_problems(cap.get("aliases"), f"{where}.aliases"))
    problems.extend(_string_list_problems(cap.get("dependencies"), f"{where}.dependencies"))
    if cap.get("evidence_class") not in EVIDENCE_CLASSES:
        problems.append(f"{where}.evidence_class is invalid")
    problems.extend(_exec_problems(cap, where))
    return problems


def _id_problems(cap: Mapping[str, Any], where: str) -> list[str]:
    ident = cap.get("id")
    if not isinstance(ident, str) or _ID_RE.fullmatch(ident) is None:
        return [f"{where}.id is invalid"]
    command = cap.get("command")
    if not isinstance(command, dict):
        return []
    argv = command.get("argv")
    if isinstance(argv, list) and ident != " ".join(str(part) for part in argv):
        return [f"{where}.id does not match command.argv"]
    return []


def _exec_problems(cap: Mapping[str, Any], where: str) -> list[str]:
    executable = cap.get("executable")
    reason = cap.get("executable_reason")
    status = _status_of(cap)
    problems: list[str] = []
    if not isinstance(executable, bool):
        problems.append(f"{where}.executable must be bool")
    if not isinstance(reason, str):
        return [*problems, f"{where}.executable_reason must be str"]
    if executable is True and reason != "":
        problems.append(f"{where}.executable_reason must be empty when executable")
    if executable is False and not reason:
        problems.append(f"{where}.executable_reason must be set when not executable")
    if executable is True and status != "qualified":
        problems.append(f"{where}: executable with non-qualified status")
    if status == "retired" and executable is True:
        problems.append(f"{where}: retired capability is executable")
    return problems


def _status_of(cap: Mapping[str, Any]) -> object:
    qual = cap.get("qualification")
    if not isinstance(qual, dict):
        return None
    return qual.get("status")


def _command_block_problems(command: object, where: str) -> list[str]:
    problems = _key_problems(command, _COMMAND_KEYS, frozenset(), where)
    if not isinstance(command, dict):
        return problems
    problems.extend(_argv_problems(command.get("argv"), f"{where}.argv"))
    problems.extend(_text_problems(command.get("role"), f"{where}.role"))
    if not isinstance(command.get("read_only"), bool):
        problems.append(f"{where}.read_only must be bool")
    problems.extend(_string_list_problems(command.get("writes"), f"{where}.writes"))
    if not isinstance(command.get("consequential"), bool):
        problems.append(f"{where}.consequential must be bool")
    if not _optional_timeout(command.get("timeout_s")):
        problems.append(f"{where}.timeout_s is invalid")
    return problems


def _argv_problems(argv: object, where: str) -> list[str]:
    if not isinstance(argv, list) or not argv:
        return [f"{where} must be a non-empty list"]
    if all(isinstance(part, str) and _TOKEN_RE.fullmatch(part) for part in argv):
        return []
    return [f"{where} has an invalid token"]


def _args_problems(args: object, where: str) -> list[str]:
    if not isinstance(args, list):
        return [f"{where} must be a list"]
    problems: list[str] = []
    for index, arg in enumerate(args):
        problems.extend(_one_arg_problems(arg, f"{where}[{index}]"))
    return problems


def _one_arg_problems(arg: object, where: str) -> list[str]:
    problems = _key_problems(arg, _ARG_REQUIRED, _ARG_OPTIONAL, where)
    if not isinstance(arg, dict):
        return problems
    problems.extend(_text_problems(arg.get("name"), f"{where}.name", allow_empty=False))
    if arg.get("kind") not in _ARG_KINDS:
        problems.append(f"{where}.kind is invalid")
    if arg.get("type") not in _ARG_TYPES:
        problems.append(f"{where}.type is invalid")
    if not isinstance(arg.get("required"), bool):
        problems.append(f"{where}.required must be bool")
    if not isinstance(arg.get("repeatable"), bool):
        problems.append(f"{where}.repeatable must be bool")
    if not isinstance(arg.get("help"), str):
        problems.append(f"{where}.help must be str")
    problems.extend(_default_problems(arg.get("default"), where))
    problems.extend(_choice_problems(arg.get("choices"), where))
    problems.extend(_bound_key_problems(arg, where))
    problems.extend(_path_role_problems(arg, where))
    problems.extend(_item_type_problems(arg, where))
    return problems


def _default_problems(default: object, where: str) -> list[str]:
    if default is None:
        return []
    if not isinstance(default, str):
        return [f"{where}.default must be a string or null"]
    if _nonportable(default):
        return [f"{where}.default is not portable"]
    return []


def _choice_problems(choices: object, where: str) -> list[str]:
    if choices is None:
        return []
    return _string_list_problems(choices, f"{where}.choices")


def _bound_key_problems(arg: Mapping[str, Any], where: str) -> list[str]:
    problems: list[str] = []
    for key in ("min", "max"):
        if key in arg and not _is_number(arg[key]):
            problems.append(f"{where}.{key} must be a number")
    return problems


def _path_role_problems(arg: Mapping[str, Any], where: str) -> list[str]:
    if "path_role" not in arg:
        if arg.get("type") == "path":
            return [f"{where}.path_role is required for path args"]
        return []
    if arg.get("path_role") not in _PATH_ROLES:
        return [f"{where}.path_role is invalid"]
    if arg.get("type") != "path":
        return [f"{where}.path_role is only valid for path args"]
    return []


def _item_type_problems(arg: Mapping[str, Any], where: str) -> list[str]:
    if "item_type" not in arg:
        return []
    if arg.get("item_type") not in _ITEM_TYPES:
        return [f"{where}.item_type is invalid"]
    if arg.get("type") != "csv":
        return [f"{where}.item_type is only valid for csv args"]
    return []


def _io_problems(cap: Mapping[str, Any], where: str) -> list[str]:
    problems = _input_problems(cap.get("inputs"), f"{where}.inputs")
    problems.extend(_output_problems(cap.get("outputs"), f"{where}.outputs"))
    problems.extend(_limit_problems(cap.get("limits"), f"{where}.limits"))
    problems.extend(_post_problems(cap.get("post_checks"), f"{where}.post_checks"))
    return problems


def _input_problems(inputs: object, where: str) -> list[str]:
    problems = _key_problems(inputs, _INPUT_KEYS, frozenset(), where)
    if not isinstance(inputs, dict):
        return problems
    problems.extend(_string_list_problems(inputs.get("data_labels"), f"{where}.data_labels"))
    if not isinstance(inputs.get("requires_dataset"), bool):
        problems.append(f"{where}.requires_dataset must be bool")
    problems.extend(
        _text_problems(inputs.get("availability_timing"), f"{where}.availability_timing")
    )
    return problems


def _output_problems(outputs: object, where: str) -> list[str]:
    problems = _key_problems(outputs, _OUTPUT_KEYS, frozenset(), where)
    if not isinstance(outputs, dict):
        return problems
    problems.extend(_string_list_problems(outputs.get("scores"), f"{where}.scores"))
    problems.extend(
        _text_problems(outputs.get("receipt_schema"), f"{where}.receipt_schema", allow_empty=True)
    )
    problems.extend(_string_list_problems(outputs.get("receipt_flag"), f"{where}.receipt_flag"))
    marker = outputs.get("receipt_stdout_marker")
    if not isinstance(marker, str):
        problems.append(f"{where}.receipt_stdout_marker must be str")
    return problems


def _limit_problems(limits: object, where: str) -> list[str]:
    problems = _key_problems(limits, _LIMIT_KEYS, frozenset(), where)
    if not isinstance(limits, dict):
        return problems
    if not _optional_timeout(limits.get("max_runtime_s")):
        problems.append(f"{where}.max_runtime_s is invalid")
    if not _optional_timeout(limits.get("max_output_chars")):
        problems.append(f"{where}.max_output_chars is invalid")
    return problems


def _post_problems(checks: object, where: str) -> list[str]:
    if not isinstance(checks, list):
        return [f"{where} must be a list"]
    problems: list[str] = []
    for index, item in enumerate(checks):
        problems.extend(_one_post_problems(item, f"{where}[{index}]"))
    return problems


def _one_post_problems(item: object, where: str) -> list[str]:
    problems = _key_problems(item, _POST_REQUIRED, _POST_OPTIONAL, where)
    if not isinstance(item, dict):
        return problems
    if not isinstance(item.get("kind"), str) or not str(item.get("kind")).strip():
        problems.append(f"{where}.kind must be a non-empty string")
    if "from" in item and not isinstance(item.get("from"), str):
        problems.append(f"{where}.from must be a string")
    return problems


def _qual_block_problems(cap: Mapping[str, Any], where: str) -> list[str]:
    qual = cap.get("qualification")
    problems = _key_problems(qual, _QUAL_KEYS, frozenset(), f"{where}.qualification")
    if not isinstance(qual, dict):
        return problems
    if qual.get("status") not in QUALIFICATION_STATUSES:
        problems.append(f"{where}.qualification.status is invalid")
    problems.extend(_string_list_problems(qual.get("evidence"), f"{where}.qualification.evidence"))
    problems.extend(_text_problems(qual.get("ruleset"), f"{where}.qualification.ruleset"))
    audited = qual.get("audited_on")
    if not isinstance(audited, str) or _DATE_RE.fullmatch(audited) is None:
        problems.append(f"{where}.qualification.audited_on is invalid")
    return problems


def _family_doc_problems(doc: Mapping[str, Any], census: FamilyCensus) -> list[str]:
    problems = _family_header_problems(doc, census)
    families = doc.get("families")
    if not isinstance(families, list):
        return [*problems, "families must be a list"]
    seen: set[str] = set()
    for entry in families:
        problems.extend(_one_family_problems(entry, seen))
    problems.extend(_family_order_problems(families, census))
    return problems


def _family_header_problems(doc: Mapping[str, Any], census: FamilyCensus) -> list[str]:
    problems: list[str] = []
    if doc.get("schema") != SCHEMA:
        problems.append("family schema is invalid")
    if doc.get("view") != "families":
        problems.append("family view is invalid")
    if doc.get("status") != "all":
        problems.append("family audit status must be all")
    counts = doc.get("counts")
    if not isinstance(counts, dict):
        return [*problems, "family counts must be an object"]
    if counts.get("required") != len(census.required):
        problems.append("required family count drifted")
    if counts.get("live") != len(census.live):
        problems.append("live family count drifted")
    if counts.get("retired") != len(census.retired):
        problems.append("retired family count drifted")
    if doc.get("total") != _census_total(census):
        problems.append("family total drifted")
    return problems


def _one_family_problems(entry: object, seen: set[str]) -> list[str]:
    if not isinstance(entry, dict):
        return ["family entry is not an object"]
    ident = str(entry.get("id", ""))
    problems: list[str] = []
    if ident in seen:
        problems.append(f"duplicate family id: {ident}")
    seen.add(ident)
    if entry.get("runner") is not None:
        problems.append(f"{ident}: family runner must be null")
    if entry.get("executable") is not False:
        problems.append(f"{ident}: family is executable")
    if entry.get("executable_reason") != NO_RUNNER_REASON:
        problems.append(f"{ident}: family executable_reason drifted")
    problems.extend(_retired_field_problems(ident, entry))
    return problems


def _retired_field_problems(ident: str, entry: Mapping[str, Any]) -> list[str]:
    status = entry.get("status")
    if status not in {"required", "live", "retired"}:
        return [f"{ident}: family status is invalid"]
    reason = entry.get("retired_reason")
    version = entry.get("last_receipt_schema_version")
    if status != "retired":
        if reason is not None or version is not None:
            return [f"{ident}: non-retired family carries retirement metadata"]
        return []
    return _retired_ok(ident, reason, version)


def _retired_ok(ident: str, reason: object, version: object) -> list[str]:
    problems: list[str] = []
    if not isinstance(reason, str) or not reason.strip():
        problems.append(f"{ident}: retired family is missing a reason")
    if isinstance(version, bool) or not isinstance(version, int):
        problems.append(f"{ident}: retired family is missing last_receipt_schema_version")
    return problems


def _family_order_problems(families: Sequence[object], census: FamilyCensus) -> list[str]:
    required = [
        entry["id"]
        for entry in families
        if isinstance(entry, dict) and entry.get("status") == "required"
    ]
    live = [
        entry["id"]
        for entry in families
        if isinstance(entry, dict) and entry.get("status") == "live"
    ]
    retired = [
        entry["id"]
        for entry in families
        if isinstance(entry, dict) and entry.get("status") == "retired"
    ]
    problems: list[str] = []
    if required != list(census.required):
        problems.append("required families are not in catalog order")
    if live != sorted(live):
        problems.append("live families are not sorted")
    if retired != sorted(retired):
        problems.append("retired families are not sorted")
    return problems


def _scan_text(text: str, where: str) -> list[str]:
    problems: list[str] = []
    if _HOSTPATH.search(text):
        problems.append(f"{where}: output contains an absolute path or URL")
    if _SECRET.search(text):
        problems.append(f"{where}: output contains a secret-like token")
    if _FQDN.search(text):
        problems.append(f"{where}: output contains a hostname")
    return problems


def _key_problems(
    obj: object, required: frozenset[str], optional: frozenset[str], where: str
) -> list[str]:
    if not isinstance(obj, dict):
        return [f"{where} must be an object"]
    allowed = required | optional
    missing = sorted(required - set(obj))
    extra = sorted(set(obj) - allowed)
    problems = [f"{where} missing {key}" for key in missing]
    problems.extend(f"{where} unknown key {key}" for key in extra)
    return problems


def _text_problems(value: object, where: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, str):
        return [f"{where} must be str"]
    if not allow_empty and not value.strip():
        return [f"{where} must be non-empty"]
    return []


def _string_list_problems(value: object, where: str) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return []
    return [f"{where} must be a list of strings"]


def _task_problems(tasks: object, where: str) -> list[str]:
    problems = _string_list_problems(tasks, where)
    if problems or not isinstance(tasks, list):
        return problems
    return [
        f"{where} task is not snake_case: {item}"
        for item in tasks
        if not isinstance(item, str) or _TASK_RE.fullmatch(item) is None
    ]


def _alias_problems(aliases: object, where: str) -> list[str]:
    problems = _string_list_problems(aliases, where)
    if problems or not isinstance(aliases, list):
        return problems
    return [
        f"{where} alias must be lowercase: {item}"
        for item in aliases
        if not isinstance(item, str) or item != item.lower() or not item.strip()
    ]


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _optional_timeout(value: object) -> bool:
    if value is None:
        return True
    return _nonnegative_int(value)


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _number_or_none(value: object) -> int | float | None:
    if not _is_number(value):
        return None
    return _as_number(value)


def _as_number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a number")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _fit(text: str, width: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= width:
        return collapsed
    if width <= 3:
        return collapsed[:width]
    return collapsed[: width - 3] + "..."


def _cells(*parts: tuple[str, int]) -> str:
    return " ".join(_fit(text, width).ljust(width) for text, width in parts).rstrip()


def _list_lines(doc: Mapping[str, Any]) -> list[str]:
    lines = [_cells(("id", 32), ("status", 13), ("exec", 4), ("purpose", 48))]
    for cap in doc["capabilities"]:
        flag = "yes" if cap["executable"] else "no"
        lines.append(
            _cells(
                (str(cap["id"]), 32),
                (str(cap["qualification"]["status"]), 13),
                (flag, 4),
                (str(cap["purpose"]), 48),
            )
        )
    return lines


def _family_lines(doc: Mapping[str, Any]) -> list[str]:
    counts = doc["counts"]
    nxt = "null" if doc["next_offset"] is None else str(doc["next_offset"])
    lines = [
        _fit(
            f"counts required={counts['required']} live={counts['live']} retired={counts['retired']}",
            100,
        ),
        _fit(
            f"status={doc['status']} total={doc['total']} offset={doc['offset']} "
            f"limit={doc['limit']} next_offset={nxt}",
            100,
        ),
        _cells(("id", 36), ("status", 10), ("exec", 5), ("reason", 46)),
    ]
    for entry in doc["families"]:
        flag = "yes" if entry["executable"] else "no"
        lines.append(
            _cells(
                (str(entry["id"]), 36),
                (str(entry["status"]), 10),
                (flag, 5),
                (str(entry["executable_reason"]), 46),
            )
        )
    return lines


def _describe_lines(doc: Mapping[str, Any]) -> list[str]:
    cap = doc.get("capability")
    if not isinstance(cap, dict):
        return [
            _fit(f"error: {doc.get('error', 'unknown capability')} id={doc.get('id', '')}", 100)
        ]
    lines = [
        _fit(f"id: {cap['id']}", 100),
        _fit(
            f"status: {cap['qualification']['status']} executable: {str(cap['executable']).lower()}",
            100,
        ),
        _fit(f"purpose: {cap['purpose']}", 100),
        _fit(f"tasks: {', '.join(cap['tasks'])}", 100),
        _fit(f"class: {cap['evidence_class']}", 100),
        _fit(f"argv: {' '.join(cap['command']['argv'])}", 100),
    ]
    if cap["executable_reason"]:
        lines.append(_fit(f"reason: {cap['executable_reason']}", 100))
    for arg in cap["args"]:
        required = str(arg["required"]).lower()
        lines.append(_fit(f"arg: {arg['name']} {arg['type']} required={required}", 100))
    return lines


def _check_lines(doc: Mapping[str, Any]) -> list[str]:
    if doc["ok"]:
        return ["ok"]
    return ["problems:", *(_fit(str(item), 100) for item in doc["problems"])]


def _emit(doc: Mapping[str, Any], *, as_json: bool, lines: list[str]) -> None:
    if as_json:
        typer.echo(dumps_document(doc), nl=False)
        return
    for line in lines:
        typer.echo(line)


@capabilities_app.command("list")
def capabilities_list(
    as_json: bool = typer.Option(False, "--json", help="Print one JSON document."),
    executable_only: bool = typer.Option(
        False, "--executable-only", help="Qualified executable capabilities only."
    ),
) -> None:
    """List lab capabilities. Unqualified commands are present and not executable."""
    doc = build_capability_document(executable_only=executable_only)
    _emit(doc, as_json=as_json, lines=_list_lines(doc))


@capabilities_app.command("describe")
def capabilities_describe(
    capability_id: str = typer.Argument(..., help="Capability id. Quote ids that contain spaces."),
    as_json: bool = typer.Option(False, "--json", help="Print one JSON document."),
) -> None:
    """Print one capability contract. Unknown ids fail closed."""
    doc = describe_capability(capability_id)
    _emit(doc, as_json=as_json, lines=_describe_lines(doc))
    if doc.get("error"):
        raise typer.Exit(code=1)


@capabilities_app.command("families")
def capabilities_families(
    as_json: bool = typer.Option(False, "--json", help="Print one JSON document."),
    status: str = typer.Option("all", "--status", help="required, live, retired, or all."),
    limit: int = typer.Option(50, "--limit"),
    offset: int = typer.Option(0, "--offset"),
) -> None:
    """List scorecard families. None of them is a standalone runner."""
    try:
        doc = build_family_document(status=status, limit=limit, offset=offset)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit(doc, as_json=as_json, lines=_family_lines(doc))


@capabilities_app.command("check")
def capabilities_check(
    as_json: bool = typer.Option(False, "--json", help="Print one JSON document."),
) -> None:
    """Audit the capability contract. Exit 1 when any problem is found."""
    problems = audit_capabilities(include_families=True)
    doc = {"schema": SCHEMA, "view": "check", "ok": not problems, "problems": problems}
    _emit(doc, as_json=as_json, lines=_check_lines(doc))
    if problems:
        raise typer.Exit(code=1)


__all__ = [
    "FAIL_CLOSED_REASON",
    "FamilyCensus",
    "LabCommand",
    "NO_RUNNER_REASON",
    "SCHEMA",
    "audit_capabilities",
    "audit_family_contract",
    "build_capability_document",
    "build_family_document",
    "capabilities_app",
    "commands_from_app",
    "describe_capability",
    "dumps_document",
    "find_repo_root",
    "list_lab_commands",
    "load_family_census",
]
