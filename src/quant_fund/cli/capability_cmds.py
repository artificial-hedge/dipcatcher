"""Read-only ``dipcatcher capabilities`` contract.

The research catalog is imported only when a family view or selective
``--only`` choices are built. Command introspection loads ``quant_fund.cli.main``
on first use, not at import. Published defaults drop absolute paths and URL
hostnames. ``executable`` is true only for overlay status ``qualified`` whose
command, arg hints, and evidence files all resolve. A required family is
executable only when the selective surface has a runner for it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from quant_fund.cli.capability_audit import (
    _bind_field_problems,
    _coverage_problems,
    _scan_text,
    _structural_problems,
)
from quant_fund.cli.capability_families import (
    FamilyCensus,
    _census_from_sets,
    _census_total,
    _family_doc_problems,
    _family_rows,
    _validate_family_page,
    apply_required_hint,
    apply_status_note,
    bind_selective_arg,
    command_id_binds,
    overlay_bind_id,
    published_scores,
    selective_contract_problems,
)
from quant_fund.cli.capability_introspect import _collect
from quant_fund.cli.capability_model import (
    _ID_RE,
    LabCommand,
    _as_number,
    _is_number,
    _nonportable,
    _optional_int,
    _order_arg,
    _positive_int,
    dumps_document,
    find_repo_root,
)
from quant_fund.cli.capability_overlay import (
    AUDITED_ON,
    CURATED_OVERLAY,
    EVIDENCE_CLASSES,
    FAIL_CLOSED_REASON,
    QUALIFICATION_STATUSES,
    RESEARCH_ONLY_ID,
    RULESET,
    SCHEMA,
    overlay_sha256,
    qualified_ids,
)

_CONSEQUENTIAL_TOKENS = frozenset({"train", "optimize", "paper"})
_WEAKENING_ARGS = frozenset({"--check-index", "--write", "--claim-live", "--dev"})
_HINT_KEYS = frozenset(
    {
        "type",
        "item_type",
        "min",
        "max",
        "choices",
        "default",
        "path_role",
        "bind",
        "goal_terms",
        "required",
    }
)
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
_STATUS_REASON = {
    "retired": "retired capability is not executable",
    "experimental": "experimental capability is not executable",
    "unqualified": "unqualified capability is not executable",
}
_EVIDENCE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*\.py$")

capabilities_app = typer.Typer(
    help="Read-only capability and benchmark-family contract. Missing qualification fails closed.",
    no_args_is_help=True,
)


@dataclass
class _Built:
    document: dict[str, Any]
    problems: list[str]
    commands: tuple[LabCommand, ...]


def commands_from_app(app: Any) -> tuple[LabCommand, ...]:
    """Introspect a Typer app or an already-built Click group."""
    root = typer.main.get_command(app) if isinstance(app, typer.Typer) else app
    return tuple(sorted(_collect(root), key=lambda item: item.id))


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
    """Page family metadata. Executable only when required and a runner exists."""
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
        problems.extend(_selective_audit(built.document))
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
            cap, extra = _overlay_capability(entry, None, repo, root_tokens, set(known))
            caps.append(cap)
            problems.extend(extra)
    return caps


def _overlay_capability(
    entry: Mapping[str, Any],
    command: LabCommand | None,
    repo: Path | None,
    root_tokens: set[str],
    known_ids: set[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    shape = _shape_problems(entry)
    if shape:
        return _malformed_capability(entry), shape
    args, gates = _entry_gates(entry, command, repo, root_tokens, known_ids)
    status = str(entry["qualification"]["status"])
    executable, reason = _decide(status, gates, in_overlay=True)
    executable, reason = _apply_force(entry, executable, reason)
    executable, reason = apply_status_note(entry, executable, reason, _STATUS_REASON.get(status))
    cap = _from_overlay(entry, args, executable, reason)
    if executable:
        _coerce_executable_numbers(cap)
    problems = [*gates, *_force_problems(str(entry["id"]), status, executable)]
    return cap, problems


def _entry_gates(
    entry: Mapping[str, Any],
    command: LabCommand | None,
    repo: Path | None,
    root_tokens: set[str],
    known_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    problems = _class_problems(entry)
    problems.extend(_budget_problems(entry))
    if command is None:
        return [], [*problems, *_phantom_gates(entry, root_tokens, known_ids or set())]
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
    if not command_id_binds(ident, argv):
        return [f"{ident}: id does not match command.argv"]
    return []


def _malformed_capability(entry: Mapping[str, Any]) -> dict[str, Any]:
    raw = entry.get("id")
    ident = raw if isinstance(raw, str) and _ID_RE.fullmatch(raw) else "invalid"
    cap = _unqualified_capability(LabCommand((ident,), "", False, ()))
    cap["purpose"] = "Malformed qualification overlay entry."
    cap["executable_reason"] = "overlay entry is malformed"
    return cap


def _phantom_gates(
    entry: Mapping[str, Any], root_tokens: set[str], known_ids: set[str]
) -> list[str]:
    """A selective alias is real when its argv is an introspected lab command.

    ``research-only`` is not its own Typer command. It stays executable when
    ``research`` is, and the audit uses the same argv check.
    """
    ident = str(entry["id"])
    argv = [str(part) for part in entry["command"]["argv"]]
    problems: list[str] = []
    if not argv or argv[0] not in root_tokens:
        problems.append(f"{ident}: command.argv[0] is not a real lab command")
    if " ".join(argv) not in known_ids:
        problems.append(f"{ident}: command argv is not a real lab command")
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
    kept = _drop_weakening(ident, selected, problems, qualified=status == "qualified")
    problems.extend(bind_selective_arg(ident, kept))
    return kept, problems


def _selective_only(ident: str, name: str) -> bool:
    """``research-only`` publishes ``--only`` even though ``research`` does not."""
    return ident == RESEARCH_ONLY_ID and name == "--only"


def _selective_only_arg(ident: str, hint: object, problems: list[str]) -> dict[str, Any]:
    base = {
        "name": "--only",
        "kind": "option",
        "type": "str",
        "required": False,
        "repeatable": False,
        "help": "Qualified family ids to run.",
    }
    applied = hint if isinstance(hint, dict) else None
    if hint is not None and applied is None:
        problems.append(f"{ident}: arg hint --only must be an object")
    return _with_hint(base, applied, problems, f"{ident} --only")


def _unknown_hint_problems(
    ident: str, hints: object, by_name: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    if not isinstance(hints, dict):
        return []
    return [
        f"{ident}: unknown arg hint: {name}"
        for name in hints
        if name not in by_name and not _selective_only(ident, str(name))
    ]


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
            if _selective_only(ident, token):
                selected.append(_selective_only_arg(ident, hints.get(token), problems))
                continue
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
    unknown = _unknown_hint_key(hint, where)
    if unknown:
        problems.append(unknown)
        return _order_arg(arg)
    _apply_type_hint(arg, hint, problems, where)
    _apply_bound_hint(arg, hint, problems, where, "min", higher_is_tighter=True)
    _apply_bound_hint(arg, hint, problems, where, "max", higher_is_tighter=False)
    _apply_choices_hint(arg, hint, problems, where)
    _apply_default_hint(arg, hint, problems, where)
    apply_required_hint(arg, hint, problems, where)
    if "path_role" in hint:
        arg["path_role"] = str(hint["path_role"])
    if "item_type" in hint:
        arg["item_type"] = str(hint["item_type"])
    _apply_bind_hint(arg, hint, problems, where)
    return _order_arg(arg)


def _unknown_hint_key(hint: Mapping[str, Any], where: str) -> str:
    extra = sorted(key for key in hint if key not in _HINT_KEYS)
    if not extra:
        return ""
    return f"{where}: unknown arg hint key {extra[0]}"


def _apply_bind_hint(
    out: dict[str, Any], hint: Mapping[str, Any], problems: list[str], where: str
) -> None:
    if "bind" not in hint and "goal_terms" not in hint:
        return
    trial = dict(out)
    if "bind" in hint:
        trial["bind"] = hint["bind"]
    if "goal_terms" in hint:
        trial["goal_terms"] = hint["goal_terms"]
    found = _bind_field_problems(trial, where)
    if found:
        problems.extend(found)
        return
    _store_bind(out, trial)


def _store_bind(out: dict[str, Any], trial: Mapping[str, Any]) -> None:
    out["bind"] = trial["bind"]
    if "goal_terms" not in trial:
        return
    out["goal_terms"] = _ordered_goal_terms(trial["goal_terms"], trial.get("choices"))


def _ordered_goal_terms(raw: Mapping[str, Any], choices: object) -> dict[str, list[str]]:
    order = choices if isinstance(choices, list) else []
    ordered: dict[str, list[str]] = {}
    for key in order:
        token = str(key)
        if token not in raw:
            continue
        ordered[token] = [str(item) for item in raw[token]]
    return ordered


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


def _unpublished_budget() -> tuple[None, None, None]:
    """Non-executable rows publish null ceilings; the client reads those as 0."""
    return None, None, None


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
    timeout_s = _optional_int(command.get("timeout_s"))
    runtime_s = _optional_int(limits.get("max_runtime_s"))
    output_chars = _optional_int(limits.get("max_output_chars"))
    if not executable:
        timeout_s, runtime_s, output_chars = _unpublished_budget()
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
        timeout_s=timeout_s,
        consequential=bool(command["consequential"]),
        args=args,
        data_labels=[str(item) for item in inputs["data_labels"]],
        requires_dataset=bool(inputs["requires_dataset"]),
        availability=str(inputs["availability_timing"]),
        scores=published_scores(str(entry["id"]), outputs["scores"]),
        receipt_schema=str(outputs["receipt_schema"]),
        receipt_flag=[str(item) for item in outputs["receipt_flag"]],
        receipt_marker=str(outputs["receipt_stdout_marker"]),
        dependencies=[str(item) for item in entry["dependencies"]],
        post_checks=[dict(item) for item in entry["post_checks"]],
        max_runtime_s=runtime_s,
        max_output_chars=output_chars,
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


def _coerce_executable_numbers(cap: dict[str, Any]) -> None:
    """Publish real numbers for executable rows and keep defaults inside bounds.

    Click and the overlay emit ``"2"`` and ``"0.05"``. An advertised default
    outside ``min``/``max`` is clamped to the bound so the contract does not
    tell the caller to pass a value the overlay forbids.
    """
    args = cap.get("args")
    if not isinstance(args, list):
        return
    for arg in args:
        if isinstance(arg, dict):
            _coerce_number_arg(arg)


def _coerce_number_arg(arg: dict[str, Any]) -> None:
    typ = arg.get("type")
    if typ not in {"int", "float"}:
        return
    if "default" in arg:
        arg["default"] = _as_published_number(arg.get("default"), str(typ))
    for key in ("min", "max"):
        if key in arg and arg[key] is not None:
            arg[key] = _as_published_number(arg[key], str(typ))
    _clamp_numeric_default(arg)


def _as_published_number(value: object, typ: str) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _cast_number(float(value), typ)
    if isinstance(value, str):
        return _number_text(value.strip(), typ)
    return value


def _number_text(text: str, typ: str) -> object:
    try:
        parsed = int(text) if typ == "int" else float(text)
    except ValueError:
        return text
    if isinstance(parsed, float) and not math.isfinite(parsed):
        return text
    return parsed


def _cast_number(number: float, typ: str) -> int | float:
    if typ == "int" and number.is_integer():
        return int(number)
    return number


def _clamp_numeric_default(arg: dict[str, Any]) -> None:
    default = arg.get("default")
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return
    arg["default"] = _clamp_to_bounds(default, arg.get("min"), arg.get("max"), arg.get("type"))


def _clamp_to_bounds(default: int | float, lo: object, hi: object, typ: object) -> int | float:
    clamped = _apply_low(default, lo)
    clamped = _apply_high(clamped, hi)
    if typ == "int" and isinstance(clamped, float) and clamped.is_integer():
        return int(clamped)
    return clamped


def _apply_low(default: int | float, lo: object) -> int | float:
    if isinstance(lo, bool) or not isinstance(lo, (int, float)):
        return default
    if default < lo:
        return lo
    return default


def _apply_high(default: int | float, hi: object) -> int | float:
    if isinstance(hi, bool) or not isinstance(hi, (int, float)):
        return default
    if default > hi:
        return hi
    return default


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
    root = find_repo_root()
    if root is None or not (root / "src" / "quant_fund").is_dir():
        return None
    return root


def _group_overlay(
    overlay: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for entry in overlay:
        grouped.setdefault(overlay_bind_id(entry), []).append(entry)
    return grouped


def _selective_audit(cap_doc: Mapping[str, Any]) -> list[str]:
    census = load_family_census()
    limit = max(_census_total(census), 1)
    families = build_family_document(status="all", limit=limit, offset=0, census=census)
    rows = families["families"]
    if not isinstance(rows, list):
        return ["families must be a list"]
    return selective_contract_problems(cap_doc, rows)


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
    """List scorecard families. Required families with a runner use research --only."""
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
    "overlay_sha256",
    "qualified_ids",
]
