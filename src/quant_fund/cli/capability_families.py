"""Family census, selective-surface bridge, and family-view checks.

The research registry and ``family_surface`` are imported inside the functions
that read them, not when this module is imported.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from quant_fund.cli.capability_overlay import PARTIAL_RUN_SCHEMA, RESEARCH_ONLY_ID, SCHEMA

_FAMILY_STATUSES = frozenset({"required", "live", "retired", "all"})


@dataclass(frozen=True)
class FamilyCensus:
    """Ordered required names, sorted live names, and retired rows."""

    required: tuple[str, ...]
    live: tuple[str, ...]
    retired: tuple[tuple[str, str, int], ...]


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
        **family_view_fields(ident, status),
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
    problems.extend(family_row_problems(entry))
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


def selective_choices() -> list[str]:
    """Executable family ids in catalog order. Imports the research registry."""
    from quant_fund.research.family_surface import list_executable

    return [str(row["id"]) for row in list_executable()]


def published_scores(capability_id: str, declared: object) -> list[str]:
    """Score labels for one capability. ``research-only`` follows the runners."""
    if capability_id == RESEARCH_ONLY_ID:
        return _runner_scores()
    if isinstance(declared, list):
        return [str(item) for item in declared]
    return []


def bind_selective_arg(capability_id: str, args: list[dict[str, Any]]) -> list[str]:
    """Fill ``--only`` choices from the executable runners. No-op for other ids."""
    if capability_id != RESEARCH_ONLY_ID:
        return []
    choices = selective_choices()
    problems = _missing_only(choices)
    matched = False
    for arg in args:
        if arg.get("name") != "--only":
            continue
        matched = True
        arg["choices"] = choices
        arg["type"] = "csv"
        arg["item_type"] = "str"
        arg["required"] = True
    if not matched:
        problems.append("research-only: selective capability is missing --only")
    return problems


def family_view_fields(family_id: str, status: str) -> dict[str, Any]:
    """Family-view fields from ``RUNNERS`` and ``family_contract``.

    Bulk live and retired ids that have no runner use the surface refusal
    text without a per-id catalog lookup.
    """
    from quant_fund.research.family_surface import RUNNERS, FamilySurfaceError, family_contract

    if family_id not in RUNNERS and status != "required":
        return _fields_from_status(family_id, status)
    try:
        contract = family_contract(family_id)
    except FamilySurfaceError as exc:
        row = _fields_from_status(family_id, status)
        row["executable_reason"] = str(exc)
        return row
    return _fields_from_contract(contract, status)


def _missing_only(choices: list[str]) -> list[str]:
    if choices:
        return []
    return ["research-only: --only choices are empty"]


def _runner_scores() -> list[str]:
    from quant_fund.research.family_surface import RUNNERS

    seen: list[str] = []
    for spec in RUNNERS.values():
        for score in spec.scores:
            if score not in seen:
                seen.append(score)
    return seen


def _fields_from_contract(contract: Mapping[str, Any], status: str) -> dict[str, Any]:
    executable = bool(contract.get("executable")) and status == "required"
    return {
        "runner": _runner_text(contract, executable),
        "executable": executable,
        "executable_reason": _contract_refusal(contract, status, executable),
        "purpose": str(contract.get("purpose") or ""),
        "tasks": _str_list(contract.get("tasks")),
        "scores": _str_list(contract.get("scores")),
        "resource_limit_s": _limit(contract.get("resource_limit_s")),
    }


def _fields_from_status(family_id: str, status: str) -> dict[str, Any]:
    return {
        "runner": None,
        "executable": False,
        "executable_reason": _status_refusal(family_id, status),
        "purpose": "",
        "tasks": [],
        "scores": [],
        "resource_limit_s": 0,
    }


def _runner_text(contract: Mapping[str, Any], executable: bool) -> str | None:
    if not executable:
        return None
    runner = contract.get("runner")
    if isinstance(runner, str) and runner:
        return runner
    return f"research --only {contract.get('id', '')}"


def _contract_refusal(contract: Mapping[str, Any], status: str, executable: bool) -> str:
    if executable:
        return ""
    refusal = contract.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        return refusal
    return _status_refusal(str(contract.get("id", "")), status)


def _status_refusal(family_id: str, status: str) -> str:
    if status == "retired":
        return (
            f"{family_id} is retired; new selection is refused and historical receipts still verify"
        )
    if status == "live":
        return (
            f"{family_id} is live-optional but has no selective runner; "
            "refusing to execute it from the family surface"
        )
    if status == "required":
        return f"{family_id} has no implemented selective runner"
    return f"{family_id} is not a known benchmark family"


def _str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def family_row_problems(entry: Mapping[str, Any]) -> list[str]:
    """Structural problems for one capabilities family row."""
    ident = str(entry.get("id", ""))
    problems = _family_run_problems(ident, entry)
    problems.extend(_family_meta_problems(ident, entry))
    return problems


def selective_contract_problems(
    cap_doc: Mapping[str, Any],
    family_rows: Sequence[object],
) -> list[str]:
    """Cross-check executable families, ``--only`` choices, and the two research modes."""
    executable = [
        str(row["id"])
        for row in family_rows
        if isinstance(row, dict) and row.get("executable") is True
    ]
    problems = _runners_cover(executable)
    problems.extend(_only_choice_problems(cap_doc, executable))
    problems.extend(_research_mode_problems(cap_doc))
    return problems


def _family_run_problems(ident: str, entry: Mapping[str, Any]) -> list[str]:
    if entry.get("executable") is True:
        return _executable_family_shape(ident, entry)
    return _inert_family_shape(ident, entry)


def _executable_family_shape(ident: str, entry: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if entry.get("runner") != f"research --only {ident}":
        problems.append(f"{ident}: executable family runner drifted")
    if entry.get("executable_reason") != "":
        problems.append(f"{ident}: executable family must not carry a refusal")
    if entry.get("status") != "required":
        problems.append(f"{ident}: executable family is not required")
    purpose = entry.get("purpose")
    if not isinstance(purpose, str) or not purpose.strip():
        problems.append(f"{ident}: executable family is missing a purpose")
    return problems


def _inert_family_shape(ident: str, entry: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if entry.get("executable") is not False:
        problems.append(f"{ident}: family executable flag is invalid")
    if entry.get("runner") is not None:
        problems.append(f"{ident}: non-executable family publishes a runner")
    reason = entry.get("executable_reason")
    if not isinstance(reason, str) or not reason.strip():
        problems.append(f"{ident}: non-executable family is missing a refusal")
    return problems


def _family_meta_problems(ident: str, entry: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if not isinstance(entry.get("purpose"), str):
        problems.append(f"{ident}: family purpose is invalid")
    if not isinstance(entry.get("tasks"), list):
        problems.append(f"{ident}: family tasks are invalid")
    if not isinstance(entry.get("scores"), list):
        problems.append(f"{ident}: family scores are invalid")
    limit = entry.get("resource_limit_s")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        problems.append(f"{ident}: family resource_limit_s is invalid")
    elif entry.get("executable") is True and limit <= 0:
        problems.append(f"{ident}: executable family is missing resource_limit_s")
    return problems


def _runners_cover(executable: Sequence[str]) -> list[str]:
    from quant_fund.research.family_surface import RUNNERS

    return [
        f"{ident}: executable family has no RUNNERS entry"
        for ident in executable
        if ident not in RUNNERS
    ]


def _only_choice_problems(cap_doc: Mapping[str, Any], executable: Sequence[str]) -> list[str]:
    expected = list(executable)
    problems: list[str] = []
    found = False
    for cap in _cap_list(cap_doc):
        for arg in _arg_list(cap):
            if arg.get("name") != "--only":
                continue
            found = True
            if arg.get("choices") != expected:
                problems.append(
                    f"{cap.get('id')}: --only choices do not match executable family ids"
                )
    if not found:
        problems.append("no capability publishes --only")
    return problems


def _research_mode_problems(cap_doc: Mapping[str, Any]) -> list[str]:
    by_id = {str(cap.get("id")): cap for cap in _cap_list(cap_doc)}
    partial = by_id.get(RESEARCH_ONLY_ID)
    if partial is None:
        return ["research-only capability is missing"]
    problems = _partial_mode_problems(partial)
    full = by_id.get("research")
    if full is not None:
        problems.extend(_full_mode_problems(full))
    return problems


def _partial_mode_problems(cap: Mapping[str, Any]) -> list[str]:
    outputs = cap.get("outputs")
    problems: list[str] = []
    schema = outputs.get("receipt_schema") if isinstance(outputs, dict) else None
    marker = outputs.get("receipt_stdout_marker") if isinstance(outputs, dict) else None
    if schema != PARTIAL_RUN_SCHEMA:
        problems.append("research-only: receipt_schema must be dipcatcher.family_run.v1")
    if marker != "json=":
        problems.append("research-only: receipt_stdout_marker must be json=")
    if cap.get("post_checks") != []:
        problems.append("research-only: post_checks must be empty")
    return problems


def _full_mode_problems(cap: Mapping[str, Any]) -> list[str]:
    checks = cap.get("post_checks")
    problems: list[str] = []
    if not isinstance(checks, list) or not any(
        isinstance(item, dict) and item.get("kind") == "verify_research" for item in checks
    ):
        problems.append("research: full scorecard must post-check verify-research")
    if "--only" in _arg_names(cap):
        problems.append("research: full scorecard must not publish --only")
    return problems


def _cap_list(cap_doc: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = cap_doc.get("capabilities")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _arg_list(cap: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = cap.get("args")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _arg_names(cap: Mapping[str, Any]) -> set[str]:
    return {str(arg.get("name")) for arg in _arg_list(cap)}


_MODE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def overlay_bind_id(entry: Mapping[str, Any]) -> str:
    """Lab command id an overlay entry attaches to, falling back to its own id."""
    command = entry.get("command")
    if isinstance(command, dict):
        argv = command.get("argv")
        if isinstance(argv, list) and argv:
            return " ".join(str(part) for part in argv)
    return str(entry.get("id", ""))


def command_id_binds(ident: str, argv: object) -> bool:
    """True when ``ident`` is the command or a ``<root>-<mode>`` variant of it."""
    if not isinstance(argv, list) or not argv:
        return False
    joined = " ".join(str(part) for part in argv)
    if ident == joined:
        return True
    root = str(argv[0])
    if " " in ident or not ident.startswith(root + "-"):
        return False
    return _MODE_ID.fullmatch(ident) is not None


def apply_status_note(
    entry: Mapping[str, Any],
    executable: bool,
    reason: str,
    generic: str | None,
) -> tuple[bool, str]:
    """Prefer an overlay qualification note over the generic status sentence."""
    if executable:
        return executable, reason
    text = _qualification_note(entry)
    if not text:
        return executable, reason
    if not reason or reason == generic:
        return False, text
    return False, f"{text}; {reason}"


def apply_required_hint(
    out: dict[str, Any], hint: Mapping[str, Any], problems: list[str], where: str
) -> None:
    """Allow an overlay to tighten an optional argument to required."""
    if "required" not in hint:
        return
    incoming = hint["required"]
    if not isinstance(incoming, bool):
        problems.append(f"{where}: overlay required must be bool")
        return
    if out.get("required") is True and incoming is False:
        problems.append(f"{where}: overlay loosens required")
        return
    out["required"] = incoming


def _qualification_note(entry: Mapping[str, Any]) -> str:
    qual = entry.get("qualification")
    if not isinstance(qual, dict):
        return ""
    note = qual.get("reason")
    if not isinstance(note, str):
        return ""
    return " ".join(note.split())
