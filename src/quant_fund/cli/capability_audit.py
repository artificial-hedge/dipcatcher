"""Structural checks for a published ``dipcatcher.capabilities.v1`` document."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from quant_fund.cli.capability_families import command_id_binds
from quant_fund.cli.capability_model import (
    _ID_RE,
    LabCommand,
    _is_number,
    _nonnegative_int,
    _nonportable,
    _optional_timeout,
)
from quant_fund.cli.capability_overlay import EVIDENCE_CLASSES, QUALIFICATION_STATUSES, SCHEMA

_ARG_TYPES = frozenset({"flag", "int", "float", "str", "enum", "path", "csv"})
_ARG_KINDS = frozenset({"option", "argument"})
_PATH_ROLES = frozenset({"config", "input", "output_dir", "any"})
_ITEM_TYPES = frozenset({"int", "float", "str"})

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

_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_TASK_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

_HOSTPATH = re.compile(r"/home(?:/|\"|$)|/Users/|[A-Za-z]:[\\/]|https?://")
_SECRET = re.compile(
    r"(?i)(api[_-]?key\s*[:=]|secret[_-]?key|BEGIN [A-Z0-9 ]*PRIVATE KEY|"
    r"AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})"
)
_FQDN = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|dev|ai)\b", re.I)


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
    if isinstance(argv, list) and not command_id_binds(ident, argv):
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
    if isinstance(default, str):
        return _portable_default(default, where)
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return [f"{where}.default must be a string, number, or null"]
    return []


def _portable_default(default: str, where: str) -> list[str]:
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
