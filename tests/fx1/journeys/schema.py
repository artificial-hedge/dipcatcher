"""YAML schema for one chat journey.

A journey file is a YAML list of mappings. Ids are ``<category>-NNN`` and the
category is one of the ten P1 buckets. ``in_chat`` journeys may not use
``shell``, ``edit_config``, or ``expert`` — those are outside the conversation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CATEGORIES: tuple[str, ...] = (
    "onboarding",
    "models",
    "data",
    "planning",
    "execution",
    "followup",
    "verification",
    "cancel",
    "recovery",
    "export",
)
ENDPOINT_MODES: frozenset[str] = frozenset(
    {"ok", "missing", "bad_key", "unknown_model", "rate_limited", "down", "slow"}
)
VIOLATIONS: frozenset[str] = frozenset({"forbidden_headline", "live_claim", "unlabeled_synthetic"})
DECISIONS: frozenset[str] = frozenset({"once", "session", "deny"})
STATUSES: frozenset[str] = frozenset(
    {"ok", "withheld", "error", "cancelled", "budget", "no_endpoint", "none"}
)
MODELS: tuple[str, ...] = ("fx1", "fx1-lite")
OUT_OF_CHAT_OPS: frozenset[str] = frozenset({"shell", "edit_config", "expert"})
_OPS: tuple[str, ...] = (
    "say",
    "cancel",
    "restart",
    "wait_job",
    "shell",
    "edit_config",
    "expert",
)
_ID = re.compile(rf"^({'|'.join(CATEGORIES)})-[0-9]{{3}}$")
_JOURNEY_KEYS = frozenset(
    {"id", "category", "title", "critical", "in_chat", "setup", "steps", "final"}
)
_SETUP_KEYS = frozenset(
    {
        "endpoints",
        "replies",
        "web",
        "research",
        "flash",
        "harness",
        "sessions",
        "jobs",
        "approvals",
        "accept_keys",
    }
)
_STEP_KEYS = frozenset(
    {
        *_OPS,
        "secret",
        "input",
        "expect",
        "forbid",
        "status",
        "model",
        "approvals",
        "files",
        "session",
        "evidence",
        "sent",
        "case_insensitive",
    }
)
_FINAL_KEYS = frozenset(
    {
        "expect",
        "forbid",
        "status",
        "model",
        "job",
        "restarted",
        "approvals",
        "files",
        "session",
        "evidence",
        "sent",
        "case_insensitive",
    }
)


class JourneySchemaError(ValueError):
    """The journey YAML broke the DSL. ``problems`` is one line per fault."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


@dataclass(frozen=True)
class Expectation:
    """One substring check. ``case_insensitive`` folds both sides."""

    text: str
    case_insensitive: bool = False


@dataclass(frozen=True)
class FileExpect:
    """A path under the journey root. Prefix ``config:``, ``session:``, or ``cwd:``."""

    path: str
    exists: bool = True
    mode: str = ""
    glob: str = ""
    contains: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApprovalExpect:
    """A partial match against one recorded approval."""

    key: str = ""
    approved: bool | None = None
    scope: str = ""
    kind: str = ""


@dataclass(frozen=True)
class SessionExpect:
    """Persisted-session checks. Unset fields are skipped."""

    turns_at_least: int | None = None
    model: str = ""
    messages_contain: tuple[str, ...] = ()
    messages_forbid: tuple[str, ...] = ()
    job: str = ""


@dataclass(frozen=True)
class EvidenceExpect:
    """Substring checks against the evidence report and ``last_result`` status."""

    status: str = ""
    contains: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplySpec:
    """One queued stub completion. ``kind`` is ok, tool, violation, or error."""

    kind: str
    text: str = ""
    tool_name: str = ""
    tool_args_json: str = "{}"
    delay_s: float = 0.0
    status_code: int = 200
    error_code: str = "error"
    error_message: str = "error"
    error_type: str = "server_error"
    violation: str = ""


@dataclass(frozen=True)
class WebDoc:
    """One scripted page. ``status`` in 200–399 is a successful fetch."""

    url: str
    title: str = ""
    status: int = 200
    text: str = ""


@dataclass(frozen=True)
class FlashSeed:
    """One pre-seeded flash entry."""

    text: str
    task: str = ""
    private: bool = False
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandScript:
    """Scripted harness stdout. ``stdout is None`` uses the synthetic default."""

    exit_code: int = 0
    stdout: str | None = None
    stderr: str = ""
    receipt_path: str = ""
    receipt_body: str = ""


@dataclass(frozen=True)
class SessionSeed:
    """A session file planted before the first turn."""

    model: str
    messages: tuple[tuple[str, str], ...] = ()
    job_state: str = "idle"


@dataclass(frozen=True)
class JobSeed:
    """A job-ledger row planted before startup reconcile."""

    kind: str
    goal: str
    state: str = "running"


@dataclass(frozen=True)
class ApprovalScript:
    """One scripted prompt answer. An empty ``key`` matches the next prompt."""

    decision: str
    key: str = ""


@dataclass(frozen=True)
class Setup:
    """Hermetic world for one journey. Omitted endpoints default to ``missing``."""

    endpoints: tuple[tuple[str, str], ...] = ()
    replies: tuple[ReplySpec, ...] = ()
    web: tuple[WebDoc, ...] = ()
    web_hold: bool = False
    flash: tuple[FlashSeed, ...] = ()
    harness_mode: str = "fake"
    commands: tuple[tuple[str, CommandScript], ...] = ()
    sessions: tuple[SessionSeed, ...] = ()
    jobs: tuple[JobSeed, ...] = ()
    approvals: tuple[ApprovalScript, ...] = ()
    accept_keys: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Step:
    """One user action plus the assertions that apply to its new output."""

    op: str
    text: str = ""
    secrets: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    cancel: str = ""
    resume: bool = False
    shell: tuple[str, ...] = ()
    edit_path: str = ""
    edit_text: str = ""
    expert: str = ""
    expect: tuple[Expectation, ...] = ()
    forbid: tuple[Expectation, ...] = ()
    status: str = ""
    model: str = ""
    approvals: tuple[ApprovalExpect, ...] | None = None
    files: tuple[FileExpect, ...] = ()
    session: SessionExpect | None = None
    evidence: EvidenceExpect | None = None
    sent: tuple[Expectation, ...] = ()


@dataclass(frozen=True)
class FinalState:
    """Assertions against the whole transcript and the end state."""

    expect: tuple[Expectation, ...] = ()
    forbid: tuple[Expectation, ...] = ()
    status: str = ""
    model: str = ""
    job: str = ""
    restarted: bool | None = None
    approvals: tuple[ApprovalExpect, ...] | None = None
    files: tuple[FileExpect, ...] = ()
    session: SessionExpect | None = None
    evidence: EvidenceExpect | None = None
    sent: tuple[Expectation, ...] = ()


@dataclass(frozen=True)
class Journey:
    """One frozen conversation. ``source`` is the yaml path for diagnostics."""

    id: str
    category: str
    title: str
    critical: bool
    in_chat: bool
    setup: Setup
    steps: tuple[Step, ...]
    final: FinalState = FinalState()
    source: str = ""


def load_journeys(directory: Path | None = None) -> tuple[Journey, ...]:
    """Load and validate every ``*.yaml`` file. Raise on the first collection fault set."""

    root = _directory(directory)
    problems: list[str] = []
    found: list[Journey] = []
    paths = sorted(path for path in root.glob("*.yaml") if path.is_file())
    if not paths:
        raise JourneySchemaError([f"no journey yaml files in {root}"])
    for path in paths:
        found.extend(_load_file(path, problems))
    _reject_duplicate_ids(found, problems)
    if problems:
        raise JourneySchemaError(problems)
    return tuple(sorted(found, key=lambda item: item.id))


def parse_journey(data: Mapping[str, Any], *, source: str = "<memory>") -> Journey:
    """Parse one journey mapping. ``source`` prefixes error lines."""

    _reject_unknown(data, _JOURNEY_KEYS, source)
    journey_id = _as_str(data.get("id"), f"{source}: id")
    _require_id(journey_id, source)
    category = _as_str(data.get("category"), f"{source}: category")
    _require_category(category, journey_id, source)
    title = _as_str(data.get("title"), f"{source}: title").strip()
    if not title:
        raise JourneySchemaError([f"{source}: title must be non-empty"])
    critical = _as_bool(data.get("critical"), f"{source}: critical")
    in_chat = _as_bool(data.get("in_chat"), f"{source}: in_chat")
    setup = _parse_setup(_mapping(data.get("setup", {}), f"{source}: setup"), source)
    steps = _parse_steps(data.get("steps"), source, in_chat=in_chat)
    final = _parse_final(_mapping(data.get("final", {}), f"{source}: final"), source)
    return Journey(
        id=journey_id,
        category=category,
        title=title,
        critical=critical,
        in_chat=in_chat,
        setup=setup,
        steps=steps,
        final=final,
        source=source,
    )


def _directory(directory: Path | None) -> Path:
    if directory is None:
        return Path(__file__).resolve().parent
    return directory


def _load_file(path: Path, problems: list[str]) -> list[Journey]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        problems.append(f"{path.name}: invalid YAML ({type(exc).__name__})")
        return []
    if not isinstance(loaded, list) or not loaded:
        problems.append(f"{path.name}: file must be a non-empty YAML list")
        return []
    found: list[Journey] = []
    for index, item in enumerate(loaded):
        where = f"{path.name}[{index}]"
        if not isinstance(item, Mapping):
            problems.append(f"{where}: journey must be a mapping")
            continue
        try:
            found.append(parse_journey(item, source=where))
        except JourneySchemaError as exc:
            problems.extend(exc.problems)
    return found


def _reject_duplicate_ids(found: list[Journey], problems: list[str]) -> None:
    seen: dict[str, str] = {}
    for journey in found:
        previous = seen.get(journey.id)
        if previous is not None:
            problems.append(f"duplicate id {journey.id} in {previous} and {journey.source}")
        seen[journey.id] = journey.source


def _require_id(journey_id: str, source: str) -> None:
    if _ID.fullmatch(journey_id) is None:
        raise JourneySchemaError([f"{source}: id {journey_id!r} must be <category>-NNN"])


def _require_category(category: str, journey_id: str, source: str) -> None:
    if category not in CATEGORIES:
        raise JourneySchemaError([f"{source}: unknown category {category!r}"])
    if not journey_id.startswith(category + "-"):
        raise JourneySchemaError([f"{source}: id prefix must match category {category!r}"])


def _parse_setup(data: Mapping[str, Any], source: str) -> Setup:
    where = f"{source}: setup"
    _reject_unknown(data, _SETUP_KEYS, where)
    return Setup(
        endpoints=_endpoints(data.get("endpoints", {}), where),
        replies=_replies(data.get("replies", []), where),
        web=_web(data.get("web", []), where),
        web_hold=_web_hold(data.get("research", {}), where),
        flash=_flash(data.get("flash", []), where),
        harness_mode=_harness_mode(data.get("harness", "fake"), where),
        commands=_commands(data.get("harness", "fake"), where),
        sessions=_sessions(data.get("sessions", []), where),
        jobs=_jobs(data.get("jobs", []), where),
        approvals=_approvals(data.get("approvals", []), where),
        accept_keys=_accept_keys(data.get("accept_keys", {}), where),
    )


def _endpoints(value: Any, where: str) -> tuple[tuple[str, str], ...]:
    data = _mapping(value, f"{where}.endpoints")
    rows: list[tuple[str, str]] = []
    for model, mode in data.items():
        if model not in MODELS:
            raise JourneySchemaError([f"{where}.endpoints: unknown model {model!r}"])
        text = _as_str(mode, f"{where}.endpoints.{model}")
        if text not in ENDPOINT_MODES:
            raise JourneySchemaError([f"{where}.endpoints.{model}: unknown mode {text!r}"])
        rows.append((str(model), text))
    return tuple(rows)


def _replies(value: Any, where: str) -> tuple[ReplySpec, ...]:
    rows = _list(value, f"{where}.replies")
    return tuple(_one_reply(item, f"{where}.replies[{index}]") for index, item in enumerate(rows))


def _one_reply(value: Any, where: str) -> ReplySpec:
    data = _mapping(value, where)
    allowed = {"ok", "tool_call", "violation", "error", "delay_s", "content"}
    _reject_unknown(data, allowed, where)
    delay = _as_float(data.get("delay_s", 0), f"{where}.delay_s")
    if "violation" in data:
        return _violation_reply(data, where, delay)
    if "error" in data:
        return _error_reply(data, where, delay)
    if "tool_call" in data:
        return _tool_reply(data, where, delay)
    if "ok" not in data:
        raise JourneySchemaError([f"{where}: reply needs ok, tool_call, violation, or error"])
    return ReplySpec(kind="ok", text=_as_str(data.get("ok"), f"{where}.ok"), delay_s=delay)


def _violation_reply(data: Mapping[str, Any], where: str, delay: float) -> ReplySpec:
    name = _as_str(data.get("violation"), f"{where}.violation")
    if name not in VIOLATIONS:
        raise JourneySchemaError([f"{where}: unknown violation {name!r}"])
    return ReplySpec(kind="violation", violation=name, delay_s=delay)


def _error_reply(data: Mapping[str, Any], where: str, delay: float) -> ReplySpec:
    raw = data.get("error")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return ReplySpec(kind="error", status_code=raw, delay_s=delay, error_type="server_error")
    body = _mapping(raw, f"{where}.error")
    _reject_unknown(body, {"status", "code", "message", "type"}, f"{where}.error")
    status = _as_int(body.get("status", 500), f"{where}.error.status")
    return ReplySpec(
        kind="error",
        status_code=status,
        error_code=_as_str(body.get("code", "error"), f"{where}.error.code"),
        error_message=_as_str(body.get("message", "error"), f"{where}.error.message"),
        error_type=_as_str(body.get("type", "server_error"), f"{where}.error.type"),
        delay_s=delay,
    )


def _tool_reply(data: Mapping[str, Any], where: str, delay: float) -> ReplySpec:
    tool = _mapping(data.get("tool_call"), f"{where}.tool_call")
    _reject_unknown(tool, {"name", "args"}, f"{where}.tool_call")
    name = _as_str(tool.get("name"), f"{where}.tool_call.name")
    args = tool.get("args", {})
    if not isinstance(args, Mapping):
        raise JourneySchemaError([f"{where}.tool_call.args must be a mapping"])
    text = ""
    if "ok" in data:
        text = _as_str(data.get("ok"), f"{where}.ok")
    elif "content" in data:
        text = _as_str(data.get("content"), f"{where}.content")
    encoded = json.dumps(dict(args), sort_keys=True)
    return ReplySpec(kind="tool", text=text, tool_name=name, tool_args_json=encoded, delay_s=delay)


def _web(value: Any, where: str) -> tuple[WebDoc, ...]:
    rows = _list(value, f"{where}.web")
    return tuple(_one_web(item, f"{where}.web[{index}]") for index, item in enumerate(rows))


def _one_web(value: Any, where: str) -> WebDoc:
    data = _mapping(value, where)
    _reject_unknown(data, {"url", "title", "status", "text"}, where)
    return WebDoc(
        url=_as_str(data.get("url"), f"{where}.url"),
        title=_as_str(data.get("title", ""), f"{where}.title"),
        status=_as_int(data.get("status", 200), f"{where}.status"),
        text=_as_str(data.get("text", ""), f"{where}.text"),
    )


def _web_hold(value: Any, where: str) -> bool:
    if value in ({}, None):
        return False
    data = _mapping(value, f"{where}.research")
    _reject_unknown(data, {"hold"}, f"{where}.research")
    return _as_bool(data.get("hold", False), f"{where}.research.hold")


def _flash(value: Any, where: str) -> tuple[FlashSeed, ...]:
    rows = _list(value, f"{where}.flash")
    return tuple(_one_flash(item, f"{where}.flash[{index}]") for index, item in enumerate(rows))


def _one_flash(value: Any, where: str) -> FlashSeed:
    data = _mapping(value, where)
    _reject_unknown(data, {"text", "task", "private", "tags"}, where)
    tags = _str_list(data.get("tags", []), f"{where}.tags")
    return FlashSeed(
        text=_as_str(data.get("text"), f"{where}.text"),
        task=_as_str(data.get("task", ""), f"{where}.task"),
        private=_as_bool(data.get("private", False), f"{where}.private"),
        tags=tuple(tags),
    )


def _harness_mode(value: Any, where: str) -> str:
    if isinstance(value, str):
        _require_harness_mode(value, where)
        return value
    data = _mapping(value, f"{where}.harness")
    _reject_unknown(data, {"mode", "commands"}, f"{where}.harness")
    mode = _as_str(data.get("mode", "fake"), f"{where}.harness.mode")
    _require_harness_mode(mode, where)
    return mode


def _require_harness_mode(mode: str, where: str) -> None:
    if mode not in {"fake", "real"}:
        raise JourneySchemaError([f"{where}.harness: mode must be fake or real"])


def _commands(value: Any, where: str) -> tuple[tuple[str, CommandScript], ...]:
    if isinstance(value, str) or value is None:
        return ()
    data = _mapping(value, f"{where}.harness")
    raw = data.get("commands", {})
    commands = _mapping(raw, f"{where}.harness.commands")
    rows: list[tuple[str, CommandScript]] = []
    for name, script in commands.items():
        rows.append((str(name), _one_command(script, f"{where}.harness.commands.{name}")))
    return tuple(rows)


def _one_command(value: Any, where: str) -> CommandScript:
    data = _mapping(value, where)
    _reject_unknown(data, {"exit", "stdout", "stderr", "receipt"}, where)
    stdout = _as_str(data["stdout"], f"{where}.stdout") if "stdout" in data else None
    path, body = _receipt(data.get("receipt"), where)
    return CommandScript(
        exit_code=_as_int(data.get("exit", 0), f"{where}.exit"),
        stdout=stdout,
        stderr=_as_str(data.get("stderr", ""), f"{where}.stderr"),
        receipt_path=path,
        receipt_body=body,
    )


def _relative_path(path: str, where: str) -> str:
    if not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise JourneySchemaError([f"{where}: path must stay inside the journey root"])
    return path


def _receipt(value: Any, where: str) -> tuple[str, str]:
    if value is None:
        return "", ""
    data = _mapping(value, f"{where}.receipt")
    _reject_unknown(data, {"path", "body"}, f"{where}.receipt")
    path = _as_str(data.get("path", ""), f"{where}.receipt.path")
    if path:
        path = _relative_path(path, f"{where}.receipt")
    return path, _as_str(data.get("body", ""), f"{where}.receipt.body")


def _sessions(value: Any, where: str) -> tuple[SessionSeed, ...]:
    rows = _list(value, f"{where}.sessions")
    return tuple(
        _one_session(item, f"{where}.sessions[{index}]") for index, item in enumerate(rows)
    )


def _one_session(value: Any, where: str) -> SessionSeed:
    data = _mapping(value, where)
    _reject_unknown(data, {"model", "messages", "job"}, where)
    model = _model_name(data.get("model", "fx1"), f"{where}.model")
    messages = _messages(data.get("messages", []), where)
    job_state = _job_state_of(data.get("job", {}), where)
    return SessionSeed(model=model, messages=messages, job_state=job_state)


def _messages(value: Any, where: str) -> tuple[tuple[str, str], ...]:
    rows = _list(value, f"{where}.messages")
    parsed: list[tuple[str, str]] = []
    for index, item in enumerate(rows):
        data = _mapping(item, f"{where}.messages[{index}]")
        _reject_unknown(data, {"role", "content"}, f"{where}.messages[{index}]")
        role = _as_str(data.get("role"), f"{where}.messages[{index}].role")
        content = _as_str(data.get("content"), f"{where}.messages[{index}].content")
        parsed.append((role, content))
    return tuple(parsed)


def _job_state_of(value: Any, where: str) -> str:
    if value in ({}, None):
        return "idle"
    data = _mapping(value, f"{where}.job")
    _reject_unknown(data, {"state"}, f"{where}.job")
    state = _as_str(data.get("state", "idle"), f"{where}.job.state")
    if state not in {"idle", "running", "interrupted", "finished"}:
        raise JourneySchemaError([f"{where}.job.state: unsupported state {state!r}"])
    return state


def _jobs(value: Any, where: str) -> tuple[JobSeed, ...]:
    rows = _list(value, f"{where}.jobs")
    return tuple(_one_job(item, f"{where}.jobs[{index}]") for index, item in enumerate(rows))


def _one_job(value: Any, where: str) -> JobSeed:
    data = _mapping(value, where)
    _reject_unknown(data, {"kind", "goal", "state"}, where)
    state = _as_str(data.get("state", "running"), f"{where}.state")
    if state not in {"queued", "running"}:
        raise JourneySchemaError([f"{where}.state must be queued or running"])
    return JobSeed(
        kind=_as_str(data.get("kind"), f"{where}.kind"),
        goal=_as_str(data.get("goal"), f"{where}.goal"),
        state=state,
    )


def _approvals(value: Any, where: str) -> tuple[ApprovalScript, ...]:
    rows = _list(value, f"{where}.approvals")
    return tuple(
        _one_approval(item, f"{where}.approvals[{index}]") for index, item in enumerate(rows)
    )


def _one_approval(value: Any, where: str) -> ApprovalScript:
    data = _mapping(value, where)
    _reject_unknown(data, {"decision", "key"}, where)
    decision = _as_str(data.get("decision"), f"{where}.decision")
    if decision not in DECISIONS:
        raise JourneySchemaError([f"{where}.decision must be once, session, or deny"])
    return ApprovalScript(decision=decision, key=_as_str(data.get("key", ""), f"{where}.key"))


def _accept_keys(value: Any, where: str) -> tuple[tuple[str, str], ...]:
    data = _mapping(value, f"{where}.accept_keys")
    rows: list[tuple[str, str]] = []
    for model, key in data.items():
        if model not in MODELS:
            raise JourneySchemaError([f"{where}.accept_keys: unknown model {model!r}"])
        text = _as_str(key, f"{where}.accept_keys.{model}")
        _require_secret_length(text, f"{where}.accept_keys.{model}")
        rows.append((str(model), text))
    return tuple(rows)


def _parse_steps(value: Any, source: str, *, in_chat: bool) -> tuple[Step, ...]:
    rows = _list(value, f"{source}: steps")
    if not rows:
        raise JourneySchemaError([f"{source}: steps must be a non-empty list"])
    parsed = tuple(_one_step(item, f"{source}: steps[{index}]") for index, item in enumerate(rows))
    if in_chat:
        _reject_out_of_chat(parsed, source)
    return parsed


def _reject_out_of_chat(steps: tuple[Step, ...], source: str) -> None:
    used = sorted({step.op for step in steps if step.op in OUT_OF_CHAT_OPS})
    if used:
        names = ", ".join(used)
        raise JourneySchemaError([f"{source}: in_chat journey cannot use {names}"])


def _one_step(value: Any, where: str) -> Step:
    data = _mapping(value, where)
    _reject_unknown(data, _STEP_KEYS, where)
    op = _step_op(data, where)
    default_ci = _as_bool(data.get("case_insensitive", False), f"{where}.case_insensitive")
    secrets = _secret_field(data.get("secret"), where) if "secret" in data else ()
    inputs = _text_field(data.get("input"), f"{where}.input") if "input" in data else ()
    if op != "say" and (secrets or inputs):
        raise JourneySchemaError([f"{where}: secret and input belong on a say step"])
    _require_wait(data, where, op)
    return Step(
        op=op,
        text=_say_text(data, where, op),
        secrets=secrets,
        inputs=inputs,
        cancel=_cancel_value(data, where, op),
        resume=_resume_value(data, where, op),
        shell=_shell_argv(data, where, op),
        edit_path=_edit_path(data, where, op),
        edit_text=_edit_text(data, where, op),
        expert=_expert_name(data, where, op),
        expect=_expectations(data.get("expect", []), f"{where}.expect", default_ci),
        forbid=_expectations(data.get("forbid", []), f"{where}.forbid", default_ci),
        status=_status_value(data.get("status", ""), f"{where}.status"),
        model=_optional_model(data.get("model", ""), f"{where}.model"),
        approvals=_approval_expects(data, where),
        files=_files(data.get("files", []), f"{where}.files"),
        session=_session_expect(data.get("session"), f"{where}.session"),
        evidence=_evidence_expect(data.get("evidence"), f"{where}.evidence"),
        sent=_expectations(data.get("sent", []), f"{where}.sent", default_ci),
    )


def _step_op(data: Mapping[str, Any], where: str) -> str:
    found = [name for name in _OPS if name in data]
    if len(found) != 1:
        raise JourneySchemaError([f"{where}: step needs exactly one of {', '.join(_OPS)}"])
    return found[0]


def _say_text(data: Mapping[str, Any], where: str, op: str) -> str:
    if op != "say":
        return ""
    text = _as_str(data.get("say"), f"{where}.say")
    if not text.strip():
        raise JourneySchemaError([f"{where}.say must be non-empty"])
    return text


def _cancel_value(data: Mapping[str, Any], where: str, op: str) -> str:
    if op != "cancel":
        return ""
    value = _as_str(data.get("cancel"), f"{where}.cancel")
    if value not in {"turn", "job"}:
        raise JourneySchemaError([f"{where}.cancel must be turn or job"])
    return value


def _resume_value(data: Mapping[str, Any], where: str, op: str) -> bool:
    if op != "restart":
        return False
    raw = data.get("restart")
    if isinstance(raw, bool):
        if not raw:
            raise JourneySchemaError([f"{where}.restart false is not a step"])
        return False
    body = _mapping(raw, f"{where}.restart")
    _reject_unknown(body, {"resume"}, f"{where}.restart")
    return _as_bool(body.get("resume", False), f"{where}.restart.resume")


def _shell_argv(data: Mapping[str, Any], where: str, op: str) -> tuple[str, ...]:
    if op != "shell":
        return ()
    argv = _str_list(data.get("shell"), f"{where}.shell")
    if not argv:
        raise JourneySchemaError([f"{where}.shell must be a non-empty argv list"])
    return tuple(argv)


def _edit_path(data: Mapping[str, Any], where: str, op: str) -> str:
    if op != "edit_config":
        return ""
    body = _mapping(data.get("edit_config"), f"{where}.edit_config")
    _reject_unknown(body, {"path", "text"}, f"{where}.edit_config")
    return _relative_path(_as_str(body.get("path"), f"{where}.edit_config.path"), where)


def _edit_text(data: Mapping[str, Any], where: str, op: str) -> str:
    if op != "edit_config":
        return ""
    body = _mapping(data.get("edit_config"), f"{where}.edit_config")
    return _as_str(body.get("text", ""), f"{where}.edit_config.text")


def _expert_name(data: Mapping[str, Any], where: str, op: str) -> str:
    if op != "expert":
        return ""
    name = _as_str(data.get("expert"), f"{where}.expert").strip()
    if not name:
        raise JourneySchemaError([f"{where}.expert must name a harness command"])
    return name


def _require_wait(data: Mapping[str, Any], where: str, op: str) -> None:
    if op == "wait_job" and data.get("wait_job") is not True:
        raise JourneySchemaError([f"{where}.wait_job must be true"])


def _approval_expects(data: Mapping[str, Any], where: str) -> tuple[ApprovalExpect, ...] | None:
    if "approvals" not in data:
        return None
    return _approval_expect_list(data.get("approvals"), f"{where}.approvals")


def _parse_final(data: Mapping[str, Any], source: str) -> FinalState:
    where = f"{source}: final"
    _reject_unknown(data, _FINAL_KEYS, where)
    default_ci = _as_bool(data.get("case_insensitive", False), f"{where}.case_insensitive")
    restarted = None
    if "restarted" in data:
        restarted = _as_bool(data.get("restarted"), f"{where}.restarted")
    approvals = None
    if "approvals" in data:
        approvals = _approval_expect_list(data.get("approvals"), f"{where}.approvals")
    return FinalState(
        expect=_expectations(data.get("expect", []), f"{where}.expect", default_ci),
        forbid=_expectations(data.get("forbid", []), f"{where}.forbid", default_ci),
        status=_status_value(data.get("status", ""), f"{where}.status"),
        model=_optional_model(data.get("model", ""), f"{where}.model"),
        job=_as_str(data.get("job", ""), f"{where}.job"),
        restarted=restarted,
        approvals=approvals,
        files=_files(data.get("files", []), f"{where}.files"),
        session=_session_expect(data.get("session"), f"{where}.session"),
        evidence=_evidence_expect(data.get("evidence"), f"{where}.evidence"),
        sent=_expectations(data.get("sent", []), f"{where}.sent", default_ci),
    )


def _expectations(value: Any, where: str, default_ci: bool) -> tuple[Expectation, ...]:
    rows = _list(value, where)
    return tuple(
        _one_expectation(item, f"{where}[{index}]", default_ci) for index, item in enumerate(rows)
    )


def _one_expectation(value: Any, where: str, default_ci: bool) -> Expectation:
    if isinstance(value, str):
        if not value:
            raise JourneySchemaError([f"{where}: empty substring"])
        return Expectation(value, default_ci)
    data = _mapping(value, where)
    _reject_unknown(data, {"text", "case_insensitive"}, where)
    text = _as_str(data.get("text"), f"{where}.text")
    if not text:
        raise JourneySchemaError([f"{where}.text must be non-empty"])
    if "case_insensitive" in data:
        flag = _as_bool(data.get("case_insensitive"), f"{where}.case_insensitive")
    else:
        flag = default_ci
    return Expectation(text, flag)


def _files(value: Any, where: str) -> tuple[FileExpect, ...]:
    rows = _list(value, where)
    return tuple(_one_file(item, f"{where}[{index}]") for index, item in enumerate(rows))


def _one_file(value: Any, where: str) -> FileExpect:
    data = _mapping(value, where)
    _reject_unknown(data, {"path", "exists", "mode", "glob", "contains", "forbid"}, where)
    mode = _as_str(data.get("mode", ""), f"{where}.mode")
    if mode and not re.fullmatch(r"[0-7]{3,4}", mode):
        raise JourneySchemaError([f"{where}.mode must be an octal string"])
    return FileExpect(
        path=_as_str(data.get("path"), f"{where}.path"),
        exists=_as_bool(data.get("exists", True), f"{where}.exists"),
        mode=mode,
        glob=_as_str(data.get("glob", ""), f"{where}.glob"),
        contains=tuple(_str_list(data.get("contains", []), f"{where}.contains")),
        forbid=tuple(_str_list(data.get("forbid", []), f"{where}.forbid")),
    )


def _approval_expect_list(value: Any, where: str) -> tuple[ApprovalExpect, ...]:
    rows = _list(value, where)
    return tuple(_one_approval_expect(item, f"{where}[{index}]") for index, item in enumerate(rows))


def _one_approval_expect(value: Any, where: str) -> ApprovalExpect:
    data = _mapping(value, where)
    _reject_unknown(data, {"key", "approved", "scope", "kind"}, where)
    approved = None
    if "approved" in data:
        approved = _as_bool(data.get("approved"), f"{where}.approved")
    return ApprovalExpect(
        key=_as_str(data.get("key", ""), f"{where}.key"),
        approved=approved,
        scope=_as_str(data.get("scope", ""), f"{where}.scope"),
        kind=_as_str(data.get("kind", ""), f"{where}.kind"),
    )


def _session_expect(value: Any, where: str) -> SessionExpect | None:
    if value is None:
        return None
    data = _mapping(value, where)
    _reject_unknown(
        data, {"turns_at_least", "model", "messages_contain", "messages_forbid", "job"}, where
    )
    turns = None
    if "turns_at_least" in data:
        turns = _as_int(data.get("turns_at_least"), f"{where}.turns_at_least")
    return SessionExpect(
        turns_at_least=turns,
        model=_optional_model(data.get("model", ""), f"{where}.model"),
        messages_contain=tuple(
            _str_list(data.get("messages_contain", []), f"{where}.messages_contain")
        ),
        messages_forbid=tuple(
            _str_list(data.get("messages_forbid", []), f"{where}.messages_forbid")
        ),
        job=_as_str(data.get("job", ""), f"{where}.job"),
    )


def _evidence_expect(value: Any, where: str) -> EvidenceExpect | None:
    if value is None:
        return None
    data = _mapping(value, where)
    _reject_unknown(data, {"status", "contains"}, where)
    return EvidenceExpect(
        status=_as_str(data.get("status", ""), f"{where}.status"),
        contains=tuple(_str_list(data.get("contains", []), f"{where}.contains")),
    )


def _secret_field(value: Any, where: str) -> tuple[str, ...]:
    rows = _text_field(value, f"{where}.secret")
    for item in rows:
        _require_secret_length(item, f"{where}.secret")
    return rows


def _require_secret_length(text: str, where: str) -> None:
    if len(text) >= 8 or "{" in text:
        return
    raise JourneySchemaError([f"{where}: secrets must be at least 8 characters"])


def _text_field(value: Any, where: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(_str_list(value, where))


def _str_list(value: Any, where: str) -> list[str]:
    rows = _list(value, where)
    return [_as_str(item, f"{where}[{index}]") for index, item in enumerate(rows)]


def _status_value(value: Any, where: str) -> str:
    if value in ("", None):
        return ""
    text = _as_str(value, where)
    if text not in STATUSES:
        raise JourneySchemaError([f"{where}: unknown status {text!r}"])
    return text


def _optional_model(value: Any, where: str) -> str:
    if value in ("", None):
        return ""
    return _model_name(value, where)


def _model_name(value: Any, where: str) -> str:
    text = _as_str(value, where)
    if text not in MODELS:
        raise JourneySchemaError([f"{where}: model must be fx1 or fx1-lite"])
    return text


def _reject_unknown(data: Mapping[str, Any], allowed: AbstractSet[str], where: str) -> None:
    extra = sorted(str(key) for key in data if str(key) not in allowed)
    if extra:
        names = ", ".join(extra)
        raise JourneySchemaError([f"{where}: unknown fields {names}"])


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise JourneySchemaError([f"{where} must be a mapping"])


def _list(value: Any, where: str) -> list[Any]:
    if isinstance(value, list):
        return value
    raise JourneySchemaError([f"{where} must be a list"])


def _as_str(value: Any, where: str) -> str:
    if isinstance(value, str):
        return value
    raise JourneySchemaError([f"{where} must be a string"])


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    raise JourneySchemaError([f"{where} must be a bool"])


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JourneySchemaError([f"{where} must be an int"])
    return value


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JourneySchemaError([f"{where} must be a number"])
    return float(value)
