"""Chat tool schemas and policy-gated execution.

Tool text that came from memory, the web, or a harness process is wrapped
before it is returned to the model. Injection notices are shown to the
operator and never change an approval decision. Policy refusals return only
the violation code.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fx1.honesty import Fx1HonestyError, scrub_model_text, validate_fx1_output
from fx1.interactive import profiles
from fx1.interactive.model_ux import dim_text, scrub_transcript
from fx1.interactive.policy import (
    HarnessCallPolicy,
    PolicyViolation,
    check_flash_save,
    check_web_query,
    scan_injection,
    security_notice,
    wrap_untrusted,
)
from fx1.interactive.runctx import new_run_id

CHAT_TOOL_NAMES: tuple[str, ...] = (
    "flash_retrieve",
    "web_lookup",
    "flash_save",
    "superpower_run",
    "harness_run",
)
_CONSEQUENTIAL = frozenset({"train", "optimize", "paper"})
_MAX_TOOL_CALLS = 32
_MAX_ARG_CHARS = 32_768
_MAX_JSON_DEPTH = 32
_SHOW_CAP = 4_000
HarnessValidator = Callable[[str, list[str]], list[str]]


def policy_roots(cwd: Path | None = None, config_dir: Path | None = None) -> dict[str, Path]:
    """Path roots a harness call may touch.

    ``config`` is ``<cwd>/configs``. ``input`` is ``<cwd>/data``. ``output``
    is ``<cwd>/receipts``. ``runs`` is ``$FX1_CONFIG_DIR/runs``. Output roots
    are created by :func:`ensure_output_roots`, not by this function.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    config = profiles.config_dir() if config_dir is None else Path(config_dir)
    return {
        "config": root / "configs",
        "output": root / "receipts",
        "runs": config / "runs",
        "input": root / "data",
    }


def ensure_output_roots(roots: Mapping[str, Path]) -> None:
    """Create the output roots if they are missing. Read roots are left alone."""
    for key in ("output", "runs"):
        path = roots.get(key)
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)


def chat_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style tool definitions for one chat turn."""
    return [
        _function(
            "flash_retrieve",
            "Search persistent research memory (flash context) for prior findings "
            "relevant to a query. Returns findings with sources, uncertainty, and conflicts.",
            {"query": {"type": "string"}},
            ["query"],
        ),
        _function(
            "web_lookup",
            "A bounded public-web lookup: search and read up to two sources. "
            "Use for facts outside the conversation; deep research belongs to /research.",
            {"query": {"type": "string"}},
            ["query"],
        ),
        _function(
            "flash_save",
            "Persist a finding to flash context for later sessions.",
            {
                "text": {"type": "string"},
                "task": {"type": "string"},
                "uncertainty": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            ["text"],
        ),
        _function(
            "superpower_run",
            "Plan and run the qualified research families for a goal. Uses the same "
            "selector and honesty gate as /superpower. Retired, unknown, and "
            "unqualified families are refused.",
            {"goal": {"type": "string"}},
            ["goal"],
        ),
        _function(
            "harness_run",
            "Run one registered dipcatcher harness command (e.g. doctor, "
            "verify-research, research). Consequential commands require operator "
            "approval. Fail-closed on unknown names.",
            {
                "name": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
            },
            ["name"],
        ),
    ]


@dataclass
class ToolEnv:
    """Seams a tool needs. Nothing here reads ``input()`` or ``os.environ``."""

    store: Any
    researcher: Any
    harness: Any
    approvals: Any
    validator: HarnessValidator | None
    cancel_event: Any
    session_id: str
    say: Callable[[str], None]
    say_error: Callable[[Exception], None]
    run_superpower: Callable[[str], str]
    cwd: Path
    config_dir: Path
    emit: Callable[[str, Mapping[str, Any]], None] | None = None
    calls_used: int = 0
    last_failed: bool = False


def execute_tool_call(call: Mapping[str, Any], env: ToolEnv) -> str:
    """Run one tool call. Policy violations return the code and nothing else."""
    limited = _call_budget(env)
    if limited is not None:
        return limited
    try:
        return _dispatch_tool(call, env)
    except PolicyViolation as exc:
        _emit_policy(env, exc.code)
        return exc.code


def _dispatch_tool(call: Mapping[str, Any], env: ToolEnv) -> str:
    name, arguments = parse_tool_call(call)
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"unknown tool {name!r}"
    return handler(arguments, env)


def _call_budget(env: ToolEnv) -> str | None:
    if env.calls_used >= _MAX_TOOL_CALLS:
        return "too_many_calls"
    env.calls_used = env.calls_used + 1
    return None


def parse_tool_call(call: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """``(name, arguments)``. Malformed argument JSON becomes an empty object."""
    function = call.get("function", {})
    if not isinstance(function, dict):
        function = {}
    raw_name = function.get("name", "")
    name = raw_name if isinstance(raw_name, str) else ""
    raw = function.get("arguments")
    if raw is None or raw == "" or raw == b"":
        raw = "{}"
    if isinstance(raw, str) and _blob_too_large(raw):
        return name, {}
    return name, _load_arguments(raw)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(value)


def _load_arguments(raw: object) -> dict[str, Any]:
    """Object arguments, or ``{}`` when the blob is not a JSON object."""
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, (str, bytes, bytearray)):
        return {}
    try:
        parsed = json.loads(raw, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError, OverflowError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _blob_too_large(raw: str) -> bool:
    return len(raw) > _MAX_ARG_CHARS or _json_too_deep(raw)


def _json_too_deep(text: str) -> bool:
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        depth, in_string, escaped, over = _json_step(ch, depth, in_string, escaped)
        if over:
            return True
    return False


def _json_step(ch: str, depth: int, in_string: bool, escaped: bool) -> tuple[int, bool, bool, bool]:
    if in_string:
        return _json_string_step(ch, depth, escaped)
    if ch == '"':
        return depth, True, False, False
    if ch in "{[":
        nxt = depth + 1
        return nxt, False, False, nxt > _MAX_JSON_DEPTH
    if ch in "}]":
        return max(0, depth - 1), False, False, False
    return depth, False, False, False


def _json_string_step(ch: str, depth: int, escaped: bool) -> tuple[int, bool, bool, bool]:
    if escaped or ch == "\\":
        return depth, True, not escaped, False
    if ch == '"':
        return depth, False, False, False
    return depth, True, False, False


def run_registered_harness(name: str, extra_args: Sequence[str], env: ToolEnv) -> str:
    """Policy, approval, then run. The return value is text for the model."""
    env.last_failed = False
    try:
        argv = _checked_argv(name, extra_args, env)
    except PolicyViolation as exc:
        env.last_failed = True
        _emit_policy(env, exc.code)
        env.say(exc.code)
        return exc.code
    if not _approved(name, argv, env):
        env.say(f"[harness] {name}: not approved — skipped")
        return "not approved"
    result = _invoke(name, argv, env)
    env.last_failed = _command_failed(result)
    return _present_harness(name, result, env)


def present_untrusted(kind: str, source: str, raw: str, say: Callable[[str], None]) -> str:
    """Scan *raw*, tell the operator, and return a scrubbed envelope.

    The scan sees the original text. A forbidden headline must not replace
    the page before the injection notice is shown, and it must not be what
    the model receives.
    """
    notice = security_notice(scan_injection(raw))
    if notice:
        say(dim_text(notice))
    cleaned = scrub_transcript(scrub_model_text(raw))
    return wrap_untrusted(kind, source, cleaned)


def _function(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _flash_retrieve(arguments: Mapping[str, Any], env: ToolEnv) -> str:
    from fx1.flash.retrieve import context_block, release_private, retrieve

    hits = release_private(retrieve(env.store, str(arguments.get("query", "")), k=4), env.approvals)
    if not hits:
        return "no relevant flash context found"
    for hit in hits:
        env.store.mark_used(hit.entry.id)
    source = ",".join(hit.entry.id[:8] for hit in hits) or "flash"
    return present_untrusted("memory", source, context_block(hits), env.say)


def _flash_save(arguments: Mapping[str, Any], env: ToolEnv) -> str:
    try:
        text = check_flash_save(str(arguments.get("text", "")))
    except PolicyViolation as exc:
        return exc.code
    try:
        validate_fx1_output(text)
    except Fx1HonestyError:
        return "withheld by the fx-1 honesty gate"
    entry = env.store.add(
        text=scrub_transcript(text),
        task=scrub_transcript(str(arguments.get("task", ""))),
        uncertainty=arguments.get("uncertainty"),
        tags=["model-saved"],
    )
    return f"saved flash entry {entry.id}"


def _web_lookup(arguments: Mapping[str, Any], env: ToolEnv) -> str:
    query = str(arguments.get("query", ""))
    if not query.strip():
        return "empty query"
    try:
        cleaned = check_web_query(query, _private_texts(env.store))
    except PolicyViolation as exc:
        return exc.code
    return _web_report(cleaned, env)


def _superpower_run(arguments: Mapping[str, Any], env: ToolEnv) -> str:
    goal = str(arguments.get("goal", "")).strip()
    if not goal:
        return "superpower_run requires a goal"
    return env.run_superpower(goal)


def _harness_run(arguments: Mapping[str, Any], env: ToolEnv) -> str:
    raw_name = arguments.get("name", "")
    name = raw_name if isinstance(raw_name, str) else ""
    return run_registered_harness(name, _string_args(arguments.get("args")), env)


def _string_args(raw: object) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise PolicyViolation("bad_arg", "arguments must be strings")
    return [_string_arg(item) for item in raw]


def _string_arg(item: object) -> str:
    if not isinstance(item, str):
        raise PolicyViolation("bad_arg", "arguments must be strings")
    return item


def _private_texts(store: Any) -> list[str]:
    texts: list[str] = []
    for entry in store.all():
        if getattr(entry, "private", False):
            texts.append(str(getattr(entry, "text", "")))
    return texts


def _web_report(query: str, env: ToolEnv) -> str:
    from fx1.webresearch.engine import ResearchBudget

    budget = ResearchBudget(
        max_duration_s=30,
        max_searches=1,
        max_sources=2,
        max_depth=0,
        search_pause_s=0.0,
    )
    report = env.researcher.run(
        query, budget=budget, queries=[query], cancel_event=env.cancel_event
    )
    parts, urls = _web_parts(report)
    if not parts:
        return "web lookup found nothing (or the network is unavailable)"
    return present_untrusted("web", ",".join(urls), "\n\n".join(parts), env.say)


def _web_parts(report: Any) -> tuple[list[str], list[str]]:
    parts: list[str] = []
    urls: list[str] = []
    for source in report.sources:
        if source.ok:
            parts.append(f"SOURCE {source.url}\n{source.text[:2500]}")
            urls.append(str(source.url))
    return parts, urls


def _checked_argv(name: str, args: Sequence[str], env: ToolEnv) -> list[str]:
    from fx1.harness import HARNESS_REGISTRY

    roots = policy_roots(env.cwd, env.config_dir)
    ensure_output_roots(roots)
    policy = HarnessCallPolicy(
        registry_names=[command.name for command in HARNESS_REGISTRY],
        validator=env.validator,
        roots=tuple(roots.values()),
    )
    return policy.check(name, [str(item) for item in args])


def _consequential(name: str) -> bool:
    from fx1.harness import HARNESS_REGISTRY

    command = next((item for item in HARNESS_REGISTRY if item.name == name), None)
    if command is None:
        return name in _CONSEQUENTIAL
    return command.role.value == "model_training" or name in _CONSEQUENTIAL


def _approved(name: str, argv: Sequence[str], env: ToolEnv) -> bool:
    if not _consequential(name):
        return True
    return _ask_harness(name, argv, env)


def _ask_harness(name: str, argv: Sequence[str], env: ToolEnv) -> bool:
    from fx1.interactive.approvals import ActionBounds

    bounds = ActionBounds(argv=tuple(argv), writes=True)
    question = f"run harness command {name!r} (consequential)?"
    allowed = env.approvals.ask(
        question, kind="consequential", key=f"harness:{name}", bounds=bounds
    )
    return bool(allowed)


def _invoke(name: str, argv: list[str], env: ToolEnv) -> Any:
    from fx1.harness import Harness

    harness: Any = env.harness if env.harness is not None else Harness()
    if not _accepts_extended(harness):
        return harness.run(name, argv)
    extra = {"DIPCATCHER_SESSION_ID": env.session_id, "DIPCATCHER_RUN_ID": new_run_id()}
    extended: Any = harness
    return extended.run(name, argv, cancel_event=env.cancel_event, env_extra=extra)


def _accepts_extended(harness: Any) -> bool:
    run = getattr(harness, "run", None)
    if not callable(run):
        return False
    try:
        signature = inspect.signature(run)
    except (TypeError, ValueError):
        return False
    params = signature.parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return True
    return "cancel_event" in params and "env_extra" in params


def _clip_output(text: str) -> str:
    if len(text) <= _SHOW_CAP:
        return text
    extra = len(text) - _SHOW_CAP
    return f"{text[:_SHOW_CAP]}\n[truncated: {extra} chars omitted]"


def _visible_output(raw: object) -> str:
    return scrub_model_text(scrub_transcript(_clip_output(str(raw))))


def _present_harness(name: str, result: Any, env: ToolEnv) -> str:
    stdout = _visible_output(getattr(result, "stdout", "") or "")
    stderr = _visible_output(getattr(result, "stderr", "") or "")
    if stdout:
        env.say(stdout)
    if stderr:
        env.say_error(RuntimeError(stderr))
    flags = _flag_line(result)
    if flags:
        env.say(f"[harness] {name}: {flags}")
    body = stdout if stdout.strip() else "(command produced no output)"
    if flags:
        body = f"{body}\n{flags}"
    return present_untrusted("harness", name, body, env.say)


def _emit_policy(env: ToolEnv, code: str) -> None:
    if env.emit is None:
        return
    env.emit("concierge.policy", {"code": code})


def _command_failed(result: Any) -> bool:
    if bool(getattr(result, "timed_out", False)):
        return True
    code = getattr(result, "exit_code", 1)
    if isinstance(code, bool) or not isinstance(code, int):
        return True
    return code != 0


def _flag_line(result: Any) -> str:
    flags: list[str] = []
    if getattr(result, "timed_out", False):
        flags.append("timed out")
    if getattr(result, "cancelled", False):
        flags.append("cancelled")
    if getattr(result, "truncated", False):
        flags.append("truncated")
    if not flags:
        return ""
    return "harness result: " + ", ".join(flags)


_HANDLERS: dict[str, Callable[[Mapping[str, Any], ToolEnv], str]] = {
    "flash_retrieve": _flash_retrieve,
    "flash_save": _flash_save,
    "web_lookup": _web_lookup,
    "superpower_run": _superpower_run,
    "harness_run": _harness_run,
}
