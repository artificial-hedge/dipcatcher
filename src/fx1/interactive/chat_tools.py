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


def execute_tool_call(call: Mapping[str, Any], env: ToolEnv) -> str:
    """Run one tool call. Policy violations return the code and nothing else."""
    name, arguments = parse_tool_call(call)
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"unknown tool {name!r}"
    try:
        return handler(arguments, env)
    except PolicyViolation as exc:
        _emit_policy(env, exc.code)
        return exc.code


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


def run_registered_harness(name: str, extra_args: Sequence[str], env: ToolEnv) -> str:
    """Policy, approval, then run. The return value is text for the model."""
    try:
        argv = _checked_argv(name, extra_args, env)
    except PolicyViolation as exc:
        _emit_policy(env, exc.code)
        env.say(exc.code)
        return exc.code
    if not _approved(name, env):
        env.say(f"[harness] {name}: not approved — skipped")
        return "not approved"
    result = _invoke(name, argv, env)
    return _present_harness(name, result, env)


def present_untrusted(kind: str, source: str, raw: str, say: Callable[[str], None]) -> str:
    """Scan *raw*, tell the operator, and return a scrubbed envelope."""
    raw = scrub_model_text(raw)
    notice = security_notice(scan_injection(raw))
    if notice:
        say(dim_text(notice))
    return wrap_untrusted(kind, source, scrub_transcript(raw))


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


def _approved(name: str, env: ToolEnv) -> bool:
    if not _consequential(name):
        return True
    question = f"run harness command {name!r} (consequential)?"
    return bool(env.approvals.ask(question, kind="consequential", key=f"harness:{name}"))


def _invoke(name: str, argv: list[str], env: ToolEnv) -> Any:
    from fx1.harness import Harness

    harness = env.harness if env.harness is not None else Harness()
    if not _accepts_extended(harness):
        return harness.run(name, argv)
    extra = {"DIPCATCHER_SESSION_ID": env.session_id, "DIPCATCHER_RUN_ID": new_run_id()}
    return harness.run(name, argv, cancel_event=env.cancel_event, env_extra=extra)


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


def _present_harness(name: str, result: Any, env: ToolEnv) -> str:
    stdout = scrub_model_text(scrub_transcript(str(getattr(result, "stdout", "") or "")))
    stderr = scrub_model_text(scrub_transcript(str(getattr(result, "stderr", "") or "")))
    if stdout:
        env.say(stdout.rstrip()[-4000:])
    if stderr:
        env.say_error(RuntimeError(stderr.strip()[-4000:]))
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
