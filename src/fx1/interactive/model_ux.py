"""Operator-facing model, endpoint, and usage text for the concierge.

Credential values never appear in these strings. A stored URL that fails
normalization becomes a safe error; an unset endpoint stays the existing
honest "no endpoint" sentence. ANSI is omitted when ``NO_COLOR`` is set.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fx1.interactive import profiles
from fx1.interactive.model_client import ErrorKind, ModelClient, ModelError, ModelIdentity
from fx1.interactive.redact import redact_text
from fx1.interactive.system_prompt import Mode, build_system_prompt
from fx1.interactive.usage_budget import UsageTracker
from fx1.serve.backends import SamplingParams

_SECRET_ERROR = re.compile(r"(?i)(?:\bsk-[A-Za-z0-9]+|\bapi[_-]?key\b|\bbearer\b|\bsecret\b)")
_HELLO = (
    "dipcatcher concierge — model {model}. Plain text chats; "
    "/superpower, /research, /flash, /help. Ctrl-D or /exit to leave."
)
_MODEL_COLUMNS: tuple[str, ...] = (
    "active",
    "model",
    "status",
    "host",
    "source",
    "fingerprint",
    "served",
    "latency",
)
_MODEL_WIDTHS: dict[str, int] = {
    "active": 1,
    "model": 8,
    "status": 11,
    "host": 12,
    "source": 10,
    "fingerprint": 8,
    "served": 10,
    "latency": 8,
}
_SHRINK_ORDER: tuple[str, ...] = ("host", "served", "status", "source", "fingerprint", "model")


def public_error(exc: BaseException) -> str:
    """Safe operator text. Credential-shaped messages become a fixed sentence."""
    text = redact_text(str(exc))
    if _SECRET_ERROR.search(text):
        return f"{type(exc).__name__}: command failed"
    return text


def no_endpoint_text(model: str) -> str:
    """Honest unset-endpoint sentence. No hostname is invented."""
    return (
        f"no model endpoint configured — set one with `/keys set {model}`, or export "
        "MOONSHOT_API_KEY / FX1_BASE_URL. Slash commands still work."
    )


def scrub_transcript(text: str) -> str:
    """Remove env secrets and credential shapes before text is stored or shown."""
    from fx1.interactive.sessions import scrub_secret_text

    return redact_text(scrub_secret_text(text))


def backend_supports_cancel(backend: Any) -> bool:
    """True when *backend* accepts ``cancel_event`` and ``deadline``."""
    if isinstance(backend, ModelClient):
        return True
    return bool(getattr(backend, "supports_cancel", False))


@dataclass(frozen=True)
class OpenedModel:
    """Result of constructing a chat backend. ``detail`` is safe to print."""

    backend: Any | None
    status: str
    detail: str


def open_backend(model: str, factory: Callable[..., Any]) -> OpenedModel:
    """Open *model* via *factory*. ``no_endpoint`` when no key is configured."""
    info, problem = _resolved(model)
    if problem is not None:
        return OpenedModel(None, "error", problem)
    if info is None:
        return OpenedModel(None, "no_endpoint", no_endpoint_text(model))
    return _construct(factory, model, info)


def use_lines(model: str) -> list[str]:
    """Confirmation for ``/use``. A missing endpoint warns and does not refuse."""
    lines = [f"active model {model!r}; conversation kept"]
    try:
        info = profiles.resolve_endpoint_info(model)
    except (ValueError, KeyError) as exc:
        lines.append(public_error(exc))
        return lines
    if info is None:
        lines.append(f"{model} has no endpoint configured — /keys set {model}")
    return lines


def startup_lines(model: str) -> list[str]:
    """Banner, hello, endpoint presence, and credential-store warnings."""
    from fx1.interactive.orb import banner as orb_banner
    from fx1.interactive.orb import orbs_enabled

    host, extra = _endpoint_banner(model)
    lines = [orb_banner(model, host, color=orbs_enabled()), _HELLO.format(model=model), *extra]
    lines.extend(f"warning: {item}" for item in profiles.store_health())
    return lines


def endpoint_stored_line(model: str, path: Path, describe: str) -> str:
    """Presence-only confirmation. *describe* is the completion route."""
    return f"{describe} stored for {model!r} in {path} (mode 0600)"


def render_model_error(exc: ModelError) -> str:
    """``str(exc)`` plus the next command for auth and unknown-model failures."""
    text = str(exc)
    step = _next_step(exc)
    if step and step not in text:
        return f"{text} Next: {step}"
    return text


def outbound_messages(
    messages: Sequence[Mapping[str, Any]],
    model: str,
    *,
    tools: Sequence[str] = (),
    mode: Mode = "chat",
) -> list[dict[str, Any]]:
    """One leading system prompt, then user, assistant, and tool messages.

    Every stored system message is hoisted into the prompt notes. The built
    prompt is not written back into the transcript.
    """
    prompt = build_system_prompt(model, tools=tools, mode=mode, notes=_system_notes(messages))
    return [{"role": "system", "content": prompt}, *_dialogue(messages)]


def sampling_for(tracker: UsageTracker) -> SamplingParams | None:
    """``max_tokens`` only when a token ceiling is set. May raise BudgetExceeded."""
    cap = tracker.request_max_tokens()
    if cap is None:
        return None
    return SamplingParams(max_tokens=cap)


def completion_kwargs(
    tracker: UsageTracker,
    backend: Any,
    cancel: Any,
    *,
    pass_deadline: bool,
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Keyword arguments for one completion. Cancel is passed only when supported."""
    kwargs: dict[str, Any] = {}
    sampling = sampling_for(tracker)
    if sampling is not None:
        kwargs["sampling"] = sampling
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    if not backend_supports_cancel(backend):
        return kwargs
    _add_cancel(kwargs, tracker, cancel, pass_deadline=pass_deadline)
    return kwargs


@dataclass(frozen=True)
class CallSnapshot:
    """Usage and identity observed after one successful model call."""

    usage: dict[str, int] | None
    identity: ModelIdentity | None
    latency_s: float


def read_snapshot(backend: Any, started: float, clock: Callable[[], float]) -> CallSnapshot:
    """Copy usage and identity off *backend*. Missing fields stay None."""
    identity = getattr(backend, "last_identity", None)
    if not isinstance(identity, ModelIdentity):
        identity = None
    return CallSnapshot(
        usage=_usage_dict(getattr(backend, "last_usage", None)),
        identity=identity,
        latency_s=max(0.0, float(clock()) - float(started)),
    )


def charge_reported_usage(tracker: UsageTracker, backend: Any) -> None:
    """Charge whatever usage the backend reported. ``None`` is unmetered."""
    tracker.charge_usage(_usage_dict(getattr(backend, "last_usage", None)))


def footer_lines(
    model: str,
    latency_s: float | None,
    usage: Mapping[str, int] | None,
    identity: ModelIdentity | None,
) -> list[str]:
    """Dim footer. The requested model id is never replaced by the served id."""
    parts = [model, _latency_phrase(latency_s), _token_phrase(usage)]
    if identity is not None and identity.served and identity.served != model:
        parts.append(f"served {identity.served}")
    lines = ["· " + " · ".join(parts)]
    if identity is not None and identity.matched is False:
        served = identity.served or "unreported"
        lines.append(f"warning: served model {served} differs from requested model {model}")
    return [dim_text(line) for line in lines]


def render_models_table(active: str, stats: Mapping[str, Mapping[str, Any]]) -> str:
    """Width-safe presence table for both hosted models."""
    widths = _fit_widths(_terminal_width())
    header = {column: column for column in _MODEL_COLUMNS}
    rule = {column: "-" * widths[column] for column in _MODEL_COLUMNS}
    body = [_model_row(active, item, stats) for item in profiles.list_models()]
    lines = [_row_line(header, widths), _row_line(rule, widths)]
    lines.extend(_row_line(row, widths) for row in body)
    return "\n".join(lines)


def render_approvals(rows: Sequence[Mapping[str, Any]]) -> str:
    """One line per recorded decision. Questions are already redacted."""
    if not rows:
        return "no approvals recorded"
    return "\n".join(_approval_line(row) for row in rows)


def dim_text(text: str) -> str:
    """Dim *text* when the terminal allows color. ``NO_COLOR`` forces plain text."""
    if "NO_COLOR" in os.environ:
        return text
    from fx1.interactive.terminal import Style, detect_caps

    caps = detect_caps()
    if not caps.color:
        return text
    return Style(caps).dim(text)


def _resolved(model: str) -> tuple[Any | None, str | None]:
    try:
        return profiles.resolve_endpoint_info(model), None
    except (ValueError, KeyError) as exc:
        return None, public_error(exc)


def _construct(factory: Callable[..., Any], model: str, info: Any) -> OpenedModel:
    try:
        backend = factory(api_key=info.api_key, model=model, api_url=info.completion_url)
    except Exception as exc:  # noqa: BLE001 — a bad factory must not kill the shell
        return OpenedModel(None, "error", public_error(exc))
    return OpenedModel(backend, "ok", "")


def _endpoint_banner(model: str) -> tuple[str, list[str]]:
    try:
        info = profiles.resolve_endpoint_info(model)
    except (ValueError, KeyError) as exc:
        return "unconfigured", [public_error(exc)]
    if info is None:
        return "unconfigured", [
            f"No endpoint configured. Run /keys set {model} to add your API key and URL."
        ]
    return info.host, [f"endpoint source {info.source} fingerprint {info.fingerprint}"]


def _next_step(exc: ModelError) -> str:
    if exc.kind is ErrorKind.AUTH:
        return f"/keys set {exc.model}"
    if exc.kind is ErrorKind.UNKNOWN_MODEL:
        return "/models"
    return ""


def _system_notes(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    notes: list[str] = []
    for message in messages:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            notes.append(content)
    return notes


def _dialogue(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    body: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            continue
        body.append(dict(message))
    return body


def _add_cancel(
    kwargs: dict[str, Any],
    tracker: UsageTracker,
    cancel: Any,
    *,
    pass_deadline: bool,
) -> None:
    if cancel is not None:
        kwargs["cancel_event"] = cancel
    if not pass_deadline:
        return
    deadline = tracker.deadline()
    if deadline is not None:
        kwargs["deadline"] = deadline


def _usage_dict(usage: Any) -> dict[str, int] | None:
    if not isinstance(usage, Mapping) or not usage:
        return None
    copied: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool):
            copied[key] = value
    return copied or None


def _token_phrase(usage: Mapping[str, int] | None) -> str:
    if usage is None:
        return "tokens unreported"
    total = usage.get("total_tokens")
    if isinstance(total, int) and not isinstance(total, bool):
        return f"{total} tokens"
    return _token_sum(usage)


def _token_sum(usage: Mapping[str, int]) -> str:
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return f"{prompt + completion} tokens"
    return "tokens unreported"


def _latency_phrase(latency_s: float | None) -> str:
    if latency_s is None:
        return "latency unreported"
    return f"{max(0.0, float(latency_s)):.2f}s"


def _terminal_width() -> int:
    try:
        from fx1.interactive.terminal import detect_caps

        return int(detect_caps().width)
    except (OSError, ValueError):
        return 80


def _fit_widths(available: int) -> dict[str, int]:
    widths = dict(_MODEL_WIDTHS)
    gaps = 2 * (len(_MODEL_COLUMNS) - 1)
    overflow = sum(widths.values()) + gaps - max(40, available)
    cursor = 0
    while overflow > 0 and cursor < 200:
        key = _SHRINK_ORDER[cursor % len(_SHRINK_ORDER)]
        if widths[key] > 3:
            widths[key] -= 1
            overflow -= 1
        cursor += 1
    return widths


def _clip(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "~"


def _row_line(row: Mapping[str, str], widths: Mapping[str, int]) -> str:
    cells = [
        _clip(str(row.get(column, "")), widths[column]).ljust(widths[column])
        for column in _MODEL_COLUMNS
    ]
    return "  ".join(cells).rstrip()


def _model_row(
    active: str,
    item: Mapping[str, str],
    stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    model = str(item.get("model", ""))
    stat = stats.get(model, {})
    return {
        "active": "*" if model == active else "",
        "model": model,
        "status": str(item.get("status", "-")),
        "host": str(item.get("host", "-")),
        "source": str(item.get("source", "unset")),
        "fingerprint": str(item.get("fingerprint", "-")),
        "served": _served_text(stat.get("served")),
        "latency": _latency_cell(stat.get("latency_s")),
    }


def _served_text(served: Any) -> str:
    if isinstance(served, str) and served:
        return served
    return "-"


def _latency_cell(latency: Any) -> str:
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        return "-"
    return f"{float(latency):.2f}s"


def _approval_line(row: Mapping[str, Any]) -> str:
    key = row.get("key") or "-"
    return (
        f"{row.get('scope', '-')} {row.get('kind', '-')} key={key} "
        f"approved={row.get('approved')} reusable={row.get('reusable')}"
    )


_MODELS_USAGE = "usage: /models [check [--live] [fx1|fx1-lite]]"


def models_command(
    rest: str,
    *,
    active: str,
    stats: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, dict[str, Any]]]:
    """``/models`` table, or ``check`` text plus live observations to merge."""
    tokens = _model_tokens(rest)
    if not tokens:
        return render_models_table(active, stats), {}
    if tokens[0] != "check":
        raise ValueError(_MODELS_USAGE)
    report = _run_check(tokens[1:])
    return _check_text(report), _live_updates(report)


def _model_tokens(rest: str) -> list[str]:
    import shlex

    return shlex.split(rest)


def _run_check(tokens: list[str]) -> Any:
    from fx1.interactive.models_check import check_models

    live, models = _parse_check(tokens)
    return check_models(models, live=live)


def _parse_check(tokens: list[str]) -> tuple[bool, tuple[str, ...] | None]:
    live = False
    chosen: list[str] = []
    for token in tokens:
        if token == "--live":
            live = True
            continue
        if token.startswith("-"):
            raise ValueError(_MODELS_USAGE)
        chosen.append(token)
    if len(chosen) > 1:
        raise ValueError(_MODELS_USAGE)
    if not chosen:
        return live, None
    return live, (chosen[0],)


def _check_text(report: Any) -> str:
    from fx1.interactive.models_check import format_check_report

    return format_check_report(report)


def _live_updates(report: Any) -> dict[str, dict[str, Any]]:
    updates: dict[str, dict[str, Any]] = {}
    for row in report.models:
        if not row.contacted:
            continue
        updates[str(row.requested_model)] = _live_stat(row)
    return updates


def _live_stat(row: Any) -> dict[str, Any]:
    return {
        "served": row.served_model,
        "latency_s": _latency_seconds(row.latency_ms),
        "matched": row.matched,
    }


def _latency_seconds(latency_ms: Any) -> float | None:
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)):
        return None
    return float(latency_ms) / 1000.0
