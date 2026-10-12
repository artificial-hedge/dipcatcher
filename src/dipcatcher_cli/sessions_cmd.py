"""Non-interactive ``dipcatcher sessions`` and ``dipcatcher resume``.

Exit codes are 0 (ok), 1 (runtime error), and 2 (usage, an unknown or
ambiguous session, or a refusal). JSON on stdout is one document with
schema ``dipcatcher.sessions.v1``. Secrets are not printed.

``resume`` with no session opens chat the same way as ``chat --resume``.
A prefix that is not the newest session is restored inside that console.
An interrupted job is marked interrupted when it was still running, and it
is never started again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast

SESSIONS_SCHEMA = "dipcatcher.sessions.v1"
_DEFAULT_LIST_LIMIT = 50
_DEFAULT_MAX_SESSIONS = 200
_DEFAULT_MAX_AGE_DAYS = 90
EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2
_LIST_ALL = 1_000_000_000
_MODEL_IDS = ("fx1", "fx1-lite")
_SUBCOMMANDS = frozenset({"list", "show", "export", "rm", "prune"})
_SHOW_FIELDS = ("id", "model", "status", "revision", "created_at", "updated_at", "truncated")
_JOB_NOTES = {
    "interrupted": "an interrupted job is not restarted",
    "running": "a running job is not restarted",
}
_USAGE_MARKERS = (
    "no session",
    "ambiguous",
    "empty session",
    "refusing session",
    "missing session",
    "must be a non-negative",
    "unknown model",
)
_TTY_MESSAGE = (
    "resume requires an interactive terminal on stdin and stdout; "
    "use explicit lab commands or fxi one-shot commands for scripts"
)
_EPILOG = """
exit codes:
  0    ok
  1    runtime error
  2    usage error, ambiguous or unknown session, or refusal

json documents use schema dipcatcher.sessions.v1.
Sessions are read from $FX1_SESSION_DIR, else $FX1_CONFIG_DIR/sessions,
else ~/.fx1/sessions. Corrupt files are warnings on stderr.
An interrupted job is not restarted.
"""
_RESUME_EPILOG = """
With no session, this is dipcatcher chat --resume: chat opens, and the
newest session is reloaded when one exists.
last or a prefix must match one session; otherwise the command exits 2.
An interrupted job is not restarted.
"""

Clock = Callable[[], datetime]


class _UsageError(Exception):
    """A usage problem or a refusal. The process exits 2."""


def sessions_main(
    argv: Sequence[str] | None = None,
    *,
    clock: Clock | None = None,
    isatty: Callable[[], bool] | None = None,
    read_line: Callable[[], str] | None = None,
) -> None:
    """Run ``dipcatcher sessions``. Raises ``SystemExit`` with 0, 1, or 2.

    ``clock`` freezes prune age checks. ``isatty`` and ``read_line`` override
    the ``rm`` confirmation prompt. Omitting the subcommand lists sessions.
    """
    parser = _sessions_parser()
    options = parser.parse_args(_normalize(_argv(argv)))
    _run_sessions(options, clock=clock, isatty=isatty, read_line=read_line)


def resume_main(
    argv: Sequence[str] | None = None,
    *,
    run_console: Callable[..., None] | None = None,
    isatty: Callable[[], bool] | None = None,
) -> None:
    """Open chat on a saved session. Raises ``SystemExit`` on refusal.

    No session argument calls ``run_console(model=..., resume=True)``.
    Any other resolved id is restored by ``concierge.restore(session_id)``
    when that method exists, otherwise by the same steps as
    ``Concierge.restore_latest`` for that one id. The job is not started.
    """
    parser = _resume_parser()
    options = parser.parse_args(_argv(argv))
    if not _console_tty(isatty):
        parser.error(_TTY_MESSAGE)
    from fx1.interactive.sessions import SessionError

    try:
        target = _resume_target(options.session)
        model, explicit = _resume_model(options.model)
        opener = run_console if run_console is not None else _load_run_console()
        _open_resume(opener, model=model, explicit=explicit, target=target)
    except _UsageError as exc:
        _fail_usage(str(exc))
    except SessionError as exc:
        _fail_store(exc)


def _sessions_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dipcatcher sessions",
        description="List, show, export, remove, and prune saved chat sessions.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_list(sub)
    _add_show(sub)
    _add_export(sub)
    _add_rm(sub)
    _add_prune(sub)
    return parser


def _add_list(sub: Any) -> None:
    listed = sub.add_parser("list", help="newest sessions: id, model, status, turns, updated")
    listed.add_argument(
        "--limit",
        type=int,
        default=_DEFAULT_LIST_LIMIT,
        help="maximum rows (default: %(default)s)",
    )
    listed.add_argument(
        "--json",
        action="store_true",
        help="print one dipcatcher.sessions.v1 document",
    )


def _add_show(sub: Any) -> None:
    shown = sub.add_parser("show", help="metadata and turn table for one session")
    shown.add_argument("session", help="session id, unique prefix, or last")
    shown.add_argument(
        "--json",
        action="store_true",
        help="print one dipcatcher.sessions.v1 document",
    )
    shown.add_argument(
        "--messages",
        action="store_true",
        help="include redacted message bodies (omitted by default)",
    )


def _add_export(sub: Any) -> None:
    exported = sub.add_parser("export", help="write a redacted markdown or JSON transcript")
    exported.add_argument("session", help="session id, unique prefix, or last")
    exported.add_argument("--format", choices=("md", "json"), default="md")
    exported.add_argument(
        "--out",
        default=None,
        help="write atomically with mode 0600; refuse to overwrite",
    )


def _add_rm(sub: Any) -> None:
    removed = sub.add_parser("rm", help="delete one session")
    removed.add_argument("session", help="session id, unique prefix, or last")
    removed.add_argument(
        "--yes",
        action="store_true",
        help="delete without a confirmation prompt",
    )


def _add_prune(sub: Any) -> None:
    pruned = sub.add_parser("prune", help="drop old sessions and those past the count cap")
    pruned.add_argument("--max-sessions", type=int, default=_DEFAULT_MAX_SESSIONS)
    pruned.add_argument("--max-age-days", type=int, default=_DEFAULT_MAX_AGE_DAYS)
    pruned.add_argument("--dry-run", action="store_true", help="report ids without deleting")
    pruned.add_argument(
        "--json",
        action="store_true",
        help="print one dipcatcher.sessions.v1 document",
    )


def _resume_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dipcatcher resume",
        description=(
            "Open chat on a saved session. An interrupted job is not restarted. "
            "Requires an interactive terminal."
        ),
        epilog=_RESUME_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "session", nargs="?", default=None, help="session id, unique prefix, or last"
    )
    parser.add_argument("--model", choices=_MODEL_IDS, default=None, help="fx1 or fx1-lite")
    return parser


def _normalize(args: Sequence[str]) -> list[str]:
    items = list(args)
    if items and items[0] in {"-h", "--help"}:
        return items
    if not items or items[0] not in _SUBCOMMANDS:
        return ["list", *items]
    return items


def _argv(argv: Sequence[str] | None) -> list[str]:
    if argv is None:
        return list(sys.argv[1:])
    return list(argv)


def _run_sessions(
    options: argparse.Namespace,
    *,
    clock: Clock | None,
    isatty: Callable[[], bool] | None,
    read_line: Callable[[], str] | None,
) -> None:
    from fx1.interactive.atomicstore import StoreError
    from fx1.interactive.sessions import SessionError

    try:
        _dispatch_sessions(options, clock=clock, isatty=isatty, read_line=read_line)
    except _UsageError as exc:
        _fail_usage(str(exc))
    except SessionError as exc:
        _fail_store(exc)
    except (OSError, StoreError, json.JSONDecodeError, ValueError) as exc:
        _fail_runtime(_runtime_message(exc))


def _dispatch_sessions(
    options: argparse.Namespace,
    *,
    clock: Clock | None,
    isatty: Callable[[], bool] | None,
    read_line: Callable[[], str] | None,
) -> None:
    command = options.command
    if command == "list":
        _cmd_list(options)
        return
    if command == "show":
        _cmd_show(options)
        return
    if command == "export":
        _cmd_export(options)
        return
    if command == "rm":
        _cmd_rm(options, isatty=isatty, read_line=read_line)
        return
    if command == "prune":
        _cmd_prune(options, clock)
        return
    raise _UsageError(f"unknown sessions command {command}")


def _cmd_list(options: argparse.Namespace) -> None:
    limit = _non_negative(options.limit, "limit")
    rows, warnings = _session_store().list(limit=limit)
    cleaned = _clean_warnings(warnings)
    _emit_warnings(cleaned)
    sessions = [_public_summary(row) for row in rows]
    if options.json:
        _write_json(
            {
                "schema": SESSIONS_SCHEMA,
                "command": "list",
                "sessions": sessions,
                "warnings": cleaned,
            }
        )
        return
    _write_stdout(_format_list(sessions))


def _cmd_show(options: argparse.Namespace) -> None:
    store = _session_store()
    record = _load_ref(store, options.session)
    view = _show_view(record, messages=bool(options.messages))
    if options.json:
        _write_json({"schema": SESSIONS_SCHEMA, "command": "show", "session": view})
        return
    _write_stdout(_format_show(view))


def _cmd_export(options: argparse.Namespace) -> None:
    store = _session_store()
    record = _load_ref(store, options.session)
    body = _export_body(store, record, str(options.format))
    destination = _destination(options.out)
    if destination is None:
        _write_stdout(_ensure_newline(body), scrub=False)
        return
    _write_export(destination, body)


def _cmd_rm(
    options: argparse.Namespace,
    *,
    isatty: Callable[[], bool] | None,
    read_line: Callable[[], str] | None,
) -> None:
    store = _session_store()
    session_id = store.resolve_prefix(_require_text(options.session, "session reference"))
    reader = read_line if read_line is not None else _stdin_line
    allowed = _allow_delete(
        session_id,
        yes=bool(options.yes),
        interactive=_stdin_tty(isatty),
        read_line=reader,
    )
    if not allowed:
        raise _UsageError(f"refusing to delete session {session_id} without confirmation")
    if not store.delete(session_id):
        raise _UsageError(f"missing session {session_id}")
    _write_stdout(f"deleted {session_id}\n")


def _cmd_prune(options: argparse.Namespace, clock: Clock | None) -> None:
    max_sessions = _non_negative(options.max_sessions, "max-sessions")
    max_age_days = _non_negative(options.max_age_days, "max-age-days")
    store = _session_store()
    rows, warnings = store.list(limit=_LIST_ALL)
    cleaned = _clean_warnings(warnings)
    _emit_warnings(cleaned)
    planned = _select_prune_ids(
        rows,
        now=_now(clock),
        max_sessions=max_sessions,
        max_age_days=max_age_days,
    )
    removed = 0
    if not options.dry_run:
        removed = store.prune(max_sessions, max_age_days, clock=clock)
    payload = _prune_payload(
        dry_run=bool(options.dry_run),
        removed=removed,
        planned=planned,
        max_sessions=max_sessions,
        max_age_days=max_age_days,
        warnings=cleaned,
    )
    if options.json:
        _write_json(payload)
        return
    _write_stdout(_format_prune(planned, dry_run=bool(options.dry_run), removed=removed))


def _prune_payload(
    *,
    dry_run: bool,
    removed: int,
    planned: Sequence[str],
    max_sessions: int,
    max_age_days: int,
    warnings: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": SESSIONS_SCHEMA,
        "command": "prune",
        "dry_run": dry_run,
        "deleted": removed,
        "planned": list(planned),
        "max_sessions": max_sessions,
        "max_age_days": max_age_days,
        "warnings": list(warnings),
    }


def _load_ref(store: Any, reference: object) -> dict[str, Any]:
    session_id = store.resolve_prefix(_require_text(reference, "session reference"))
    record = store.load(session_id)
    if not isinstance(record, dict):
        _raise_session("session record must be an object")
    return record


def _public_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    turns = row.get("n_turns")
    count = turns if isinstance(turns, int) and not isinstance(turns, bool) else 0
    return {
        "id": row.get("id"),
        "model": row.get("model"),
        "status": row.get("status"),
        "turns": count,
        "updated": row.get("updated_at"),
    }


def _format_list(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["id\tmodel\tstatus\tturns\tupdated"]
    lines.extend(_list_line(row) for row in rows)
    return "\n".join(lines) + "\n"


def _list_line(row: Mapping[str, Any]) -> str:
    cells = (
        row.get("id"),
        row.get("model"),
        row.get("status"),
        row.get("turns"),
        row.get("updated"),
    )
    return "\t".join(_cell(value) for value in cells)


def _show_view(record: Mapping[str, Any], *, messages: bool) -> dict[str, Any]:
    job = _job_view(record.get("job"))
    view: dict[str, Any] = {
        "id": record.get("id"),
        "model": record.get("model"),
        "status": record.get("status"),
        "revision": record.get("revision", 0),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "truncated": record.get("truncated") is True,
        "job": job,
        "turns": _turn_views(record.get("turns"), details=messages),
    }
    note = _job_note(job.get("state"))
    if note:
        view["note"] = note
    if messages:
        view["messages"] = _message_views(record.get("messages"))
    return _as_object(_scrub_value(view), "show view must be an object")


def _job_view(job: object) -> dict[str, Any]:
    if not isinstance(job, Mapping):
        return {"state": "idle", "id": "", "resume": "none", "steps_completed": []}
    steps = job.get("steps_completed")
    done = [step for step in steps if isinstance(step, str)] if isinstance(steps, list) else []
    return {
        "state": job.get("state") if isinstance(job.get("state"), str) else "idle",
        "id": job.get("id") if isinstance(job.get("id"), str) else "",
        "resume": job.get("resume") if isinstance(job.get("resume"), str) else "",
        "steps_completed": done,
    }


def _job_note(state: object) -> str:
    if not isinstance(state, str):
        return ""
    return _JOB_NOTES.get(state, "")


def _turn_views(turns: object, *, details: bool) -> list[dict[str, Any]]:
    if not isinstance(turns, list):
        return []
    return [_turn_view(turn, details=details) for turn in turns if isinstance(turn, Mapping)]


def _turn_view(turn: Mapping[str, Any], *, details: bool) -> dict[str, Any]:
    view: dict[str, Any] = {
        "index": turn.get("index"),
        "model": turn.get("model"),
        "served_model": turn.get("served_model"),
        "outcome": turn.get("outcome"),
        "usage": _usage_view(turn.get("usage")),
        "started_at": turn.get("started_at"),
    }
    if details:
        view["detail"] = turn.get("detail")
    return view


def _usage_view(usage: object) -> dict[str, int | float] | None:
    if not isinstance(usage, Mapping):
        return None
    clean: dict[str, int | float] = {}
    for key, value in usage.items():
        if isinstance(key, str) and _is_number(value):
            clean[key] = value
    return clean


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _message_views(messages: object) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    views: list[dict[str, Any]] = []
    for item in messages:
        if isinstance(item, Mapping):
            views.append({str(key): value for key, value in item.items()})
    return views


def _format_show(view: Mapping[str, Any]) -> str:
    lines = [_meta_line(view, key) for key in _SHOW_FIELDS]
    lines.extend(_format_job(view.get("job"), view.get("note")))
    lines.extend(_format_turns(view.get("turns")))
    if "messages" in view:
        lines.extend(_format_messages(view.get("messages")))
    return "\n".join(lines) + "\n"


def _meta_line(view: Mapping[str, Any], key: str) -> str:
    value = view.get(key)
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = _cell(value)
    return f"{key}: {rendered}"


def _format_job(job: object, note: object) -> list[str]:
    if not isinstance(job, Mapping):
        return []
    lines = [
        f"job_state: {_cell(job.get('state'))}",
        f"job_id: {_cell(job.get('id'))}",
        f"job_resume: {_cell(job.get('resume'))}",
        f"job_steps: {_step_text(job.get('steps_completed'))}",
    ]
    if isinstance(note, str) and note:
        lines.append(f"note: {note}")
    return lines


def _step_text(steps: object) -> str:
    if not isinstance(steps, list):
        return ""
    return ", ".join(step for step in steps if isinstance(step, str))


def _format_turns(turns: object) -> list[str]:
    rows = turns if isinstance(turns, list) else []
    lines = ["turns:"]
    if not rows:
        lines.append("(none)")
        return lines
    lines.append("index\tmodel\tserved_model\toutcome\tusage")
    for turn in rows:
        if isinstance(turn, Mapping):
            lines.append(_turn_line(turn))
    return lines


def _turn_line(turn: Mapping[str, Any]) -> str:
    cells = (
        turn.get("index"),
        turn.get("model"),
        turn.get("served_model"),
        turn.get("outcome"),
        _usage_cell(turn.get("usage")),
    )
    line = "\t".join(_cell(value) for value in cells)
    detail = turn.get("detail")
    if "detail" not in turn or not isinstance(detail, str) or not detail:
        return line
    return f"{line}\t{detail}"


def _usage_cell(usage: object) -> str:
    if not isinstance(usage, Mapping):
        return ""
    parts: list[str] = []
    for key in sorted(usage):
        value = usage[key]
        if isinstance(key, str) and _is_number(value):
            parts.append(f"{key}={value}")
    return ",".join(parts)


def _format_messages(messages: object) -> list[str]:
    rows = messages if isinstance(messages, list) else []
    lines = ["messages:"]
    for item in rows:
        if isinstance(item, Mapping):
            lines.append(f"- {_cell(item.get('role'))}: {_cell(item.get('content'))}")
    return lines


def _export_body(store: Any, record: Mapping[str, Any], fmt: str) -> str:
    if fmt == "json":
        return _export_json_document(store, record)
    if fmt == "md":
        return _as_text(store.export_markdown(record), "session export must be text")
    raise _UsageError(f"unknown export format {fmt!r}")


def _export_json_document(store: Any, record: Mapping[str, Any]) -> str:
    loaded = json.loads(store.export_json(record))
    document = {
        "schema": SESSIONS_SCHEMA,
        "command": "export",
        "session": _as_object(loaded, "session export must be an object"),
    }
    return json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n"


def _destination(raw: object) -> Path | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise _UsageError("refusing empty --out path")
    if "\x00" in raw or "\n" in raw or "\r" in raw:
        raise _UsageError("refusing unsafe --out path")
    return Path(raw)


def _write_export(path: Path, text: str) -> None:
    if os.path.lexists(path):
        raise _UsageError(f"refusing to overwrite {path}")
    from fx1.interactive.atomicstore import atomic_write_text

    try:
        atomic_write_text(path, _ensure_newline(text), mode=0o600, exclusive=True)
    except FileExistsError as exc:
        raise _UsageError(f"refusing to overwrite {path}") from exc
    print(_scrub(f"wrote {path}"), file=sys.stderr, flush=True)


def _select_prune_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
    max_sessions: int,
    max_age_days: int,
) -> list[str]:
    """Match ``SessionStore.prune`` selection. Newest rows come first."""
    cutoff = now.timestamp() - (max_age_days * 86400)
    aged: list[str] = []
    fresh: list[str] = []
    for row in rows:
        session_id = row.get("id")
        if not isinstance(session_id, str):
            continue
        if _older_than(row.get("updated_at"), cutoff):
            aged.append(session_id)
        else:
            fresh.append(session_id)
    return _dedupe(aged + fresh[max_sessions:])


def _older_than(updated: object, cutoff: float) -> bool:
    if not isinstance(updated, str):
        return False
    stamp = _parse_utc(updated)
    return stamp is not None and stamp.timestamp() < cutoff


def _parse_utc(value: str) -> datetime | None:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _now(clock: Clock | None) -> datetime:
    current = datetime.now(UTC) if clock is None else clock()
    if not isinstance(current, datetime):
        _raise_session("clock must return a datetime")
    if current.tzinfo is None:
        return current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def _resume_target(reference: object) -> str | None:
    if reference is None:
        return None
    store = _session_store()
    session_id = store.resolve_prefix(_require_text(reference, "session reference"))
    if not isinstance(session_id, str):
        _raise_session("session reference must be a string")
    store.load(session_id)
    return session_id


def _resume_model(selected: object) -> tuple[str, bool]:
    if selected is None:
        return _default_model(), False
    if not isinstance(selected, str):
        raise _UsageError("model must be a string")
    return _require_model(selected), True


def _open_resume(
    opener: Callable[..., None],
    *,
    model: str,
    explicit: bool,
    target: str | None,
) -> None:
    if target is None or _is_latest(target):
        opener(model=model, resume=True, model_explicit=explicit)
        return
    preferred = model if explicit else None
    opener(model=model, concierge_factory=_resume_factory(target, preferred))


def _is_latest(session_id: str) -> bool:
    latest = _session_store().latest()
    return isinstance(latest, dict) and latest.get("id") == session_id


def _resume_factory(session_id: str, preferred: str | None) -> Callable[[], Any]:
    def factory() -> Any:
        from fx1.interactive.sessions import SessionError

        try:
            concierge = _new_concierge(preferred or _default_model())
            _restore_specific(concierge, session_id, preferred)
        except _UsageError as exc:
            _fail_usage(str(exc))
        except SessionError as exc:
            _fail_store(exc)
        return concierge

    return factory


def _new_concierge(model: str) -> Any:
    import typer

    from fx1.interactive.approvals import ApprovalGate, console_decide_fn
    from fx1.interactive.concierge import Concierge

    def echo(message: object = "", nl: bool = True) -> None:
        typer.echo(message, nl=nl)

    output = cast("Callable[..., None]", echo)
    return Concierge(
        model=model,
        input_fn=input,
        output_fn=output,
        approvals=ApprovalGate(decide_fn=console_decide_fn(input, echo)),
        session_store=_session_store(),
    )


def _restore_specific(concierge: Any, session_id: str, preferred: str | None) -> None:
    """Restore one id. ``restore(session_id)`` is used when the console has it.

    Otherwise a running job is marked interrupted and the transcript is
    installed. Nothing starts the job.
    """
    hook = getattr(concierge, "restore", None)
    if callable(hook):
        hook(session_id)
        _apply_preferred(concierge, preferred)
        _ensure_interrupt_notice(concierge)
        return
    _restore_via_store(concierge, session_id, preferred)


def _apply_preferred(concierge: Any, preferred: str | None) -> None:
    if preferred is None:
        return
    concierge.model = _require_model(preferred)
    session = getattr(concierge, "_session", None)
    if isinstance(session, dict):
        session["model"] = concierge.model


def _ensure_interrupt_notice(concierge: Any) -> None:
    session = getattr(concierge, "_session", None)
    if _job_state(session) != "interrupted":
        return
    notice = str(getattr(concierge, "resume_notice", "") or "")
    if "not restarted" in notice:
        return
    suffix = "the previous job was not restarted"
    concierge.resume_notice = f"{notice}; {suffix}" if notice else suffix


def _restore_via_store(concierge: Any, session_id: str, preferred: str | None) -> None:
    store = getattr(concierge, "_session_store", None)
    if store is None:
        concierge.resume_notice = "session store is not configured"
        return
    record = store.load(session_id)
    if not isinstance(record, dict):
        _raise_session("session record must be an object")
    _install_record(concierge, _interrupt_running(store, record), preferred)


def _interrupt_running(store: Any, record: dict[str, Any]) -> dict[str, Any]:
    if _job_state(record) != "running":
        return record
    session_id = record.get("id")
    if not isinstance(session_id, str):
        _raise_session("refusing session id")
    updated = store.mark_interrupted(session_id)
    if not isinstance(updated, dict):
        _raise_session("session record must be an object")
    return updated


def _install_record(concierge: Any, record: dict[str, Any], preferred: str | None) -> None:
    concierge._session = record
    restored = record.get("model") if isinstance(record.get("model"), str) else concierge.model
    chosen = preferred if preferred is not None else restored
    if not isinstance(chosen, str):
        raise _UsageError("session model is missing")
    concierge.model = _require_model(str(chosen))
    if preferred is not None:
        record["model"] = concierge.model
    concierge._messages = _message_dicts(record)
    budget = record.get("budget")
    concierge._budget_overrides = dict(budget) if isinstance(budget, dict) else {}
    notice = f"resumed session {record.get('id')} on model {concierge.model}"
    if _job_state(record) == "interrupted":
        notice += "; the previous job was not restarted"
    concierge.resume_notice = notice


def _message_dicts(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    return [dict(item) for item in messages if isinstance(item, dict)]


def _job_state(record: object) -> str | None:
    if not isinstance(record, Mapping):
        return None
    job = record.get("job")
    if not isinstance(job, Mapping):
        return None
    state = job.get("state")
    if isinstance(state, str):
        return state
    return None


def _default_model() -> str:
    from fx1.interactive.profiles import DEFAULT_MODEL

    return str(DEFAULT_MODEL)


def _require_model(model: str) -> str:
    from fx1.interactive.profiles import check_model

    try:
        return check_model(model)
    except KeyError as exc:
        raise _UsageError(_scrub(str(exc))) from exc


def _load_run_console() -> Callable[..., None]:
    from fx1.interactive.concierge import run_console

    return run_console


def _console_tty(isatty: Callable[[], bool] | None) -> bool:
    if isatty is not None:
        return bool(isatty())
    return sys.stdin.isatty() and sys.stdout.isatty()


def _stdin_tty(isatty: Callable[[], bool] | None) -> bool:
    if isatty is not None:
        return bool(isatty())
    return sys.stdin.isatty()


def _allow_delete(
    session_id: str,
    *,
    yes: bool,
    interactive: bool,
    read_line: Callable[[], str],
) -> bool:
    if yes:
        return True
    if not interactive:
        return False
    prompt = _scrub(f"delete session {session_id}? [y/N] ")
    print(prompt, file=sys.stderr, end="", flush=True)
    try:
        answer = read_line()
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def _session_store() -> Any:
    from fx1.interactive.sessions import SessionStore

    return SessionStore()


def _raise_session(message: str) -> NoReturn:
    from fx1.interactive.sessions import SessionError

    raise SessionError(message)


def _scrub(text: str) -> str:
    from fx1.interactive.redact import redact_text

    return redact_text(text)


def _scrub_value(value: object) -> Any:
    from fx1.interactive.redact import redact_value

    return redact_value(value)


def _stdin_line() -> str:
    return sys.stdin.readline()


def _require_text(value: object, label: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    raise _UsageError(f"{label} must be a non-empty string")


def _non_negative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _UsageError(f"{label} must be a non-negative integer")
    return value


def _as_text(value: object, message: str) -> str:
    if isinstance(value, str):
        return value
    _raise_session(message)


def _as_object(value: object, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _raise_session(message)
    return {str(key): item for key, item in value.items()}


def _clean_warnings(warnings: Sequence[str]) -> list[str]:
    return [_scrub(item) for item in warnings]


def _emit_warnings(warnings: Sequence[str]) -> None:
    for warning in warnings:
        print(warning, file=sys.stderr, flush=True)


def _cell(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _ensure_newline(text: str) -> str:
    if text.endswith("\n"):
        return text
    return text + "\n"


def _write_stdout(text: str, *, scrub: bool = True) -> None:
    sys.stdout.write(_scrub(text) if scrub else text)
    sys.stdout.flush()


def _write_json(payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    _write_stdout(rendered + "\n")


def _format_prune(planned: Sequence[str], *, dry_run: bool, removed: int) -> str:
    if dry_run:
        lines = [f"dry-run: would delete {len(planned)}", *planned]
        return "\n".join(lines) + "\n"
    return f"deleted {removed}\n"


def _store_exit(exc: BaseException) -> int:
    text = str(exc).lower()
    if any(marker in text for marker in _USAGE_MARKERS):
        return EXIT_USAGE
    return EXIT_RUNTIME


def _runtime_message(exc: BaseException) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "session export was not valid JSON"
    return str(exc)


def _fail_usage(message: str) -> NoReturn:
    print(_scrub(message), file=sys.stderr, flush=True)
    raise SystemExit(EXIT_USAGE)


def _fail_store(exc: BaseException) -> NoReturn:
    print(_scrub(str(exc)), file=sys.stderr, flush=True)
    raise SystemExit(_store_exit(exc)) from exc


def _fail_runtime(message: str) -> NoReturn:
    print(_scrub(message), file=sys.stderr, flush=True)
    raise SystemExit(EXIT_RUNTIME)
