"""Offline and one-call checks for the fx1 and fx1-lite endpoints.

Offline mode validates configuration and does not open a socket. Live mode
makes one ``ModelClient.ping`` per selected configured model and records the
identity that call proved. A mismatch between the requested and served ids
fails the check. Nothing here claims a model is released or generally available.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

CHECK_SCHEMA = "dipcatcher.models_check.v1"
LIST_SCHEMA = "dipcatcher.models.v1"
_DEFAULT_TIMEOUT_S = 30.0
_NO_ENDPOINT = "no_endpoint"


@dataclass(frozen=True)
class ModelCheck:
    """One model's configuration, and the live call when one was made."""

    requested_model: str
    configured: bool
    presence: str
    host: str
    source: str
    fingerprint: str
    url_valid: bool
    contacted: bool
    served_model: str | None
    matched: bool | None
    request_id: str | None
    latency_ms: float | None
    usage: dict[str, int] | None
    status: str
    error_kind: str | None
    hint: str | None

    def as_json(self) -> dict[str, Any]:
        usage = None if self.usage is None else dict(self.usage)
        return {
            "requested_model": self.requested_model,
            "served_model": self.served_model,
            "matched": self.matched,
            "request_id": self.request_id,
            "host": self.host,
            "latency_ms": self.latency_ms,
            "usage": usage,
            "status": self.status,
            "error_kind": self.error_kind,
            "hint": self.hint,
            "configured": self.configured,
            "presence": self.presence,
            "source": self.source,
            "fingerprint": self.fingerprint,
            "url_valid": self.url_valid,
            "contacted": self.contacted,
        }


@dataclass(frozen=True)
class ModelsReport:
    """A ``dipcatcher.models_check.v1`` record. ``contacted`` is false offline."""

    live: bool
    generated_at: str
    fx1_version: str
    contacted: bool
    store_warnings: tuple[str, ...]
    models: tuple[ModelCheck, ...]
    note: str

    def as_json(self) -> dict[str, Any]:
        return {
            "schema": CHECK_SCHEMA,
            "live": self.live,
            "generated_at": self.generated_at,
            "fx1_version": self.fx1_version,
            "contacted": self.contacted,
            "note": self.note,
            "store_warnings": list(self.store_warnings),
            "models": [row.as_json() for row in self.models],
        }


def check_models(
    models: Sequence[str] | None = None,
    *,
    live: bool = False,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    client_factory: Callable[..., Any] | None = None,
    clock: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] | None = None,
    path: Any = None,
    environ: Mapping[str, str] | None = None,
) -> ModelsReport:
    """Check the selected models. ``models`` empty means both fx1 and fx1-lite."""
    selected = _selected(models)
    _require_timeout(timeout_s)
    rows = {row["model"]: row for row in _list(path, environ)}
    warnings = _warnings(path)
    factory = default_client_factory if client_factory is None else client_factory
    clock_fn = time.monotonic if monotonic is None else monotonic
    checks = tuple(
        _check_one(
            model,
            rows.get(model),
            live=live,
            timeout_s=timeout_s,
            client_factory=factory,
            monotonic=clock_fn,
            path=path,
            environ=environ,
        )
        for model in selected
    )
    return _report(live, checks, warnings, clock)


def report_exit_code(report: ModelsReport) -> int:
    """0 when every selected model passed; 1 on a live failure; 5 if one has no endpoint."""
    if any(row.status == "failed" for row in report.models):
        return 1
    if any(not row.configured for row in report.models):
        return 5
    return 0


def presence_report(
    *,
    path: Any = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    """Presence rows for both models plus store warnings. No network."""
    return _list(path, environ), _warnings(path)


def list_payload(rows: Sequence[Mapping[str, str]], warnings: Sequence[str]) -> dict[str, Any]:
    """``dipcatcher.models.v1`` document. Rows are presence fields only."""
    return {
        "schema": LIST_SCHEMA,
        "models": [dict(row) for row in rows],
        "store_warnings": list(warnings),
    }


def format_model_list(rows: Sequence[Mapping[str, str]], warnings: Sequence[str]) -> str:
    """Plain presence table. Does not describe a model as available."""
    lines = ["model        status       host                       source       fingerprint"]
    for row in rows:
        lines.append(_list_line(row))
    if warnings:
        lines.append("store warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def format_check_report(report: ModelsReport) -> str:
    """Plain check text. Offline text says nothing was contacted."""
    lines = [report.note, f"fx1_version={report.fx1_version} generated_at={report.generated_at}"]
    lines.extend(_format_check_row(row) for row in report.models)
    lines.extend(f"store warning: {warning}" for warning in report.store_warnings)
    return "\n".join(lines)


def default_client_factory(
    *,
    api_key: str,
    model: str,
    api_url: str,
    timeout_s: float,
) -> Any:
    """One-attempt client so a smoke check records the first response."""
    from fx1.interactive.model_client import ModelClient, RetryPolicy

    policy = RetryPolicy(max_attempts=1)
    return ModelClient(
        api_key=api_key,
        model=model,
        api_url=api_url,
        timeout_s=timeout_s,
        retry=policy,
    )


def _report(
    live: bool,
    checks: tuple[ModelCheck, ...],
    warnings: tuple[str, ...],
    clock: Callable[[], datetime] | None,
) -> ModelsReport:
    from fx1 import __version__

    note = _note(live)
    contacted = live and any(row.contacted for row in checks)
    return ModelsReport(
        live=live,
        generated_at=_stamp(clock),
        fx1_version=__version__,
        contacted=contacted,
        store_warnings=warnings,
        models=checks,
        note=note,
    )


def _note(live: bool) -> str:
    if live:
        return "live check: one call per configured model; this record is not a release claim"
    return "offline configuration check; nothing was contacted"


def _check_one(
    model: str,
    row: Mapping[str, str] | None,
    *,
    live: bool,
    timeout_s: float,
    client_factory: Callable[..., Any],
    monotonic: Callable[[], float],
    path: Any,
    environ: Mapping[str, str] | None,
) -> ModelCheck:
    offline = _offline(model, row, path, environ)
    if not live or not offline.configured:
        return offline
    return _live(offline, client_factory, timeout_s, monotonic, path, environ)


def _offline(
    model: str,
    row: Mapping[str, str] | None,
    path: Any,
    environ: Mapping[str, str] | None,
) -> ModelCheck:
    presence, host, source, fingerprint = _presence(row)
    try:
        info = _resolve(model, path, environ)
    except ValueError as exc:
        return _base(
            model,
            presence=presence,
            host=host,
            source=source,
            fingerprint=fingerprint,
            configured=False,
            url_valid=False,
            error_kind="invalid_url",
            hint=_scrub(str(exc)),
        )
    if info is None:
        hint = (
            f"no endpoint for model {model}; run `/keys set {model}` "
            "or set that model's environment variables"
        )
        return _base(
            model,
            presence=presence,
            host=host,
            source=source,
            fingerprint=fingerprint,
            configured=False,
            url_valid=False,
            error_kind=_NO_ENDPOINT,
            hint=hint,
        )
    resolved_host = info.host or host
    return _base(
        model,
        presence=presence,
        host=resolved_host,
        source=source,
        fingerprint=fingerprint,
        configured=True,
        url_valid=True,
        error_kind=None,
        hint=None,
    )


def _live(
    base: ModelCheck,
    client_factory: Callable[..., Any],
    timeout_s: float,
    monotonic: Callable[[], float],
    path: Any,
    environ: Mapping[str, str] | None,
) -> ModelCheck:
    info = _resolve(base.requested_model, path, environ)
    if info is None:
        return base
    started = monotonic()
    box = {"timed_out": False}
    try:
        client = _make_client(client_factory, info, base.requested_model, timeout_s)
        identity = _ping(client, timeout_s, box)
    except Exception as exc:
        latency = _millis(started, monotonic)
        return _from_error(base, exc, latency, timed_out=bool(box["timed_out"]))
    return _from_identity(base, identity, client, _millis(started, monotonic))


def _from_error(base: ModelCheck, exc: Exception, latency: float, *, timed_out: bool) -> ModelCheck:
    from fx1.interactive.model_client import ModelError

    if isinstance(exc, ModelError):
        kind, hint = _check_failure(exc, timed_out=timed_out)
        return _failed(base, kind, hint, exc.host or base.host, latency)
    hint = f"model check failed ({type(exc).__name__})"
    return _failed(base, "error", hint, base.host, latency)


def _check_failure(exc: Any, *, timed_out: bool) -> tuple[str, str]:
    kind = str(exc.kind.value)
    if kind in {"timeout", "deadline"} or (timed_out and kind == "cancelled"):
        return "timeout", "the model check stopped because timeout-s elapsed"
    return kind, str(exc.hint)


def _from_identity(base: ModelCheck, identity: Any, client: Any, latency: float) -> ModelCheck:
    matched = getattr(identity, "matched", None)
    served = _public_name(_text(getattr(identity, "served", None)))
    request_id = _public_name(_text(getattr(identity, "request_id", None)))
    host = _text(getattr(identity, "host", None)) or base.host
    usage = _usage_dict(getattr(client, "last_usage", None))
    matched_flag = matched if isinstance(matched, bool) else None
    if matched_flag is False:
        hint = f"served model {served!r} did not match requested {base.requested_model!r}"
        return _copied(
            base,
            contacted=True,
            served_model=served,
            matched=False,
            request_id=request_id,
            host=host,
            latency_ms=latency,
            usage=usage,
            status="failed",
            error_kind="identity_mismatch",
            hint=hint,
        )
    return _copied(
        base,
        contacted=True,
        served_model=served,
        matched=matched_flag,
        request_id=request_id,
        host=host,
        latency_ms=latency,
        usage=usage,
        status="ok",
        error_kind=None,
        hint=None,
    )


def _failed(base: ModelCheck, kind: str, hint: str, host: str, latency: float) -> ModelCheck:
    return _copied(
        base,
        contacted=True,
        host=host or base.host,
        latency_ms=latency,
        status="failed",
        error_kind=kind,
        hint=_honest_hint(_scrub(hint)),
    )


def _ping(client: Any, timeout_s: float, box: dict[str, bool]) -> Any:
    cancel = threading.Event()

    def mark() -> None:
        box["timed_out"] = True
        cancel.set()

    timer = threading.Timer(timeout_s, mark)
    timer.daemon = True
    timer.start()
    try:
        return client.ping(cancel_event=cancel)
    finally:
        timer.cancel()


def _make_client(factory: Callable[..., Any], info: Any, model: str, timeout_s: float) -> Any:
    return factory(
        api_key=info.api_key,
        model=model,
        api_url=info.completion_url,
        timeout_s=timeout_s,
    )


def _base(
    model: str,
    *,
    presence: str,
    host: str,
    source: str,
    fingerprint: str,
    configured: bool,
    url_valid: bool,
    error_kind: str | None,
    hint: str | None,
) -> ModelCheck:
    return ModelCheck(
        requested_model=model,
        configured=configured,
        presence=presence,
        host=host,
        source=source,
        fingerprint=fingerprint,
        url_valid=url_valid,
        contacted=False,
        served_model=None,
        matched=None,
        request_id=None,
        latency_ms=None,
        usage=None,
        status="not_contacted",
        error_kind=error_kind,
        hint=hint,
    )


def _copied(base: ModelCheck, **changes: Any) -> ModelCheck:
    data = base.as_json()
    data.update(changes)
    return ModelCheck(
        requested_model=str(data["requested_model"]),
        configured=bool(data["configured"]),
        presence=str(data["presence"]),
        host=str(data["host"]),
        source=str(data["source"]),
        fingerprint=str(data["fingerprint"]),
        url_valid=bool(data["url_valid"]),
        contacted=bool(data["contacted"]),
        served_model=_text(data["served_model"]),
        matched=_bool_or_none(data["matched"]),
        request_id=_text(data["request_id"]),
        latency_ms=_float_or_none(data["latency_ms"]),
        usage=_usage_dict(data["usage"]),
        status=str(data["status"]),
        error_kind=_text(data["error_kind"]),
        hint=_text(data["hint"]),
    )


def _presence(row: Mapping[str, str] | None) -> tuple[str, str, str, str]:
    if row is None:
        return "unset", "", "unset", "-"
    return (
        str(row.get("status", "unset")),
        str(row.get("host", "")),
        str(row.get("source", "unset")),
        str(row.get("fingerprint", "-")),
    )


def _selected(models: Sequence[str] | None) -> tuple[str, ...]:
    from fx1.interactive.profiles import MODELS, check_model

    if models is None or len(tuple(models)) == 0:
        return tuple(MODELS)
    return tuple(check_model(model) for model in models)


def _list(path: Any, environ: Mapping[str, str] | None) -> list[dict[str, str]]:
    from fx1.interactive.profiles import list_models

    return list_models(path=path, environ=environ)


def _warnings(path: Any) -> tuple[str, ...]:
    from fx1.interactive.profiles import store_health

    return tuple(_scrub(item) for item in store_health(path))


def _resolve(model: str, path: Any, environ: Mapping[str, str] | None) -> Any:
    from fx1.interactive.profiles import resolve_endpoint_info

    return resolve_endpoint_info(model, path=path, environ=environ)


def _require_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout-s must be a positive number")
    number = float(value)
    if number <= 0 or number != number or number == float("inf"):
        raise ValueError("timeout-s must be a positive finite number")
    return number


def _stamp(clock: Callable[[], datetime] | None) -> str:
    moment = datetime.now(UTC) if clock is None else clock()
    if moment.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return moment.astimezone(UTC).isoformat()


def _millis(started: float, clock: Callable[[], float]) -> float:
    return round(max(0.0, clock() - started) * 1000.0, 3)


def _usage_dict(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    cleaned = {
        str(key): value
        for key, value in raw.items()
        if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
    }
    if not cleaned:
        return None
    return cleaned


def _public_name(value: str | None) -> str | None:
    if value is None:
        return None
    from fx1.honesty import public_label

    return public_label(value)


def _honest_hint(hint: str) -> str:
    from fx1.honesty import scrub_model_text

    if scrub_model_text(hint) == hint:
        return hint
    return "response withheld by the fx-1 honesty gate"


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _list_line(row: Mapping[str, str]) -> str:
    model = str(row.get("model", ""))
    status = str(row.get("status", ""))
    host = str(row.get("host", ""))
    source = str(row.get("source", ""))
    fingerprint = str(row.get("fingerprint", ""))
    return f"{model:<12} {status:<12} {host:<26} {source:<12} {fingerprint}"


def _format_check_row(row: ModelCheck) -> str:
    if not row.contacted:
        configured = str(row.configured).lower()
        return (
            f"{row.requested_model} presence={row.presence} host={row.host} "
            f"source={row.source} configured={configured} not_contacted"
        )
    matched = "n/a" if row.matched is None else str(row.matched).lower()
    served = row.served_model or "-"
    kind = row.error_kind or row.status
    latency = "-" if row.latency_ms is None else str(row.latency_ms)
    return (
        f"{row.requested_model} status={row.status} served={served} matched={matched} "
        f"host={row.host} latency_ms={latency} kind={kind}"
    )


def _scrub(text: str) -> str:
    from fx1.interactive.redact import redact_text

    return redact_text(text)
