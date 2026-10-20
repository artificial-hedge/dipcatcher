"""Operator smoke for dipcatcher gate P2 (both hosted models).

For each of ``fx1`` and ``fx1-lite`` (or one model under ``--allow-partial``),
resolve the configured endpoint, run :meth:`fx1.interactive.model_client.ModelClient.ping`
as the probe, then one short temperature-0 completion. The evidence file
records requested versus served identity, latency, usage, and on failure the
error-taxonomy kind plus hint. The endpoint is stored as the host only.

The API key is never accepted on the command line. ``--base-url`` must be
paired with ``--api-key-env NAME``; the key is read from that environment
variable. A stored profile (``FX1_CONFIG_DIR``) is used when those flags are
omitted. Nothing in the evidence file, stdout, or stderr is the key, an
``Authorization`` header, or a URL with userinfo or a query.

``--require-both`` is the default: both product models run, and the process
exits 1 unless both succeed. A served-id mismatch is a failure. An omitted
served model is recorded as ``unreported`` and does not by itself fail the
run. ``--allow-partial --model X`` records that single model as partial.

Exit codes: 0 every selected model ok, 1 any failure or served-id mismatch,
2 usage error.

``content_sha256`` is the hex SHA-256 of the UTF-8 JSON object with that
field removed, serialized with ``json.dumps(sort_keys=True,
separators=(",", ":"), ensure_ascii=True)``. The file itself is
pretty-printed (``indent=2``, ``sort_keys=True``) plus a trailing newline.

This file is evidence that the harness reached the supplied endpoint on the
recorded date. It does not establish model release status, quality, or
availability for anyone else. A loopback server does not close gate P2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import fx1  # noqa: E402
from fx1.interactive.endpoint import EndpointError, EndpointSpec, normalize_endpoint  # noqa: E402
from fx1.interactive.model_client import (  # noqa: E402
    ModelClient,
    ModelError,
    ModelIdentity,
    RetryPolicy,
)
from fx1.interactive.profiles import MODELS, resolve_endpoint_info  # noqa: E402
from fx1.interactive.redact import redact_text, register_secret  # noqa: E402
from fx1.serve.backends import SamplingParams  # noqa: E402

_SCHEMA = "model_smoke.v1"
_PROMPT = "Reply with the single word pong."
_MAX_TOKENS = 16
_SEED = 0
_DISCLAIMER = (
    "This run proves the harness works with the supplied endpoint on {date} "
    "and does not establish model release status, quality, or availability for anyone else."
)
_DOES_NOT = (
    "model release status",
    "quality",
    "availability for anyone else",
)
_HELP = (
    "Probe fx1 and fx1-lite and write model_smoke.v1 evidence. "
    "The API key comes from --api-key-env or a stored profile, never from argv."
)
_PAIR = "pass --base-url and --api-key-env together; the key is read from the environment, not argv"
_BLANK = "--base-url and --api-key-env must not be blank"
_ENV_NAME = "--api-key-env must name an environment variable (letters, digits, underscore)"
_BOTH_FLAGS = "pass only one of --allow-partial and --require-both"
_NEED_MODEL = "pass --model with --allow-partial"
_NEED_BOTH = (
    "the default policy runs both fx1 and fx1-lite; pass --allow-partial to record a single model"
)
_TEXT_FIELDS = (
    "endpoint_host",
    "error_kind",
    "failed_stage",
    "hint",
    "identity_match",
    "probe_request_id",
    "request_id",
    "requested_model",
    "response_id",
    "served_model",
    "summary",
    "system_fingerprint",
)
_GIT_ENV_KEYS = (
    "PATH",
    "SYSTEMROOT",
    "PATHEXT",
    "COMSPEC",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
)
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_AUTH_LEFTOVER = re.compile(r"(?i)authorization\s*[:=]\s*(?:bearer\s*)?(?:\[REDACTED\]|\S+)?")
_URL_LEFTOVER = re.compile(r"(?i)\b[a-z][a-z0-9+.\-]*://\S+")


@dataclass(frozen=True)
class _Usage:
    message: str


@dataclass(frozen=True)
class _Args:
    models: tuple[str, ...]
    base_url: str | None
    api_key_env: str | None
    out: Path | None
    allow_partial: bool
    timeout_s: float


@dataclass(frozen=True)
class _Failure:
    kind: str
    hint: str
    summary: str
    status: int | None
    retry_after_s: float | None
    request_id: str | None = None


@dataclass(frozen=True)
class _Call:
    latency_ms: float
    identity: ModelIdentity | None
    usage: dict[str, int] | None
    failure: _Failure | None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke and return the process exit code."""
    try:
        return _main(argv)
    except Exception as exc:
        _emit_err(f"model smoke failed ({type(exc).__name__}: {exc})")
        return 1


def _main(argv: Sequence[str] | None) -> int:
    parsed = _parse(argv)
    if isinstance(parsed, _Usage):
        _emit_err(parsed.message)
        return 2
    started = _stamp()
    rows = [_one(model, parsed) for model in parsed.models]
    finished = _stamp()
    document = _document(rows, parsed, started, finished, _git_revision())
    path = parsed.out if parsed.out is not None else _default_out()
    try:
        _write(path, document)
    except OSError:
        _emit_err("could not write the smoke evidence file")
        return 2
    _print_report(path, document)
    if document.get("ok") is True:
        return 0
    return 1


def _parse(argv: Sequence[str] | None) -> _Args | _Usage:
    namespace = _parser().parse_args(list(argv) if argv is not None else None)
    problem = _validate(namespace)
    if problem is not None:
        return _Usage(problem)
    return _args_from(namespace)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fx1_model_smoke", description=_HELP)
    parser.add_argument("--model", action="append", choices=MODELS, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-both", action="store_true")
    parser.add_argument("--timeout-s", type=_positive_timeout, default=30.0)
    return parser


def _positive_timeout(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number") from None
    if math.isfinite(value) and value > 0:
        return value
    raise argparse.ArgumentTypeError("timeout must be a positive finite number")


def _validate(namespace: argparse.Namespace) -> str | None:
    pair = _pair_problem(namespace)
    if pair is not None:
        return pair
    return _model_problem(namespace)


def _pair_problem(namespace: argparse.Namespace) -> str | None:
    url_passed = namespace.base_url is not None
    env_passed = namespace.api_key_env is not None
    if url_passed != env_passed:
        return _PAIR
    if not url_passed:
        return None
    if _opt_str(namespace.base_url) is None or _opt_str(namespace.api_key_env) is None:
        return _BLANK
    env_name = _opt_str(namespace.api_key_env)
    if env_name is not None and _ENV_NAME_RE.fullmatch(env_name):
        return None
    return _ENV_NAME


def _model_problem(namespace: argparse.Namespace) -> str | None:
    has_model = isinstance(namespace.model, list) and len(namespace.model) > 0
    if namespace.allow_partial and namespace.require_both:
        return _BOTH_FLAGS
    if namespace.allow_partial:
        return None if has_model else _NEED_MODEL
    if has_model:
        return _NEED_BOTH
    return None


def _args_from(namespace: argparse.Namespace) -> _Args:
    allow_partial = bool(namespace.allow_partial)
    return _Args(
        models=_selected(namespace, allow_partial),
        base_url=_opt_str(namespace.base_url),
        api_key_env=_opt_str(namespace.api_key_env),
        out=_out_path(namespace.out),
        allow_partial=allow_partial,
        timeout_s=_timeout_value(namespace.timeout_s),
    )


def _selected(namespace: argparse.Namespace, allow_partial: bool) -> tuple[str, ...]:
    if not allow_partial:
        return _product_models()
    chosen = namespace.model
    if not isinstance(chosen, list):
        return ()
    names = {item for item in chosen if isinstance(item, str)}
    return tuple(model for model in _product_models() if model in names)


def _product_models() -> tuple[str, ...]:
    found: list[str] = []
    for item in MODELS:
        if isinstance(item, str):
            found.append(item)
    return tuple(found)


def _opt_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def _out_path(value: object) -> Path | None:
    text = _opt_str(value)
    if text is None:
        return None
    return Path(text)


def _timeout_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 30.0
    return float(value)


def _one(model: str, args: _Args) -> dict[str, Any]:
    if args.base_url is not None:
        return _override(model, args)
    return _stored(model, args.timeout_s)


def _override(model: str, args: _Args) -> dict[str, Any]:
    name = args.api_key_env or ""
    key = _env_key(name)
    if key:
        register_secret(key)
    rejected = _key_rejection(key, name)
    if rejected is not None:
        return _resolve_row(model, rejected)
    try:
        spec = normalize_endpoint(args.base_url or "")
    except EndpointError as exc:
        return _resolve_row(model, _endpoint_failure(exc))
    return _run_client(model, key, spec, args.timeout_s)


def _stored(model: str, timeout_s: float) -> dict[str, Any]:
    try:
        info = resolve_endpoint_info(model)
    except ValueError as exc:
        return _resolve_row(model, _stored_failure(exc))
    if info is None:
        return _resolve_row(model, _missing_profile_failure(model))
    register_secret(info.api_key)
    try:
        spec = normalize_endpoint(info.completion_url)
    except EndpointError as exc:
        return _resolve_row(model, _endpoint_failure(exc))
    return _run_client(model, info.api_key, spec, timeout_s)


def _env_key(name: str) -> str:
    raw = os.environ.get(name, "")
    if not isinstance(raw, str):
        return ""
    return raw.strip()


def _key_has_junk(key: str) -> bool:
    return any(ch.isspace() or unicodedata.category(ch) in {"Cc", "Cf"} for ch in key)


def _key_rejection(key: str, name: str) -> _Failure | None:
    if not key:
        return _missing_key_failure(name)
    if _key_has_junk(key):
        return _rejected_key_failure()
    return None


def _missing_key_failure(name: str) -> _Failure:
    hint = f"set environment variable {name}; the key is read from the environment, not argv"
    return _Failure("auth", hint, "missing api key", None, None)


def _rejected_key_failure() -> _Failure:
    hint = "the API key contains whitespace or control characters and was not sent"
    return _Failure("auth", hint, "rejected api key", None, None)


def _missing_profile_failure(model: str) -> _Failure:
    hint = (
        f"no endpoint is configured for {model}; store a profile or pass "
        "--base-url with --api-key-env (the key is read from the environment, not argv)"
    )
    return _Failure("auth", hint, "missing endpoint", None, None)


def _endpoint_failure(exc: EndpointError) -> _Failure:
    hint = redact_text(str(exc))
    return _Failure("invalid_endpoint", hint, "invalid endpoint", None, None)


def _stored_failure(exc: ValueError) -> _Failure:
    text = redact_text(str(exc))
    kind = "auth" if "api key" in text.lower() else "invalid_endpoint"
    return _Failure(kind, text, text, None, None)


def _client_init_failure(exc: BaseException) -> _Failure:
    hint = redact_text(str(exc)) or type(exc).__name__
    return _Failure("invalid_endpoint", hint, type(exc).__name__, None, None)


def _run_client(model: str, key: str, spec: EndpointSpec, timeout_s: float) -> dict[str, Any]:
    client = _build_client(key, model, spec.completion_url, timeout_s)
    if isinstance(client, _Failure):
        return _resolve_row(model, client, spec)
    probe = _probe(client)
    completion = None if probe.failure is not None else _complete(client)
    return _calls_row(model, spec, probe, completion)


def _build_client(key: str, model: str, url: str, timeout_s: float) -> ModelClient | _Failure:
    try:
        return ModelClient(
            api_key=key,
            model=model,
            api_url=url,
            timeout_s=timeout_s,
            retry=_smoke_retry(),
        )
    except Exception as exc:
        return _client_init_failure(exc)


def _smoke_retry() -> RetryPolicy:
    # One attempt: a 429, 500, or timeout must stay visible in the evidence.
    return RetryPolicy(max_attempts=1, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0)


def _probe(client: ModelClient) -> _Call:
    return _timed(client, client.ping)


def _complete(client: ModelClient) -> _Call:
    def action() -> ModelIdentity | None:
        client.complete(
            [{"role": "user", "content": _PROMPT}],
            sampling=SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS, seed=_SEED),
        )
        return client.last_identity

    return _timed(client, action)


def _timed(client: ModelClient, action: Callable[[], ModelIdentity | None]) -> _Call:
    started = time.monotonic()
    try:
        identity = action()
    except ModelError as exc:
        return _Call(_elapsed_ms(started), None, None, _from_model_error(exc))
    except Exception as exc:
        return _Call(_elapsed_ms(started), None, None, _from_unexpected(exc))
    return _Call(_elapsed_ms(started), identity, _copy_usage(client.last_usage), None)


def _from_model_error(exc: ModelError) -> _Failure:
    return _Failure(
        exc.kind.value, exc.hint, exc.summary, exc.status, exc.retry_after_s, exc.request_id
    )


def _from_unexpected(exc: BaseException) -> _Failure:
    hint = "the smoke call failed before a typed model error was available"
    return _Failure("malformed", hint, type(exc).__name__, None, None)


def _elapsed_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000.0, 3)


def _copy_usage(usage: dict[str, int] | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return dict(usage)


def _calls_row(
    model: str, spec: EndpointSpec, probe: _Call, completion: _Call | None
) -> dict[str, Any]:
    row = _base_row(model)
    row["endpoint_host"] = spec.host
    row["probe_ok"] = probe.failure is None
    row["probe_latency_ms"] = probe.latency_ms
    row["probe_usage"] = _copy_usage(probe.usage)
    _apply_identity(row, probe.identity, probe=True)
    completion_ms = _apply_completion(row, completion)
    row["latency_ms"] = round(probe.latency_ms + completion_ms, 3)
    match = _identity_match(probe, completion)
    row["identity_match"] = match
    if match == "mismatch":
        row["served_model"] = _mismatched_served(probe, completion)
    failure = _select_failure(probe, completion, match)
    row["ok"] = failure is None
    row["failed_stage"] = _failed_stage(probe, completion, failure, match)
    if failure is not None:
        _apply_failure(row, failure)
    if row["request_id"] is None:
        row["request_id"] = row["probe_request_id"]
    return _scrub_row(row, spec)


def _apply_completion(row: dict[str, Any], completion: _Call | None) -> float:
    if completion is None:
        return 0.0
    row["completion_ok"] = completion.failure is None
    row["completion_latency_ms"] = completion.latency_ms
    row["usage"] = _copy_usage(completion.usage)
    _apply_identity(row, completion.identity, probe=False)
    return completion.latency_ms


def _resolve_row(model: str, failure: _Failure, spec: EndpointSpec | None = None) -> dict[str, Any]:
    row = _base_row(model)
    row["failed_stage"] = "resolve"
    if spec is not None:
        row["endpoint_host"] = spec.host
    _apply_failure(row, failure)
    return _scrub_row(row, spec)


def _base_row(model: str) -> dict[str, Any]:
    return {
        "completion_latency_ms": 0.0,
        "completion_ok": False,
        "endpoint_host": None,
        "error_kind": None,
        "failed_stage": None,
        "hint": None,
        "identity_match": None,
        "latency_ms": 0.0,
        "ok": False,
        "probe_latency_ms": 0.0,
        "probe_ok": False,
        "probe_request_id": None,
        "probe_usage": None,
        "request_id": None,
        "requested_model": model,
        "response_id": None,
        "retry_after_s": None,
        "served_model": None,
        "status": None,
        "summary": None,
        "system_fingerprint": None,
        "usage": None,
    }


def _apply_identity(row: dict[str, Any], identity: ModelIdentity | None, *, probe: bool) -> None:
    if identity is None:
        return
    if probe:
        row["probe_request_id"] = identity.request_id
    elif identity.request_id is not None:
        row["request_id"] = identity.request_id
    if identity.served is not None:
        row["served_model"] = identity.served
    if identity.system_fingerprint:
        row["system_fingerprint"] = identity.system_fingerprint
    if identity.response_id is not None:
        row["response_id"] = identity.response_id


def _apply_failure(row: dict[str, Any], failure: _Failure) -> None:
    row["error_kind"] = failure.kind
    row["hint"] = failure.hint
    row["summary"] = failure.summary
    row["status"] = failure.status
    row["retry_after_s"] = failure.retry_after_s
    if failure.request_id is not None:
        row["request_id"] = failure.request_id


def _identity_match(probe: _Call, completion: _Call | None) -> str | None:
    verdicts = [_verdict(probe.identity)]
    if completion is not None:
        verdicts.append(_verdict(completion.identity))
    present = [item for item in verdicts if item is not None]
    if not present:
        return None
    if "mismatch" in present:
        return "mismatch"
    if "unreported" in present:
        return "unreported"
    return "match"


def _verdict(identity: ModelIdentity | None) -> str | None:
    if identity is None:
        return None
    if identity.matched is None:
        return "unreported"
    if identity.matched:
        return "match"
    return "mismatch"


def _select_failure(probe: _Call, completion: _Call | None, match: str | None) -> _Failure | None:
    if probe.failure is not None:
        return probe.failure
    if completion is not None and completion.failure is not None:
        return completion.failure
    if match == "mismatch":
        return _mismatch_failure(probe, completion)
    return None


def _failed_stage(
    probe: _Call,
    completion: _Call | None,
    failure: _Failure | None,
    match: str | None,
) -> str | None:
    if failure is None:
        return None
    if probe.failure is not None:
        return "probe"
    if completion is not None and completion.failure is not None:
        return "completion"
    if match == "mismatch":
        return "identity"
    return "probe"


def _mismatch_failure(probe: _Call, completion: _Call | None) -> _Failure:
    served = _mismatched_served(probe, completion)
    requested = _requested(completion) or _requested(probe) or "the requested model"
    hint = f"The server answered as {served}, not {requested}."
    return _Failure("identity_mismatch", hint, "served model id does not match", None, None)


def _mismatched_served(probe: _Call, completion: _Call | None) -> str:
    for call in (completion, probe):
        served = _mismatched_name(call)
        if served is not None:
            return served
    return "an unreported model"


def _mismatched_name(call: _Call | None) -> str | None:
    if call is None or call.identity is None or call.identity.matched is not False:
        return None
    served = _text(call.identity.served)
    if served is not None:
        return served
    return "an unreported model"


def _requested(call: _Call | None) -> str | None:
    if call is None or call.identity is None:
        return None
    return _text(call.identity.requested)


def _text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _scrub_row(row: dict[str, Any], spec: EndpointSpec | None) -> dict[str, Any]:
    cleaned = dict(row)
    for key in _TEXT_FIELDS:
        cleaned[key] = _scrub_value(cleaned.get(key), spec)
    cleaned["probe_usage"] = _scrub_usage(cleaned.get("probe_usage"), spec)
    cleaned["usage"] = _scrub_usage(cleaned.get("usage"), spec)
    return cleaned


def _scrub_value(value: object, spec: EndpointSpec | None) -> object:
    if not isinstance(value, str):
        return value
    return _scrub(value, spec)


def _scrub(text: str, spec: EndpointSpec | None) -> str:
    cleaned = text if spec is None else _strip_endpoint(text, spec)
    cleaned = redact_text(cleaned)
    cleaned = _AUTH_LEFTOVER.sub("[REDACTED]", cleaned)
    return _URL_LEFTOVER.sub("[REDACTED]", cleaned)


def _strip_endpoint(text: str, spec: EndpointSpec) -> str:
    cleaned = text.replace(spec.completion_url, spec.host)
    cleaned = cleaned.replace(spec.origin, spec.host)
    return _strip_port(cleaned, spec)


def _strip_port(text: str, spec: EndpointSpec) -> str:
    port = urllib.parse.urlparse(spec.completion_url).port
    if port is None:
        return text
    cleaned = text.replace(f"[{spec.host}]:{port}", spec.host)
    return cleaned.replace(f"{spec.host}:{port}", spec.host)


def _scrub_usage(value: object, spec: EndpointSpec | None) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    cleaned: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, int) and not isinstance(item, bool):
            cleaned[_scrub(key, spec)] = item
    if not cleaned:
        return None
    return cleaned


def _document(
    rows: list[dict[str, Any]],
    args: _Args,
    started: str,
    finished: str,
    revision: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "completion_request": {
            "max_tokens": _MAX_TOKENS,
            "prompt": _PROMPT,
            "seed": _SEED,
            "temperature": 0.0,
        },
        "disclaimer": _disclaimer(finished),
        "does_not_establish": list(_DOES_NOT),
        "finished_at": finished,
        "generated_at": finished,
        "git_revision": revision,
        "limitations": [],
        "models": rows,
        "ok": all(row.get("ok") is True for row in rows),
        "partial": args.models != MODELS,
        "policy": "allow-partial" if args.allow_partial else "require-both",
        "probe_request": {"max_tokens": 1, "method": "ModelClient.ping", "prompt": "ping"},
        "schema": _SCHEMA,
        "started_at": started,
        "tool": "fx1_model_smoke",
        "tool_version": fx1.__version__,
    }
    frozen = _scrubbed_document(body)
    frozen["content_sha256"] = _content_sha256(frozen)
    return frozen


def _disclaimer(generated: str) -> str:
    return _DISCLAIMER.format(date=generated[:10])


def _content_sha256(document: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "content_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scrubbed_document(body: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    scrubbed = redact_text(encoded)
    if scrubbed == encoded:
        return body
    return _reload_object(scrubbed)


def _reload_object(scrubbed: str) -> dict[str, Any]:
    try:
        loaded = json.loads(scrubbed)
    except json.JSONDecodeError as exc:
        raise RuntimeError("smoke evidence redaction broke the document") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError("smoke evidence redaction broke the document")
    cleaned: dict[str, Any] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise RuntimeError("smoke evidence redaction broke the document")
        cleaned[key] = value
    return cleaned


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "fx1" / f"model_smoke_{stamp}.json"


def _stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _print_report(path: Path, document: Mapping[str, Any]) -> None:
    _emit(_as_text(document.get("disclaimer")))
    _emit(f"evidence {path.resolve()}")
    rows = _rows(document.get("models"))
    if document.get("partial") is True:
        names = ", ".join(_as_text(row.get("requested_model")) for row in rows)
        _emit(f"partial: requested {names}; this is not evidence for both models")
    for row in rows:
        _emit(_status_line(row))
        if row.get("ok") is not True:
            _emit_err(_failure_line(row))


def _rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _status_line(row: Mapping[str, Any]) -> str:
    status = "ok" if row.get("ok") is True else "fail"
    host = row.get("endpoint_host") or "-"
    identity = row.get("identity_match") or "-"
    return f"{row.get('requested_model')} {status} host={host} identity={identity}"


def _failure_line(row: Mapping[str, Any]) -> str:
    kind = row.get("error_kind") or "error"
    hint = row.get("hint") or "the call failed"
    return f"{row.get('requested_model')} {kind}: {hint}"


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _emit(line: str) -> None:
    print(redact_text(line), flush=True)


def _emit_err(line: str) -> None:
    print(redact_text(line), file=sys.stderr, flush=True)


def _git_revision() -> str | None:
    completed = _git_run()
    if completed is None or completed.returncode != 0:
        return None
    revision = completed.stdout.strip().lower()
    if len(revision) == 40 and all(ch in "0123456789abcdef" for ch in revision):
        return revision
    return None


def _git_run() -> subprocess.CompletedProcess[str] | None:
    argv = [
        "git",
        "-c",
        f"safe.directory={_REPO_ROOT}",
        "-C",
        str(_REPO_ROOT),
        "rev-parse",
        "HEAD",
    ]
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.0,
            check=False,
            env=_git_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_env() -> dict[str, str]:
    env: dict[str, str] = {"GIT_TERMINAL_PROMPT": "0"}
    for key in _GIT_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


if __name__ == "__main__":
    sys.exit(main())
