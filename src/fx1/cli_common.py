"""Shared helpers for the ``fx1`` CLI subcommand modules."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

import typer

if TYPE_CHECKING:
    from fx1.sdk import Fx1Harness
    from fx1.serve.client import HarnessClient

_NO_WEIGHTS = "fx-1 harness API; chat ids fx1, fx1-lite; no released weights"
_REMOTE_HELP = (
    "Drive a remote harness API at this base URL (HarnessClient) instead of the in-process SDK."
)
_API_KEY_HELP = "X-API-Key for the remote harness; falls back to FX1_API_KEY."
_TIMEOUT_HELP = "Remote request timeout in seconds."
_BYOK_URL_HELP = "Per-request BYOK endpoint (OpenAI-compatible base URL)."
_BYOK_KEY_HELP = "Per-request BYOK API key."
_BYOK_MODEL_HELP = "Per-request BYOK model name."
_LIMIT_HELP = "Page size (max 100)."
_EVAL_ID_HELP = "eval_ id."
_EVAL_ID_CREATE_HELP = "eval_ id from eval-spec-create."
_EVALRUN_ID_HELP = "evalrun_ id."
_FTJOB_ID_HELP = "ftjob- id from ft-create."
_BACKEND_OVERRIDE_HELP = "Backend link override."
_FALLBACK_HELP = "Alternate backend on availability faults (repeatable)."
_CHAT_ID_HELP = "Stored chat.completion id (chatcmpl-*)."
_RESPONSE_ID_HELP = "Stored response id (resp_*)."
_CONV_ID_HELP = "Conversation id (conv_*)."
_VS_ID_HELP = "Vector store id (vs_*)."
_FILE_ID_HELP = "File record id (file-*)."
_VSFB_ID_HELP = "Batch id (vsfb_*)."
_ORDER_HELP = "asc | desc"
_ITEM_CURSOR_HELP = "Page cursor — an item id."
_METADATA_PAIRS_HELP = "JSON object of string pairs."
_ITEMS_JSON_ERR = "--items must be a JSON array of item dicts"
_METADATA_PAIRS_ERR = "--metadata must be a JSON object of string pairs"
_EVAL_ID_REMOTE_HELP = "Eval id returned by harness eval --remote."
_CALLBACK_SECRET_HELP = "HMAC key signing the webhook delivery."

_POLL_HELP = "Status poll interval, seconds."
_BATCH_ID_HELP = "batch_ id from batch-submit."
_MODEL_ID_OPT_HELP = "Model id — backend name, fx1, or ft:name."
_TOP_P_OPT_HELP = "Nucleus sampling mass."
_BACKEND_HELP = (
    "hosted_k3 | local_fx1 | byok (BYOK reads FX1_BYOK_BASE_URL / "
    "FX1_BYOK_API_KEY / FX1_BYOK_MODEL)"
)
_JUDGE_BACKEND_HELP = (
    "Optional 'hosted_k3' LLM judge for MT-Bench-style grading (needs "
    "MOONSHOT_API_KEY); omit for the deterministic rule-based judge."
)


def _byok_opts(
    base_url: str | None, api_key: str | None, model: str | None
) -> dict[str, str] | None:
    """Pack the three --byok-* flags into the wire override; all-or-none."""
    parts = {"base_url": base_url, "api_key": api_key, "model": model}
    if all(v is None for v in parts.values()):
        return None
    if any(v is None for v in parts.values()):
        typer.echo("--byok-base-url, --byok-api-key and --byok-model go together", err=True)
        raise typer.Exit(2)
    return {k: str(v) for k, v in parts.items()}


def _fx1_headers(
    backend: str | None,
    checkpoint_dir: Path | None,
    byok: dict[str, str] | None,
    fallbacks: list[str],
) -> dict[str, str]:
    """Pack the link override flags into the X-Fx1-* header set the
    in-process SDK path sends — the remote client packs them into the
    ``fx1`` extension object instead, same semantics."""
    headers: dict[str, str] = {}
    if backend is not None:
        headers["x-fx1-backend"] = backend
    if checkpoint_dir is not None:
        headers["x-fx1-checkpoint-dir"] = str(checkpoint_dir)
    if byok is not None:
        headers["x-fx1-byok-base-url"] = byok["base_url"]
        headers["x-fx1-byok-api-key"] = byok["api_key"]
        headers["x-fx1-byok-model"] = byok["model"]
    if fallbacks:
        headers["x-fx1-fallbacks"] = ",".join(fallbacks)
    return headers


def _surface(
    remote: str | None, api_key: str | None, timeout_s: float
) -> Fx1Harness | HarnessClient:
    """In-process SDK by default; HarnessClient when ``--remote`` is given."""
    if remote is None:
        from fx1.sdk import Fx1Harness

        return Fx1Harness()
    from fx1.serve.client import HarnessClient

    return HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )


def _last_request_id(client: Any) -> str | None:
    """``x-request-id`` off the client's last wire response, when it
    reports one — the trace id a bug report would quote. ``None`` for
    clients without a header channel or a transport that saw no
    response."""
    hmap = getattr(client, "last_response_headers", None)
    if not isinstance(hmap, Mapping):
        return None
    rid = hmap.get("x-request-id")
    return rid if isinstance(rid, str) and rid else None


def _or_exit[T](fn: Callable[[], T]) -> T:
    """Run a surface call; faults print one clean line and exit 2."""
    try:
        return fn()
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 — CLI reports the class+message, not a traceback
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _bad_arg(msg: str) -> NoReturn:
    """A flag/value fault — one clean line, exit 2, no traceback."""
    typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(code=2)


def _json_meta(raw: str | None) -> dict[str, str] | None:
    """``--metadata`` JSON — must be an object of string pairs."""
    if raw is None:
        return None
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError:
        _bad_arg(_METADATA_PAIRS_ERR)
        raise AssertionError("unreachable") from None
    if not (
        isinstance(meta, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items())
    ):
        _bad_arg(_METADATA_PAIRS_ERR)
    return meta


def _json_obj_opt(raw: str | None, flag: str) -> dict[str, Any] | None:
    """A JSON-object flag — object or None, clean exit on anything else."""
    if raw is None:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        _bad_arg(f"{flag} must be a JSON object")
    if not isinstance(obj, dict):
        _bad_arg(f"{flag} must be a JSON object")
    return obj


def _json_list_opt(raw: str | None, flag: str) -> list[dict[str, Any]] | None:
    """A JSON list-of-objects flag (testing_criteria entries)."""
    if raw is None:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        _bad_arg(f"{flag} must be a JSON list of objects")
    if not (isinstance(obj, list) and all(isinstance(it, dict) for it in obj)):
        _bad_arg(f"{flag} must be a JSON list of objects")
    return obj


def _response_frame_of(event: Any) -> tuple[str | None, str | None]:
    """One stream frame → ``(delta text, incomplete reason)``."""
    payload = event[1] if isinstance(event, tuple) else event
    if not isinstance(payload, dict):
        return None, None
    reason: str | None = None
    if payload.get("type") == "response.incomplete":
        resp = payload.get("response")
        details = resp.get("incomplete_details") if isinstance(resp, dict) else None
        if isinstance(details, dict) and isinstance(details.get("reason"), str):
            reason = details["reason"]
    if payload.get("type") == "response.failed":
        resp = payload.get("response")
        err = resp.get("error") if isinstance(resp, dict) else None
        reason = (
            f"failed: {err['message']}"
            if isinstance(err, dict) and isinstance(err.get("message"), str)
            else "failed"
        )
    if payload.get("type") == "response.cancelled":
        reason = "cancelled"
    delta = payload.get("delta")
    return delta if isinstance(delta, str) else None, reason


def _emit_response_deltas(events: Iterable[Any]) -> None:
    """Print a Responses event stream's text — every ``*.delta`` frame's
    ``delta`` string (output text and function-call arguments alike),
    nothing else. Accepts bare payload dicts (remote) or ``(event,
    payload)`` pairs (in-process SDK). A ``response.incomplete``/``failed``/
    ``cancelled`` terminal reports its reason on stderr — a quiet text
    stream would look like a full answer."""
    terminal_reason: str | None = None
    for event in events:
        delta, reason = _response_frame_of(event)
        if delta is not None:
            typer.echo(delta, nl=False)
        if reason is not None:
            terminal_reason = reason
    typer.echo()
    if terminal_reason is not None:
        typer.echo(f"[stream ended: {terminal_reason}]", err=True)


def _emit_anthropic_deltas(events: Iterable[Any]) -> None:
    """Print an Anthropic event stream — ``text_delta`` pieces inline,
    a ``tool_use`` block's ``input_json_delta`` into the same stream
    (delimited), nothing else. ``message_stop`` is the terminal frame;
    its absence means the stream was cut (the callers fail on that)."""
    for event in events:
        data = event.get("data", event) if isinstance(event, dict) else event
        payload = data if isinstance(data, dict) else {}
        ptype = payload.get("type")
        if ptype == "content_block_start":
            if (payload.get("content_block") or {}).get("type") == "tool_use":
                typer.echo("[tool_use]", nl=False)
        elif ptype == "content_block_delta":
            delta = payload.get("delta") or {}
            text = (
                delta.get("text")
                if delta.get("type") == "text_delta"
                else delta.get("partial_json")
            )
            if text:
                typer.echo(text, nl=False)
    typer.echo()


harness_app = typer.Typer(help="Inspect/run the dipcatcher harness.")


_OPERATION_ARGUMENT_MAX_BYTES = 2_000_000


def _load_prompts(path: Path) -> list[str]:
    """JSON array of strings or JSONL of str / ``{\"prompt\": ...}`` lines."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path}: empty prompts file")
    if text.startswith("["):
        items = json.loads(text)
    else:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    prompts = [it["prompt"] if isinstance(it, dict) else it for it in items]
    if not all(isinstance(p, str) and p for p in prompts):
        raise ValueError(f"{path}: every prompt must be a non-empty string")
    return prompts


def _record_json(rec: Any) -> dict[str, Any]:
    return {
        "completion_id": rec.completion_id,
        "backend": rec.backend,
        "model": rec.model,
        "ok": rec.ok,
        "latency_ms": rec.latency_ms,
        "at": rec.at,
        "usage": rec.usage,
        "error": rec.error,
        "error_class": rec.error_class,
        "prompt_sha256": rec.prompt_sha256,
        "output_sha256": rec.output_sha256,
    }


def _remote_client(remote: str, api_key: str | None, timeout_s: float) -> HarnessClient:
    from fx1.serve.client import HarnessClient

    return HarnessClient(
        remote,
        api_key=api_key or os.environ.get("FX1_API_KEY") or None,
        timeout_s=timeout_s,
    )


def _need_remote(remote: str | None) -> None:
    if remote is None:
        typer.echo(
            "error: async jobs are a wire surface; pass --remote",
            err=True,
        )
        raise typer.Exit(code=2)


def _emit_completion_deltas(chunks: list[dict[str, Any]]) -> None:
    """Print the legacy text deltas — the terminal frame carries no text,
    so ordering is by frame emission, not ``choices[i].index``."""
    for chunk in chunks:
        for choice in cast(list[dict[str, Any]], chunk.get("choices") or []):
            text = choice.get("text")
            if isinstance(text, str) and text:
                typer.echo(text, nl=False)
    typer.echo()
