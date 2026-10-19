"""``fx1 harness`` chat, response, message, and conversation commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

if TYPE_CHECKING:
    pass

from fx1.cli_common import (
    _API_KEY_HELP,
    _BACKEND_OVERRIDE_HELP,
    _BATCH_ID_HELP,
    _BYOK_KEY_HELP,
    _BYOK_MODEL_HELP,
    _BYOK_URL_HELP,
    _CALLBACK_SECRET_HELP,
    _CHAT_ID_HELP,
    _CONV_ID_HELP,
    _FALLBACK_HELP,
    _ITEM_CURSOR_HELP,
    _ITEMS_JSON_ERR,
    _LIMIT_HELP,
    _METADATA_PAIRS_ERR,
    _METADATA_PAIRS_HELP,
    _MODEL_ID_OPT_HELP,
    _ORDER_HELP,
    _POLL_HELP,
    _REMOTE_HELP,
    _RESPONSE_ID_HELP,
    _TIMEOUT_HELP,
    _TOP_P_OPT_HELP,
    _bad_arg,
    _byok_opts,
    _emit_anthropic_deltas,
    _emit_completion_deltas,
    _emit_response_deltas,
    _fx1_headers,
    _json_meta,
    _need_remote,
    _or_exit,
    _remote_client,
    harness_app,
)


@harness_app.command("respond")
def harness_respond(  # NOSONAR
    input_: str = typer.Argument(..., help="Input string, or a JSON array of Responses items."),
    model: str = typer.Option("fx1", "--model", help=_MODEL_ID_OPT_HELP),
    instructions: str | None = typer.Option(
        None, "--instructions", help="Prepended system-level instructions."
    ),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option([], "--fallback", help=_FALLBACK_HELP),
    temperature: float | None = typer.Option(None, "--temperature", help="Decode temperature."),
    top_p: float | None = typer.Option(None, "--top-p", help=_TOP_P_OPT_HELP),
    max_output_tokens: int | None = typer.Option(
        None, "--max-output-tokens", help="Output token cap."
    ),
    metadata: str | None = typer.Option(None, "--metadata", help=_METADATA_PAIRS_HELP),
    text_format: str | None = typer.Option(
        None, "--format", help='text.format JSON, e.g. \'{"type":"json_object"}\'.'
    ),
    tools: str | None = typer.Option(None, "--tools", help="JSON array of Responses tools."),
    tool_choice: str | None = typer.Option(
        None, "--tool-choice", help='"none"/"auto"/"required" or a JSON choice object.'
    ),
    max_tool_calls: int | None = typer.Option(
        None,
        "--max-tool-calls",
        help="Cap the function calls one response may carry — over the cap the "
        "turn truncates to status='incomplete'.",
    ),
    verbosity: str | None = typer.Option(
        None, "--verbosity", help="Output verbosity hint: low|medium|high."
    ),
    prompt_cache_key: str | None = typer.Option(
        None, "--prompt-cache-key", help="Provider prompt-cache key hint."
    ),
    prompt_cache_retention: str | None = typer.Option(
        None, "--prompt-cache-retention", help="in-memory|24h."
    ),
    stream: bool = typer.Option(
        False, "--stream", help="Emit Responses event deltas instead of one JSON block."
    ),
    previous_response_id: str | None = typer.Option(
        None,
        "--previous-response-id",
        help="Chain onto a stored response (resp_…) — the turn runs with the parent history.",
    ),
    conversation: str | None = typer.Option(
        None,
        "--conversation",
        help="Join a conversation container (conv_…) — its items are the turn's context.",
    ),
    background: bool = typer.Option(
        False,
        "--background",
        help="Queue the response and return a status='queued' object — poll "
        "with `response-get`, cancel with `response-cancel`.",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
) -> None:
    """``POST /v1/responses`` — the Responses API over the gated pipeline.

    The response object prints verbatim (``output[0]`` is the message
    item); a completion-log id rides ``X-Fx1-Completion-Id`` for receipt
    lookup. In-process by default — the SDK twin runs the same request
    model, link resolution, and gate.
    """
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)

    def _json_opt(raw: str | None, what: str) -> Any:
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            _bad_arg(f"invalid {what} JSON")
            raise AssertionError("unreachable") from None

    try:
        parsed = json.loads(input_)
        items: str | list[Any] = parsed if isinstance(parsed, list) else input_
    except json.JSONDecodeError:
        items = input_
    meta = _json_opt(metadata, "metadata")
    if meta is not None and not (
        isinstance(meta, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items())
    ):
        _bad_arg(_METADATA_PAIRS_ERR)
    tfmt = _json_opt(text_format, "format")
    tool_list = _json_opt(tools, "tools")
    if tool_list is not None and not isinstance(tool_list, list):
        _bad_arg("--tools must be a JSON array")
    tchoice: str | dict[str, Any] | None = None
    if tool_choice is not None:
        if tool_choice in {"none", "auto", "required"}:
            tchoice = tool_choice
        else:
            tc = _json_opt(tool_choice, "tool-choice")
            if not isinstance(tc, dict):
                _bad_arg("--tool-choice must be none/auto/required or a JSON object")
            tchoice = tc

    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        if stream:
            events, _cid = _or_exit(
                lambda: client.responses_create_stream(
                    items,
                    model=model,
                    instructions=instructions,
                    backend=backend,
                    byok=byok,
                    checkpoint_dir=checkpoint_dir,
                    fallbacks=fallbacks or None,
                    timeout_s=backend_timeout,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    metadata=meta,
                    text_format=tfmt,
                    tools=tool_list,
                    tool_choice=tchoice,
                    previous_response_id=previous_response_id,
                    conversation=conversation,
                    max_tool_calls=max_tool_calls,
                    verbosity=verbosity,
                    prompt_cache_key=prompt_cache_key,
                    prompt_cache_retention=prompt_cache_retention,
                )
            )
            _emit_response_deltas(events)
            return
        resp, _cid = _or_exit(
            lambda: client.responses_create(
                items,
                model=model,
                instructions=instructions,
                backend=backend,
                byok=byok,
                checkpoint_dir=checkpoint_dir,
                fallbacks=fallbacks or None,
                timeout_s=backend_timeout,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                metadata=meta,
                text_format=tfmt,
                tools=tool_list,
                tool_choice=tchoice,
                previous_response_id=previous_response_id,
                conversation=conversation,
                max_tool_calls=max_tool_calls,
                verbosity=verbosity,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
                background=background,
            )
        )
        typer.echo(json.dumps(resp, indent=2))
        return
    from fx1.sdk import Fx1Harness

    headers = _fx1_headers(backend, checkpoint_dir, byok, fallbacks)
    fx1: dict[str, Any] = {}
    if backend_timeout is not None:
        fx1["timeout_s"] = backend_timeout
    body: dict[str, Any] = {
        "model": model,
        "input": items,
        "instructions": instructions,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "metadata": meta,
        "tools": tool_list,
        "tool_choice": tchoice,
        "previous_response_id": previous_response_id,
        "conversation": conversation,
        "max_tool_calls": max_tool_calls,
        "prompt_cache_key": prompt_cache_key,
        "prompt_cache_retention": prompt_cache_retention,
        "background": background,
        "fx1": fx1 or None,
    }
    if tfmt is not None or verbosity is not None:
        text: dict[str, Any] = {}
        if tfmt is not None:
            text["format"] = tfmt
        if verbosity is not None:
            text["verbosity"] = verbosity
        body["text"] = text
    if stream:
        sevents, _cid = _or_exit(lambda: Fx1Harness().openai_response_stream(body, headers=headers))
        _emit_response_deltas(sevents)
        return
    resp, _cid = _or_exit(lambda: Fx1Harness().openai_response(body, headers=headers))
    typer.echo(json.dumps(resp, indent=2))


@harness_app.command("message")
def harness_message(  # NOSONAR
    input_: str = typer.Argument(
        ...,
        help="User-turn text, or a JSON array of Anthropic message objects.",
    ),
    model: str = typer.Option("fx1", "--model", help=_MODEL_ID_OPT_HELP),
    max_tokens: int = typer.Option(
        1024, "--max-tokens", help="Output token cap (required by the contract)."
    ),
    system: str | None = typer.Option(
        None,
        "--system",
        help="System prompt text, or a JSON array of {type:'text',text} blocks.",
    ),
    temperature: float | None = typer.Option(None, "--temperature", help="Decode temperature 0–1."),
    top_p: float | None = typer.Option(None, "--top-p", help=_TOP_P_OPT_HELP),
    stop: list[str] = typer.Option([], "--stop", help="Stop sequence (repeatable, at most 4)."),
    tools: str | None = typer.Option(
        None, "--tools", help="JSON array of Anthropic tools ({name, description, input_schema})."
    ),
    tool_choice: str | None = typer.Option(
        None,
        "--tool-choice",
        help='JSON choice object, e.g. \'{"type":"auto"}\' or \'{"type":"tool","name":"x"}\'.',
    ),
    user_id: str | None = typer.Option(None, "--user-id", help="metadata.user_id audit stamp."),
    stream: bool = typer.Option(
        False, "--stream", help="Emit Anthropic SSE event deltas instead of one JSON block."
    ),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option([], "--fallback", help=_FALLBACK_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
) -> None:
    """``POST /v1/messages`` — the Anthropic Messages surface over the gated
    pipeline.

    Anthropic turns (``user``/``assistant``, string or block-list content),
    ``--system``, tools in the Anthropic ``input_schema`` shape. The answer
    prints as the ``message`` object; ``--stream`` prints text deltas. The
    completion-log id rides ``X-Fx1-Completion-Id`` for receipt lookup.
    Anthropic-only knobs the pipeline cannot honor (top_k, thinking,
    cache_control, image blocks) fail closed — never silently dropped.
    """
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)

    def _json_opt(raw: str | None, what: str) -> Any:
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            _bad_arg(f"invalid {what} JSON")
            raise AssertionError("unreachable") from None

    try:
        parsed = json.loads(input_)
        messages: list[dict[str, Any]] = (
            cast(list[dict[str, Any]], parsed)
            if isinstance(parsed, list)
            else [{"role": "user", "content": input_}]
        )
    except json.JSONDecodeError:
        messages = [{"role": "user", "content": input_}]
    sys_val: str | list[Any] | None = None
    if system is not None:
        try:
            parsed_sys = json.loads(system)
        except json.JSONDecodeError:
            sys_val = system
        else:
            sys_val = parsed_sys if isinstance(parsed_sys, list) else system
    tool_list = _json_opt(tools, "tools")
    if tool_list is not None and not isinstance(tool_list, list):
        _bad_arg("--tools must be a JSON array")
    tchoice = _json_opt(tool_choice, "tool-choice")
    if tchoice is not None and not isinstance(tchoice, dict):
        _bad_arg('--tool-choice must be a JSON object, e.g. {"type":"auto"}')

    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        if stream:
            events, _cid = _or_exit(
                lambda: client.create_message_stream(
                    messages,
                    model=model,
                    max_tokens=max_tokens,
                    system=sys_val,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop or None,
                    tools=tool_list,
                    tool_choice=tchoice,
                    user_id=user_id,
                    backend=backend,
                    byok=byok,
                    checkpoint_dir=checkpoint_dir,
                    fallbacks=fallbacks or None,
                    timeout_s=backend_timeout,
                )
            )
            _emit_anthropic_deltas(events)
            return
        msg_remote, _cid = _or_exit(
            lambda: client.create_message(
                messages,
                model=model,
                max_tokens=max_tokens,
                system=sys_val,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop or None,
                tools=tool_list,
                tool_choice=tchoice,
                user_id=user_id,
                backend=backend,
                byok=byok,
                checkpoint_dir=checkpoint_dir,
                fallbacks=fallbacks or None,
                timeout_s=backend_timeout,
            )
        )
        typer.echo(json.dumps(msg_remote, indent=2))
        return
    from fx1.sdk import Fx1Harness

    headers = _fx1_headers(backend, checkpoint_dir, byok, fallbacks)
    fx1: dict[str, Any] = {}
    if backend_timeout is not None:
        fx1["timeout_s"] = backend_timeout
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "system": sys_val,
        "temperature": temperature,
        "top_p": top_p,
        "stop_sequences": stop or None,
        "tools": tool_list,
        "tool_choice": tchoice,
        "metadata": {"user_id": user_id} if user_id is not None else None,
        "fx1": fx1 or None,
    }
    if stream:
        sevents, _cid = _or_exit(
            lambda: Fx1Harness().anthropic_message_stream(body, headers=headers)
        )
        _emit_anthropic_deltas(sevents)
        return
    amsg, _cid = _or_exit(lambda: Fx1Harness().anthropic_message(body, headers=headers))
    typer.echo(json.dumps(amsg.model_dump(mode="json"), indent=2))


@harness_app.command("message-batch")
def harness_message_batch(
    input_file: Path = typer.Argument(
        ..., help="Batch input .jsonl — {custom_id, params} per line (params = /v1/messages body)."
    ),
    callback_url: str | None = typer.Option(
        None, "--callback-url", help="Terminal webhook URL (POSTs the batch once)."
    ),
    callback_secret: str | None = typer.Option(
        None, "--callback-secret", help=_CALLBACK_SECRET_HELP
    ),
    idem_key: str | None = typer.Option(
        None, "--idem-key", help="Idempotency-Key — replays the submit envelope."
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Submit and return immediately (remote leg only)."
    ),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option([], "--fallback", help=_FALLBACK_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/messages/batches`` — submit an Anthropic message batch.

    Each line of ``input_file`` is ``{custom_id, params}`` — ``params`` a
    full ``/v1/messages`` body (``stream`` inside a batch refuses at
    validation). With ``--remote`` the submit runs async server-side and
    polls to ``ended``; in-process it runs synchronously and prints the
    ended batch plus its result rows. The submitter's backend choice
    (``--backend``/BYOK/``--fallback``) applies to every item."""
    body = input_file.read_text(encoding="utf-8") if input_file.exists() else None
    if body is None:
        typer.echo(f"error: {input_file} does not exist", err=True)
        raise typer.Exit(code=2)
    try:
        requests = [json.loads(line) for line in body.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        typer.echo(f"error: {input_file} is not valid JSONL: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not requests:
        typer.echo("error: batch input is empty", err=True)
        raise typer.Exit(code=2)
    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        headers = _fx1_headers(
            backend, checkpoint_dir, _byok_opts(byok_base_url, byok_api_key, byok_model), fallbacks
        )
        sub = _or_exit(
            lambda: client.create_message_batch(
                requests,
                callback_url=callback_url,
                callback_secret=callback_secret,
                idempotency_key=idem_key,
                extra_headers=headers or None,
            )
        )
        if no_wait:
            typer.echo(json.dumps(sub, indent=2))
            return
        fin = _or_exit(lambda: client.wait_message_batch(sub["id"]))
        typer.echo(json.dumps(fin, indent=2))
        return
    from fx1.sdk import Fx1Harness

    headers = _fx1_headers(
        backend, checkpoint_dir, _byok_opts(byok_base_url, byok_api_key, byok_model), fallbacks
    )
    batch, rows = _or_exit(
        lambda: Fx1Harness().anthropic_batch(
            requests,
            headers=headers,
            callback_url=callback_url,
            callback_secret=callback_secret,
        )
    )
    typer.echo(json.dumps({"batch": batch, "results": rows}, indent=2))


@harness_app.command("message-batches")
def harness_message_batches(
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    limit: int = typer.Option(20, "--limit", help=_LIMIT_HELP),
    after_id: str | None = typer.Option(
        None, "--after-id", help="Cursor — entries newer than this batch id."
    ),
    before_id: str | None = typer.Option(
        None, "--before-id", help="Cursor — entries older than this batch id."
    ),
) -> None:
    """List remote Anthropic message batches, newest first."""
    _need_remote(remote)
    page = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).message_batches(
            limit=limit, after_id=after_id, before_id=before_id
        )
    )
    typer.echo(json.dumps(page, indent=2))


@harness_app.command("message-batch-status")
def harness_message_batch_status(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Print a remote message batch's record (``processing_status`` +
    ``request_counts`` — counts stay all-processing until the batch ends)."""
    _need_remote(remote)
    st = _or_exit(lambda: _remote_client(remote or "", api_key, timeout_s).message_batch(batch_id))
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("message-batch-wait")
def harness_message_batch_wait(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
    poll_s: float = typer.Option(0.5, "--poll", help=_POLL_HELP),
    wait_timeout_s: float | None = typer.Option(
        None, "--wait-timeout", help="Give up waiting after N seconds (batch keeps running)."
    ),
) -> None:
    """Re-attach to a ``--no-wait`` message batch and poll to ``ended``."""
    _need_remote(remote)
    rec = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).wait_message_batch(
            batch_id, poll_s=poll_s, timeout_s=wait_timeout_s
        )
    )
    typer.echo(json.dumps(rec, indent=2))


@harness_app.command("message-batch-cancel")
def harness_message_batch_cancel(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Cooperative cancel — flips to ``canceling``; in-flight items
    complete and the rest land ``canceled`` result rows."""
    _need_remote(remote)
    st = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).cancel_message_batch(batch_id)
    )
    typer.echo(json.dumps(st, indent=2))


@harness_app.command("message-batch-results")
def harness_message_batch_results(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    out: Path | None = typer.Option(
        None, "--out", help="Write the results JSONL to this path instead of stdout."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(60.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Fetch an ended message batch's results — one ``{custom_id,
    result}`` row per request (``succeeded``/``errored``/``canceled``/
    ``expired``). Before the batch ends the wire 400s."""
    _need_remote(remote)
    rows = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).message_batch_results(batch_id)
    )
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if out is not None:
        out.write_text(text, encoding="utf-8")
        typer.echo(json.dumps({"batch_id": batch_id, "rows": len(rows), "out": str(out)}))
        return
    typer.echo(text, nl=False)


@harness_app.command("message-batch-delete")
def harness_message_batch_delete(
    batch_id: str = typer.Argument(..., help=_BATCH_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """Tombstone an ended message batch — deletes its stored rows.
    A batch that isn't ended refuses (Anthropic's contract)."""
    _need_remote(remote)
    out = _or_exit(
        lambda: _remote_client(remote or "", api_key, timeout_s).delete_message_batch(batch_id)
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("message-tokens")
def harness_message_tokens(
    request_file: Path = typer.Argument(
        ..., help="JSON file: a /v1/messages-count_tokens body — {model?, messages, system?}."
    ),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(60.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/messages/count_tokens`` — the provider's own input
    count, ``{"input_tokens": N}``.

    The request body is the ``/v1/messages`` shape minus ``max_tokens``
    (and minus ``stream``); ``tools``/``tool_choice`` refuse — the
    tokenize channel sees only messages, so counting a toolful request
    would undercount. A backend without a tokenize route fails closed
    (exit 2, the wire's 501) — the harness never estimates."""
    body_text = request_file.read_text(encoding="utf-8") if request_file.exists() else None
    if body_text is None:
        typer.echo(f"error: {request_file} does not exist", err=True)
        raise typer.Exit(code=2)
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError as exc:
        typer.echo(f"error: {request_file} is not valid JSON: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not isinstance(body, dict):
        typer.echo(f"error: {request_file} must hold one JSON object", err=True)
        raise typer.Exit(code=2)
    headers = _fx1_headers(
        backend, checkpoint_dir, _byok_opts(byok_base_url, byok_api_key, byok_model), []
    )
    if remote is not None:
        n = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).count_message_tokens(
                body, extra_headers=headers or None
            )
        )
        typer.echo(json.dumps({"input_tokens": n}, indent=2))
        return
    from fx1.sdk import Fx1Harness

    n = _or_exit(lambda: Fx1Harness().anthropic_count_tokens(body, headers=headers or None))
    typer.echo(json.dumps({"input_tokens": n}, indent=2))


@harness_app.command("text-completion")
def harness_text_completion(  # NOSONAR
    prompt: str = typer.Argument(  # NOSONAR(S107) — sonar anchors S107 at the first param line
        ...,
        help="Prompt text, or a JSON array of prompt strings.",
    ),
    model: str = typer.Option("fx1", "--model", help=_MODEL_ID_OPT_HELP),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Output token cap (server default 16)."
    ),
    temperature: float | None = typer.Option(None, "--temperature", help="Decode temperature 0–1."),
    top_p: float | None = typer.Option(None, "--top-p", help=_TOP_P_OPT_HELP),
    n: int = typer.Option(1, "--n", help="Choice count per prompt element (≤8)."),
    stop: list[str] = typer.Option([], "--stop", help="Stop sequence (repeatable, at most 4)."),
    seed: int | None = typer.Option(None, "--seed", help="Deterministic decode seed."),
    echo: bool = typer.Option(
        False, "--echo", help="Prepend the prompt text to each choice (legacy FIM-style echo)."
    ),
    stream: bool = typer.Option(
        False, "--stream", help="Emit legacy text-completion SSE deltas instead of one JSON block."
    ),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option([], "--fallback", help=_FALLBACK_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
) -> None:
    """``POST /v1/completions`` — the legacy ``text_completion`` surface
    (what ``client.completions.create`` and pre-chat agents target).

    ``prompt`` takes a string or a JSON array of prompt strings; each
    element runs the gated pipeline independently and ``--n`` repeats
    within an element — a two-prompt ``--n 2`` call lands four flat
    choices. ``--stream`` prints the legacy chunk grammar's text deltas.
    ``suffix``/``best_of``/``logprobs`` are unsupported — the server
    fails them closed 422 rather than silently dropping them.
    """
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    try:
        parsed = json.loads(prompt)
        prompt_val: str | list[str] = (
            cast(list[str], parsed)
            if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed)
            else prompt
        )
    except json.JSONDecodeError:
        prompt_val = prompt

    if remote is not None:
        client = _remote_client(remote, api_key, timeout_s)
        if stream:
            chunks, _cid = _or_exit(
                lambda: client.create_completion_stream(
                    prompt_val,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop or None,
                    seed=seed,
                    echo=echo,
                    backend=backend,
                    byok=byok,
                    checkpoint_dir=checkpoint_dir,
                    fallbacks=fallbacks or None,
                    timeout_s=backend_timeout,
                )
            )
            _emit_completion_deltas(chunks)
            return
        env, _cid = _or_exit(
            lambda: client.create_completion(
                prompt_val,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop or None,
                seed=seed,
                echo=echo,
                backend=backend,
                byok=byok,
                checkpoint_dir=checkpoint_dir,
                fallbacks=fallbacks or None,
                timeout_s=backend_timeout,
            )
        )
        typer.echo(json.dumps(env, indent=2))
        return
    from fx1.sdk import Fx1Harness

    headers = _fx1_headers(backend, checkpoint_dir, byok, fallbacks)
    fx1: dict[str, Any] = {}
    if backend_timeout is not None:
        fx1["timeout_s"] = backend_timeout
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt_val,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stop": stop or None,
        "seed": seed,
        "echo": echo,
        "fx1": fx1 or None,
    }
    if stream:
        schunks, _cid = _or_exit(
            lambda: Fx1Harness().openai_completion_stream({**body, "stream": True}, headers=headers)
        )
        _emit_completion_deltas(schunks)
        return
    cenv, _cid = _or_exit(lambda: Fx1Harness().openai_completion(body, headers=headers))
    typer.echo(json.dumps(cenv, indent=2))


@harness_app.command("embed")
def harness_embed(
    inputs: list[str] = typer.Argument(..., help="Text to embed (repeatable)."),
    model: str = typer.Option("fx1", "--model", help="Embedding model id."),
    backend: str | None = typer.Option(None, "--backend", help=_BACKEND_OVERRIDE_HELP),
    checkpoint_dir: Path | None = typer.Option(None, help="For local_fx1."),
    byok_base_url: str | None = typer.Option(None, "--byok-base-url", help=_BYOK_URL_HELP),
    byok_api_key: str | None = typer.Option(None, "--byok-api-key", help=_BYOK_KEY_HELP),
    byok_model: str | None = typer.Option(None, "--byok-model", help=_BYOK_MODEL_HELP),
    fallbacks: list[str] = typer.Option([], "--fallback", help=_FALLBACK_HELP),
    encoding_format: str | None = typer.Option(None, "--encoding", help='"float" | "base64".'),
    dimensions: int | None = typer.Option(None, "--dimensions", help="Output dimensions."),
    user: str | None = typer.Option(None, "--user", help="End-user tag."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(120.0, "--timeout", help=_TIMEOUT_HELP),
    backend_timeout: float | None = typer.Option(
        None, "--backend-timeout", help="Per-call backend deadline in seconds."
    ),
) -> None:
    """``POST /v1/embeddings`` — vectors through the gated pipeline; a link
    without the channel answers a wire error, never fabricated floats.
    In-process by default."""
    byok = _byok_opts(byok_base_url, byok_api_key, byok_model)
    inp: str | list[str] = inputs[0] if len(inputs) == 1 else list(inputs)
    if remote is not None:
        env, _cid = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).embeddings_create(
                inp,
                model=model,
                backend=backend,
                byok=byok,
                checkpoint_dir=checkpoint_dir,
                fallbacks=fallbacks or None,
                timeout_s=backend_timeout,
                encoding_format=encoding_format,
                dimensions=dimensions,
                user=user,
            )
        )
        typer.echo(json.dumps(env, indent=2))
        return
    from fx1.sdk import Fx1Harness

    headers = _fx1_headers(backend, checkpoint_dir, byok, fallbacks)
    fx1: dict[str, Any] = {}
    if backend_timeout is not None:
        fx1["timeout_s"] = backend_timeout
    body: dict[str, Any] = {
        "model": model,
        "input": inp,
        "encoding_format": encoding_format,
        "dimensions": dimensions,
        "user": user,
        "fx1": fx1 or None,
    }
    env, _cid = _or_exit(lambda: Fx1Harness().openai_embeddings(body, headers=headers))
    typer.echo(json.dumps(env, indent=2))


@harness_app.command("moderate")
def harness_moderate(
    inputs: list[str] = typer.Argument(..., help="Text to classify (repeatable)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/moderations`` — the honesty gate as an OpenAI-moderations
    verdict; flagged inputs carry per-category scores. Advisory — no
    backend needed. In-process by default."""
    inp: str | list[str] = inputs[0] if len(inputs) == 1 else list(inputs)
    if remote is not None:
        out = _or_exit(lambda: _remote_client(remote, api_key, timeout_s).moderate(inp))
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().moderate(inp))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("chat-get")
def harness_chat_get(
    completion_id: str = typer.Argument(..., help=_CHAT_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/chat/completions/{id}`` — the stored envelope; missing ids
    (evicted, deleted, ``store: false``) exit 2, never a fabricated object."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).retrieve_chat_completion(
                completion_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_chat_get(completion_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("chat-update")
def harness_chat_update(
    completion_id: str = typer.Argument(..., help=_CHAT_ID_HELP),
    metadata_json: str | None = typer.Option(
        None, "--metadata", help="Replacement metadata as a JSON object of string pairs."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/chat/completions/{id}`` — replace the stored completion's
    metadata wholesale (the only mutable field); unknown ids exit 2."""
    metadata = _json_meta(metadata_json) if metadata_json is not None else {}
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).update_chat_completion(
                completion_id, metadata=metadata
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_chat_update(completion_id, metadata=metadata))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("chat-delete")
def harness_chat_delete(
    completion_id: str = typer.Argument(..., help="Stored chat.completion id."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/chat/completions/{id}`` — drop the stored envelope."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).delete_chat_completion(completion_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_chat_delete(completion_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("response-get")
def harness_response_get(
    response_id: str = typer.Argument(..., help=_RESPONSE_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/responses/{id}`` — the stored response object; missing ids
    exit 2, never a fabricated object."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).retrieve_response(response_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_response_get(response_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("response-delete")
def harness_response_delete(
    response_id: str = typer.Argument(..., help=_RESPONSE_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/responses/{id}`` — drop the stored envelope."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).delete_response(response_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_response_delete(response_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("response-cancel")
def harness_response_cancel(
    response_id: str = typer.Argument(..., help="Background response id (resp_*)."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/responses/{id}/cancel`` — cancel a queued or in-progress
    background response. Terminal responses exit 2 (``cancel_terminal``),
    unknown ids exit 2 (``not_found``)."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).cancel_response(response_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_response_cancel(response_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("response-replay")
def harness_response_replay(
    response_id: str = typer.Argument(..., help=_RESPONSE_ID_HELP),
    starting_after: int | None = typer.Option(
        None,
        "--starting-after",
        help="Resume past sequence N — only events whose sequence number "
        "exceeds N (the frames' id: cursor).",
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/responses/{id}?stream=true`` — replay the stored response
    as the Responses SSE event stream: the full grammar for a terminal
    response, or the prelude plus a live follow for a still-running
    ``background:true`` call. Prints the text-delta channel like
    ``respond --stream``; unknown/deleted ids exit 2 (``not_found``)."""
    if remote is not None:
        revents, _cid = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).responses_replay(
                response_id, starting_after=starting_after, timeout_s=timeout_s
            )
        )
        _emit_response_deltas(revents)
        return
    from fx1.sdk import Fx1Harness

    sevents, _cid = _or_exit(
        lambda: Fx1Harness().openai_response_replay(
            response_id, starting_after=starting_after, timeout_s=timeout_s
        )
    )
    _emit_response_deltas(sevents)


@harness_app.command("chat-list")
def harness_chat_list(
    model: str | None = typer.Option(None, "--model", help="Filter to this model id."),
    metadata: list[str] = typer.Option(
        [], "--metadata", help="Exact-match filter, repeatable: --metadata key=value"
    ),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help="Page cursor — a completion id."),
    before: str | None = typer.Option(None, "--before", help="Page cursor — a completion id."),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/chat/completions`` — stored completions, filtered by
    model and/or a ``--metadata key=value`` subset, paged by id."""
    meta: dict[str, str] = {}
    for pair in metadata:
        if "=" not in pair:
            typer.echo(f"error: --metadata expects key=value, got {pair!r}", err=True)
            raise typer.Exit(code=2)
        k, v = pair.split("=", 1)
        meta[k] = v
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).list_chat_completions(
                model=model,
                metadata=meta or None,
                limit=limit,
                after=after,
                before=before,
                order=order,
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().openai_chat_list(
            model=model,
            metadata=meta or None,
            limit=limit,
            after=after,
            before=before,
            order=order,
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("chat-messages")
def harness_chat_messages(
    completion_id: str = typer.Argument(..., help=_CHAT_ID_HELP),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help=_ITEM_CURSOR_HELP),
    before: str | None = typer.Option(None, "--before", help=_ITEM_CURSOR_HELP),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/chat/completions/{id}/messages`` — the request messages a
    stored completion ran on; missing ids exit 2."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).chat_completion_messages(
                completion_id, limit=limit, after=after, before=before, order=order
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().openai_chat_messages(
            completion_id, limit=limit, after=after, before=before, order=order
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("response-input-items")
def harness_response_input_items(
    response_id: str = typer.Argument(..., help=_RESPONSE_ID_HELP),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help=_ITEM_CURSOR_HELP),
    before: str | None = typer.Option(None, "--before", help=_ITEM_CURSOR_HELP),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/responses/{id}/input_items`` — the ``input`` items a
    stored response ran on; missing ids exit 2."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).response_input_items(
                response_id, limit=limit, after=after, before=before, order=order
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().openai_response_input_items(
            response_id, limit=limit, after=after, before=before, order=order
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-create")
def harness_conv_create(
    items: str | None = typer.Option(
        None, "--items", help="JSON array of seed item dicts (message items)."
    ),
    metadata: str | None = typer.Option(None, "--metadata", help=_METADATA_PAIRS_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/conversations`` — mint a ``conv_*`` container a response
    joins via ``--conversation``. Prints the object (``id`` is the join
    handle)."""
    seed: list[Any] | None = None
    if items is not None:
        try:
            parsed = json.loads(items)
        except json.JSONDecodeError:
            _bad_arg(_ITEMS_JSON_ERR)
            raise AssertionError("unreachable") from None
        if not isinstance(parsed, list):
            _bad_arg(_ITEMS_JSON_ERR)
        seed = parsed
    meta = _json_meta(metadata)
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_create(
                items=seed, metadata=meta
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_create(items=seed, metadata=meta))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-get")
def harness_conv_get(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/conversations/{id}`` — the conversation object; missing
    ids exit 2."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_get(conversation_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_get(conversation_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-update")
def harness_conv_update(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    metadata: str | None = typer.Option(
        None, "--metadata", help="JSON object of string pairs — replaces wholesale."
    ),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/conversations/{id}`` — set the conv's metadata (an omitted
    flag clears it)."""
    meta = _json_meta(metadata)
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_update(
                conversation_id, metadata=meta
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_update(conversation_id, metadata=meta))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-delete")
def harness_conv_delete(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/conversations/{id}`` — drop the container and its
    items; member responses stay retrievable on their own ids."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_delete(conversation_id)
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_delete(conversation_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-items")
def harness_conv_items(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    after: str | None = typer.Option(None, "--after", help=_ITEM_CURSOR_HELP),
    before: str | None = typer.Option(None, "--before", help=_ITEM_CURSOR_HELP),
    order: str = typer.Option("asc", "--order", help=_ORDER_HELP),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/conversations/{id}/items`` — the conv's accumulated
    items, paged by item id."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_items(
                conversation_id, limit=limit, after=after, before=before, order=order
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(
        lambda: Fx1Harness().openai_conversation_items(
            conversation_id, limit=limit, after=after, before=before, order=order
        )
    )
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-items-add")
def harness_conv_items_add(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    items: str = typer.Option(..., "--items", help="JSON array of item dicts to append."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``POST /v1/conversations/{id}/items`` — append item dicts; prints
    the minted items list."""
    try:
        parsed = json.loads(items)
    except json.JSONDecodeError:
        _bad_arg(_ITEMS_JSON_ERR)
        raise AssertionError("unreachable") from None
    if not isinstance(parsed, list):
        _bad_arg(_ITEMS_JSON_ERR)
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_items_add(
                conversation_id, parsed
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_items_add(conversation_id, parsed))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-item")
def harness_conv_item(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    item_id: str = typer.Argument(..., help="Item id inside the conv."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``GET /v1/conversations/{id}/items/{item_id}`` — one item by id;
    prints the item."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_item(
                conversation_id, item_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_item(conversation_id, item_id))
    typer.echo(json.dumps(out, indent=2))


@harness_app.command("conv-items-delete")
def harness_conv_item_delete(
    conversation_id: str = typer.Argument(..., help=_CONV_ID_HELP),
    item_id: str = typer.Argument(..., help="Item id inside the conv."),
    remote: str | None = typer.Option(None, "--remote", help=_REMOTE_HELP),
    api_key: str | None = typer.Option(None, "--api-key", help=_API_KEY_HELP),
    timeout_s: float = typer.Option(30.0, "--timeout", help=_TIMEOUT_HELP),
) -> None:
    """``DELETE /v1/conversations/{id}/items/{item_id}`` — drop one item;
    prints the conv object."""
    if remote is not None:
        out = _or_exit(
            lambda: _remote_client(remote, api_key, timeout_s).conversation_item_delete(
                conversation_id, item_id
            )
        )
        typer.echo(json.dumps(out, indent=2))
        return
    from fx1.sdk import Fx1Harness

    out = _or_exit(lambda: Fx1Harness().openai_conversation_item_delete(conversation_id, item_id))
    typer.echo(json.dumps(out, indent=2))
