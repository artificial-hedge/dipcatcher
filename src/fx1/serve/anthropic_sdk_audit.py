"""anthropic_sdk_audit — the stock ``anthropic`` Python SDK as the drop-in leg.

The harness advertises an Anthropic-compatible wire surface under
``/v1/messages*``; ``HarnessClient`` is our own client and could
silently drift in our favor. This audit points the *unmodified* official
``anthropic.AsyncAnthropic`` at the in-process ASGI app — no socket, one
deterministic stub backend — and asserts SDK-typed results throughout: a
response that parses into the SDK's own models is the compatibility
claim; a typed ``APIStatusError`` subclass carrying the ``error``
envelope is the error-shape claim.

Pinned contract (one probe per row):

- *messages* — ``create`` parses a typed ``Message`` (``msg_*`` id,
  ``role=assistant``, typed ``text`` block, ``stop_reason``, integer
  ``usage``); ``system``, ``metadata.user_id``, ``stop_sequences``,
  ``temperature``/``top_p`` bounds, tool transcripts and declared tools
  pass through; alternation, empty turns, image blocks, and the
  documented-but-unhonorable params (``thinking``, ``service_tier``,
  ``cache_control``, ``output_config``, ``diagnostics``…) refuse as
  typed 422s; a body over the byte cap is a typed 413.
- *streaming* — ``create(stream=True)`` yields the Anthropic SSE grammar
  (``message_start`` → ``content_block_start``/``delta``/``stop`` →
  ``message_delta`` → ``message_stop``) and the ``stream()`` context
  manager reassembles ``get_final_message()``; ``Last-Event-ID`` resumes
  a pinned stream; misuse is a typed 400/409.
- *count_tokens* — typed ``input_tokens``, ``system`` folds in,
  ``tools``/``tool_choice`` refuse typed 400, a backend without the
  tokenize channel fails closed 501.
- *models* — ``list`` auto-paginates via the ``after_id``/``before_id``
  cursors; ``retrieve`` parses a model card and 404s typed on an unknown
  id.
- *batches* — ``create``/``retrieve``/``list``/``cancel``/``delete``
  round-trip; ``results`` yields JSONL rows typed
  ``succeeded``/``errored``/``canceled``; ``request_counts`` sums;
  mid-flight delete and post-end cancel refuse typed.
- *errors* — 400/401/403/404/409/413/422/429/5xx map to the SDK's typed
  exception classes carrying the ``{type: "error", error: {...}}``
  envelope; ``Idempotency-Key`` replays byte-identically and conflicts
  typed; ``X-Request-ID`` echoes.

Sealed ``anthropic_sdk_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fx1.serve.backends import SamplingParams

__all__ = ["anthropic_sdk_audit", "anthropic_sdk_audit_bench"]

_STUB_HI = "stub:hi"
_FAIL = "RAISE-BACKEND"
_BLOCK = "BLOCK-ME"
_BLOCK_S = 2.0
_OVER_CAP = "x" * (1 << 21)  # >_MAX_BODY_BYTES — trips the 413 middleware


class _SdkBackend:
    """Deterministic stub — the compatibility claim is about the wire
    layer, so the backend is a fixed echo with tool + tokenize channels
    (never a real model). ``_BLOCK`` tail sleeps inside the batch worker
    so the cancel surface stays observable; ``_FAIL`` tail raises (the
    typed 502 leg). Offered ``tools`` answer a fixed ``tool_calls`` so
    the SDK's ToolUseBlock parse is claimed end-to-end; tool context in
    the transcript alone still answers text."""

    def __init__(self) -> None:
        self.last_usage: dict[str, int] | None = None
        self.last_messages: list[dict[str, Any]] | None = None
        self.counted_messages: list[dict[str, Any]] | None = None

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str:
        self.last_messages = messages
        self.last_usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        tail = str(messages[-1].get("content") or "")
        if tail == _FAIL:
            raise RuntimeError("stub backend exploded")
        if tail == _BLOCK:
            time.sleep(_BLOCK_S)
        return f"stub:{tail}"

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> Any:
        from fx1.serve.backends import ToolCompletion

        self.last_messages = messages
        self.last_usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        tail = str(messages[-1].get("content") or "")
        if tail == _FAIL:
            raise RuntimeError("stub backend exploded")
        if tail == _BLOCK:
            time.sleep(_BLOCK_S)
        if tools:
            return ToolCompletion(
                content=None,
                tool_calls=(
                    {
                        "id": "call_stub",
                        "type": "function",
                        "function": {
                            "name": tools[0].get("function", {}).get("name", "f"),
                            "arguments": '{"q": "hi"}',
                        },
                    },
                ),
                finish_reason="tool_calls",
            )
        return ToolCompletion(content=f"stub:{tail}", tool_calls=None, finish_reason="stop")

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> Iterator[str]:
        self.last_usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        yield "stub:"
        yield "alpha"

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        self.counted_messages = messages
        return sum(len(str(m.get("content") or "").split()) for m in messages) + 2


class _NoTokensBackend:
    """Complete-only stub — no ``count_tokens`` member, so the
    ``TokenCountingBackend`` check fails closed 501."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
    ) -> str:
        return f"stub:{messages[-1].get('content')}"


@contextmanager
def _sdk_client(
    *,
    env_key: str | None = None,
    rate_limit_rps: float | None = None,
    no_tokens: bool = False,
) -> Iterator[tuple[Any, _SdkBackend, Any]]:
    """``AsyncAnthropic`` bound to the in-process app over httpx's ASGI
    transport — no socket. The anthropic SDK vendored its own ``httpx2``
    fork (same trick as openai's); ``base_url`` carries no ``/v1`` — the
    SDK pins ``/v1/messages`` paths itself, so appending it would
    double-mount the prefix.

    Env keys are cleared so the loopback-authorized surface answers
    unauthenticated — except ``env_key``, which pins ``FX1_API_KEY`` for
    the credential probes; ``rate_limit_rps`` shrinks the app's token
    bucket for the 429 probe; ``no_tokens`` resolves a backend without a
    tokenize channel for the 501 leg. Yields ``(client, stub, http)`` —
    ``http`` mints managed keys through ``/harness/keys``."""
    import anthropic
    import httpx2

    from fx1.serve import api as api_mod

    stub = _SdkBackend()
    saved = {
        k: os.environ.get(k) for k in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_API_RATE_LIMIT_RPS")
    }
    try:
        # an ambient RATE_LIMIT would throttle the whole probe matrix;
        # the 429 leg passes its own explicit value instead
        os.environ.pop("FX1_API_KEY", None)
        os.environ.pop("MOONSHOT_API_KEY", None)
        os.environ.pop("FX1_API_RATE_LIMIT_RPS", None)
        if env_key is not None:
            os.environ["FX1_API_KEY"] = env_key
        resolver = (lambda *a, **k: _NoTokensBackend()) if no_tokens else (lambda *a, **k: stub)
        app = api_mod.create_app(
            backend_resolver=resolver,
            # None falls back to the env/default, so passing it through
            # unconditionally keeps the None-case identical to omitting it
            rate_limit_rps=rate_limit_rps,
        )
        transport = httpx2.ASGITransport(app=app)
        http = httpx2.AsyncClient(transport=transport, base_url="http://sdk-audit")
        yield (
            anthropic.AsyncAnthropic(
                base_url="http://sdk-audit",
                api_key="sdk-audit",
                http_client=http,
                max_retries=0,
            ),
            stub,
            http,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def _probe_messages(cl: Any, stub: _SdkBackend, out: dict[str, Any]) -> None:
    import anthropic

    m = await cl.messages.create(
        model="fx1", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
    )
    out["sdk_msg_create"] = (
        m.id.startswith("msg_")
        and m.type == "message"
        and m.role == "assistant"
        and len(m.content) == 1
        and m.content[0].type == "text"
        and m.content[0].text == _STUB_HI
        and m.stop_reason == "end_turn"
        and isinstance(m.usage.input_tokens, int)
        and isinstance(m.usage.output_tokens, int)
        and m.usage.input_tokens > 0
    )
    out["sdk_msg_model_echo"] = isinstance(m.model, str) and bool(m.model)
    sys_msg = await cl.messages.create(
        model="fx1",
        max_tokens=8,
        system="sys-note",
        messages=[{"role": "user", "content": "hi"}],
    )
    out["sdk_msg_system"] = (
        sys_msg.content[0].type == "text"
        and sys_msg.content[0].text == _STUB_HI
        and stub.last_messages is not None
        and stub.last_messages[0] == {"role": "system", "content": "sys-note"}
    )
    tagged = await cl.messages.create(
        model="fx1",
        max_tokens=8,
        metadata={"user_id": "u-1"},
        messages=[{"role": "user", "content": "hi"}],
    )
    out["sdk_msg_metadata"] = tagged.type == "message" and tagged.content[0].type == "text"
    stopped = await cl.messages.create(
        model="fx1",
        max_tokens=8,
        stop_sequences=["\n", "END"],
        messages=[{"role": "user", "content": "hi"}],
    )
    out["sdk_msg_stop_sequences"] = stopped.id.startswith("msg_")
    warm = await cl.messages.create(
        model="fx1",
        max_tokens=8,
        extra_body={"temperature": 0.3},
        messages=[{"role": "user", "content": "hi"}],
    )
    out["sdk_msg_temperature"] = warm.id.startswith("msg_")
    focused = await cl.messages.create(
        model="fx1",
        max_tokens=8,
        extra_body={"top_p": 0.5},
        messages=[{"role": "user", "content": "hi"}],
    )
    out["sdk_msg_top_p"] = focused.id.startswith("msg_")
    multi = await cl.messages.create(
        model="fx1",
        max_tokens=8,
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "stub:hi"},
            {"role": "user", "content": "again"},
        ],
    )
    out["sdk_msg_multi_turn"] = multi.content[0].text == "stub:again"
    tool_turn = await cl.messages.create(
        model="fx1",
        max_tokens=8,
        messages=[
            {"role": "user", "content": "look up"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "lookup",
                        "input": {"q": "hi"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
            },
        ],
    )
    out["sdk_msg_tool_history"] = tool_turn.id.startswith("msg_")
    tooled = await cl.messages.create(
        model="fx1",
        max_tokens=8,
        tools=[
            {
                "name": "lookup",
                "description": "look a thing up",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ],
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": "hi"}],
    )
    # offered tools surface a typed ToolUseBlock — the full round-trip
    # (wire tools → provider tool_calls → anthropic tool_use blocks →
    # stop_reason 'tool_use') is the claim, not just a parsable message
    out["sdk_msg_tools_declared"] = (
        tooled.type == "message"
        and tooled.stop_reason == "tool_use"
        and tooled.content[-1].type == "tool_use"
        and tooled.content[-1].name == "lookup"
        and tooled.content[-1].input == {"q": "hi"}
    )
    raw = await cl.messages.with_raw_response.create(
        model="fx1",
        max_tokens=8,
        extra_headers={"x-request-id": "sdk-rid-42"},
        messages=[{"role": "user", "content": "hi"}],
    )
    parsed = await raw.parse()
    out["sdk_msg_raw_headers"] = (
        parsed.id.startswith("msg_") and raw.headers["x-request-id"] == "sdk-rid-42"
    )
    try:
        # no max_tokens — the server's own validation must enforce the
        # required field even when a caller bypasses the SDK's signature
        await cl.post(
            "/v1/messages",
            body={"model": "fx1", "messages": [{"role": "user", "content": "hi"}]},
            cast_to=object,
        )
        out["sdk_msg_max_tokens_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_max_tokens_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            extra_body={"temperature": 1.5},
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_msg_temperature_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_temperature_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "again"},
            ],
        )
        out["sdk_msg_alternation_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_alternation_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            messages=[{"role": "assistant", "content": "first"}],
        )
        out["sdk_msg_assistant_first_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_assistant_first_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "AAAA",
                            },
                        }
                    ],
                }
            ],
        )
        out["sdk_msg_image_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_image_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_msg_thinking_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_thinking_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            service_tier="auto",
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_msg_service_tier_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_service_tier_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            extra_body={"cache_control": {"type": "ephemeral"}},
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_msg_cache_control_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_cache_control_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            output_config={"effort": "high"},
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_msg_output_config_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_output_config_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            diagnostics={"enabled": True},
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_msg_diagnostics_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_diagnostics_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_msg_tool_choice_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_tool_choice_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {},
                        }
                    ],
                }
            ],
        )
        out["sdk_msg_tool_use_user_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_tool_use_user_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            stop_sequences=["a", "b", "c", "d", "e"],
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_msg_stop_seq_max_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_stop_seq_max_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            metadata={"user_id": "u" * 600},
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_msg_metadata_len_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_metadata_len_422"] = True
    try:
        await cl.messages.create(model="fx1", max_tokens=8, messages=[])
        out["sdk_msg_empty_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_msg_empty_422"] = True
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            system=_OVER_CAP,
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_msg_body_cap_413"] = False
    except anthropic.RequestTooLargeError:
        out["sdk_msg_body_cap_413"] = True
    key = f"sdk-idem-{time.time_ns()}"
    body = {"role": "user", "content": "hi"}
    first = await cl.messages.create(
        model="fx1",
        max_tokens=8,
        extra_headers={"Idempotency-Key": key},
        messages=[body],
    )
    replay = await cl.messages.with_raw_response.create(
        model="fx1",
        max_tokens=8,
        extra_headers={"Idempotency-Key": key},
        messages=[body],
    )
    replayed = await replay.parse()
    out["sdk_idem_replay"] = (
        replayed.id == first.id and replay.headers.get("x-fx1-idempotent-replay") == "true"
    )
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            extra_headers={"Idempotency-Key": key},
            messages=[{"role": "user", "content": "different"}],
        )
        out["sdk_idem_conflict_409"] = False
    except anthropic.ConflictError:
        out["sdk_idem_conflict_409"] = True


async def _probe_streaming(cl: Any, out: dict[str, Any]) -> None:
    import anthropic

    events = [
        ev
        async for ev in await cl.messages.create(
            model="fx1",
            max_tokens=8,
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )
    ]
    types = [ev.type for ev in events]
    out["sdk_stream_grammar"] = (
        types[0] == "message_start"
        and types[-1] == "message_stop"
        and "content_block_start" in types
        and "content_block_delta" in types
        and "content_block_stop" in types
        and "message_delta" in types
        and types.index("content_block_start") < types.index("content_block_delta")
        and types.index("content_block_delta") < types.index("message_delta")
        and types.index("message_delta") < types.index("message_stop")
    )
    start = events[0]
    out["sdk_stream_start_fields"] = (
        start.type == "message_start"
        and start.message.id.startswith("msg_")
        and start.message.role == "assistant"
        and start.message.content == []
    )
    out["sdk_stream_text"] = (
        "".join(ev.delta.text for ev in events if ev.type == "content_block_delta") == _STUB_HI
    )
    mdelta = next(ev for ev in events if ev.type == "message_delta")
    out["sdk_stream_message_delta"] = mdelta.delta.stop_reason == "end_turn" and isinstance(
        mdelta.usage.output_tokens, int
    )
    async with cl.messages.stream(
        model="fx1", max_tokens=8, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        text = "".join([t async for t in stream.text_stream])
        final = await stream.get_final_message()
    out["sdk_stream_ctx"] = (
        text == _STUB_HI
        and final.type == "message"
        and final.stop_reason == "end_turn"
        and final.content[0].type == "text"
    )
    key = f"sdk-resume-{time.time_ns()}"
    body = [{"role": "user", "content": "hi"}]
    first = [
        ev
        async for ev in await cl.messages.create(
            model="fx1",
            max_tokens=8,
            stream=True,
            extra_headers={"Idempotency-Key": key},
            messages=body,
        )
    ]
    resumed = [
        ev
        async for ev in await cl.messages.create(
            model="fx1",
            max_tokens=8,
            stream=True,
            extra_headers={"Idempotency-Key": key, "Last-Event-ID": "2"},
            messages=body,
        )
    ]
    # the server numbers frames including ``ping`` (which the SDK
    # filters from the typed stream) — ``Last-Event-ID: 2`` skips
    # message_start/ping/content_block_start, so the resumed stream is
    # exactly the typed stream's first-delta suffix
    out["sdk_stream_resume"] = (
        len(first) >= 4
        and bool(resumed)
        and resumed[0].type == "content_block_delta"
        and [ev.type for ev in resumed] == [ev.type for ev in first[2:]]
    )
    try:
        await cl.messages.create(
            model="fx1",
            max_tokens=8,
            messages=[{"role": "user", "content": "hi"}],
            extra_headers={"Last-Event-ID": "0"},
        )
        out["sdk_stream_resume_nonstream_400"] = False
    except anthropic.BadRequestError:
        out["sdk_stream_resume_nonstream_400"] = True
    try:
        [
            ev
            async for ev in await cl.messages.create(
                model="fx1",
                max_tokens=8,
                stream=True,
                extra_headers={
                    "Idempotency-Key": f"sdk-miss-{time.time_ns()}",
                    "Last-Event-ID": "2",
                },
                messages=body,
            )
        ]
        out["sdk_stream_resume_miss_409"] = False
    except anthropic.ConflictError:
        out["sdk_stream_resume_miss_409"] = True


async def _probe_count_tokens(cl: Any, stub: _SdkBackend, out: dict[str, Any]) -> None:
    import anthropic

    n = await cl.messages.count_tokens(
        model="fx1", messages=[{"role": "user", "content": "hi there"}]
    )
    out["sdk_tokens_count"] = isinstance(n.input_tokens, int) and n.input_tokens > 0
    await cl.messages.count_tokens(
        model="fx1",
        system="sys-note",
        messages=[{"role": "user", "content": "hi"}],
    )
    out["sdk_tokens_system"] = stub.counted_messages is not None and stub.counted_messages[0] == {
        "role": "system",
        "content": "sys-note",
    }
    try:
        await cl.messages.count_tokens(
            model="fx1",
            tools=[
                {
                    "name": "lookup",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_tokens_tools_400"] = False
    except anthropic.BadRequestError:
        out["sdk_tokens_tools_400"] = True
    try:
        await cl.messages.count_tokens(
            model="fx1",
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user", "content": "hi"}],
        )
        out["sdk_tokens_thinking_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_tokens_thinking_422"] = True


async def _probe_models(cl: Any, out: dict[str, Any]) -> None:
    import anthropic

    ids = [m.id async for m in cl.models.list(limit=2)]
    out["sdk_models_list"] = {"fx1", "hosted_k3", "local_fx1", "byok"} <= set(ids) and len(ids) >= 4
    page = await cl.models.list(limit=2)
    out["sdk_models_page_shape"] = (
        len(page.data) == 2 and page.has_more is True and all(m.type == "model" for m in page.data)
    )
    page2 = await page.get_next_page()
    out["sdk_models_cursor"] = len(page2.data) == 2 and {m.id for m in page.data} != {
        m.id for m in page2.data
    }
    got = await cl.models.retrieve("fx1")
    out["sdk_models_retrieve"] = got.id == "fx1" and got.type == "model" and bool(got.display_name)
    try:
        await cl.models.retrieve("definitely-not-a-model")
        out["sdk_models_retrieve_404"] = False
    except anthropic.NotFoundError as exc:
        out["sdk_models_retrieve_404"] = exc.status_code == 404


async def _wait_ended(cl: Any, batch_id: str, *, tries: int = 400) -> Any:
    """Poll retrieve until the batch ends — the worker runs on the app's
    executor, so a short sleep loop is the honest wait."""
    for _ in range(tries):
        b = await cl.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            return b
        await asyncio.sleep(0.05)
    return b


async def _probe_batches(cl: Any, out: dict[str, Any]) -> None:
    import anthropic

    b = await cl.messages.batches.create(
        requests=[
            {
                "custom_id": "req-a",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {"user_id": "batch-u"},
                },
            },
            {
                "custom_id": "req-b",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": _FAIL}],
                },
            },
            {
                "custom_id": "req-c",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            },
        ]
    )
    out["sdk_batch_create"] = (
        b.id.startswith("msgbatch_")
        and b.type == "message_batch"
        and b.processing_status in ("in_progress", "ended")
        and isinstance(b.request_counts.processing, int)
    )
    got = await cl.messages.batches.retrieve(b.id)
    out["sdk_batch_retrieve"] = got.id == b.id and got.type == "message_batch"
    ended = await _wait_ended(cl, b.id)
    out["sdk_batch_ends"] = (
        ended.processing_status == "ended"
        and ended.ended_at is not None
        and isinstance(ended.results_url, str)
    )
    out["sdk_batch_counts"] = (
        ended.request_counts.succeeded == 2
        and ended.request_counts.errored == 1
        and ended.request_counts.processing == 0
    )
    rows = [row async for row in await cl.messages.batches.results(b.id)]
    by_id = {row.custom_id: row for row in rows}
    ok_row = by_id.get("req-a")
    bad_row = by_id.get("req-b")
    out["sdk_batch_results"] = (
        set(by_id) == {"req-a", "req-b", "req-c"}
        and ok_row is not None
        and ok_row.result.type == "succeeded"
        and ok_row.result.message.content[0].text == _STUB_HI
        and bad_row is not None
        and bad_row.result.type == "errored"
        and isinstance(bad_row.result.error.type, str)
    )
    try:
        [row async for row in await cl.messages.batches.results(b.id)]
        out["sdk_batch_results_replay"] = True
    except anthropic.APIStatusError:
        out["sdk_batch_results_replay"] = False

    cb = await cl.messages.batches.create(
        requests=[
            {
                "custom_id": "block-1",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": _BLOCK}],
                },
            },
            {
                "custom_id": "block-2",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": _BLOCK}],
                },
            },
        ]
    )
    try:
        await cl.messages.batches.delete(cb.id)
        out["sdk_batch_delete_midflight_400"] = False
    except anthropic.BadRequestError:
        out["sdk_batch_delete_midflight_400"] = True
    try:
        # ``batches.results`` short-circuits client-side while the batch
        # is in flight — the raw route still answers the wire 400
        await cl.get(f"/v1/messages/batches/{cb.id}/results", cast_to=object)
        out["sdk_batch_results_early_400"] = False
    except anthropic.BadRequestError:
        out["sdk_batch_results_early_400"] = True
    cancelled = await cl.messages.batches.cancel(cb.id)
    out["sdk_batch_cancel"] = (
        cancelled.id == cb.id
        and cancelled.processing_status == "canceling"
        and cancelled.cancel_initiated_at is not None
    )
    cb_end = await _wait_ended(cl, cb.id)
    crows = [row async for row in await cl.messages.batches.results(cb.id)]
    out["sdk_batch_cancel_ends"] = cb_end.processing_status == "ended" and any(
        row.result.type == "canceled" for row in crows
    )
    try:
        await cl.messages.batches.cancel(cb.id)
        out["sdk_batch_cancel_ended_400"] = False
    except anthropic.BadRequestError:
        out["sdk_batch_cancel_ended_400"] = True

    lb = await cl.messages.batches.create(
        requests=[
            {
                "custom_id": "l1",
                "params": {
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            }
        ]
    )
    all_ids = [x.id async for x in cl.messages.batches.list(limit=1)]
    out["sdk_batch_list_paginated"] = {b.id, cb.id, lb.id} <= set(all_ids) and len(all_ids) >= 3
    newest = all_ids[0]
    before = await cl.messages.batches.list(before_id=all_ids[-1], limit=1)
    out["sdk_batch_before_id"] = len(before.data) == 1 and before.data[0].id == all_ids[-2]
    after = await cl.messages.batches.list(after_id=newest, limit=1)
    out["sdk_batch_after_id"] = len(after.data) == 1 and after.data[0].id == all_ids[1]
    deleted = await cl.messages.batches.delete(b.id)
    out["sdk_batch_delete"] = deleted.id == b.id and deleted.type == "message_batch_deleted"
    try:
        await cl.messages.batches.retrieve(b.id)
        out["sdk_batch_deleted_404"] = False
    except anthropic.NotFoundError:
        out["sdk_batch_deleted_404"] = True
    try:
        await cl.messages.batches.retrieve("msgbatch_nonexistent")
        out["sdk_batch_get_404"] = False
    except anthropic.NotFoundError:
        out["sdk_batch_get_404"] = True
    try:
        await cl.messages.batches.create(
            requests=[
                {
                    "custom_id": "dup",
                    "params": {
                        "model": "fx1",
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                },
                {
                    "custom_id": "dup",
                    "params": {
                        "model": "fx1",
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                },
            ]
        )
        out["sdk_batch_dup_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_batch_dup_422"] = True
    try:
        await cl.messages.batches.create(
            requests=[
                {
                    "custom_id": "s1",
                    "params": {
                        "model": "fx1",
                        "max_tokens": 8,
                        "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                }
            ]
        )
        out["sdk_batch_stream_422"] = False
    except anthropic.UnprocessableEntityError:
        out["sdk_batch_stream_422"] = True


async def _probe_errors(out: dict[str, Any]) -> None:
    import anthropic

    with _sdk_client() as (cl, _stub, _http):
        try:
            await cl.messages.create(
                model="fx1",
                max_tokens=8,
                messages=[{"role": "user", "content": _FAIL}],
            )
            out["sdk_err_502"] = False
        except anthropic.InternalServerError as exc:
            out["sdk_err_502"] = exc.status_code >= 500
        try:
            await cl.post(
                "/v1/messages",
                body={
                    "model": "fx1",
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": "hot",
                },
                cast_to=object,
            )
            out["sdk_err_422"] = False
        except anthropic.UnprocessableEntityError as exc:
            out["sdk_err_422"] = exc.status_code == 422

    with _sdk_client(env_key="fx1-test-key") as (cl, _stub, http):
        try:
            cl.api_key = "wrong-key"
            await cl.messages.create(
                model="fx1",
                max_tokens=8,
                messages=[{"role": "user", "content": "hi"}],
            )
            out["sdk_err_401"] = False
        except anthropic.AuthenticationError as exc:
            body = exc.body if isinstance(exc.body, dict) else {}
            out["sdk_err_401"] = (
                exc.status_code == 401
                and body.get("type") == "error"
                and isinstance(body.get("error"), dict)
                and isinstance(body["error"].get("type"), str)
            )
        minted = await http.post(
            "/harness/keys",
            json={"name": "sdk-ro", "scopes": ["read"]},
            headers={"X-API-Key": "fx1-test-key"},
        )
        ro_key = minted.json()["key"] if minted.status_code == 201 else "nope"
        cl.api_key = ro_key
        try:
            await cl.messages.create(
                model="fx1",
                max_tokens=8,
                messages=[{"role": "user", "content": "hi"}],
            )
            out["sdk_err_403"] = False
        except anthropic.PermissionDeniedError as exc:
            out["sdk_err_403"] = exc.status_code == 403

    with _sdk_client(rate_limit_rps=0.05) as (cl, _stub, _http):
        await cl.models.retrieve("fx1")  # consume the token bucket's credit
        try:
            await cl.messages.create(
                model="fx1",
                max_tokens=8,
                messages=[{"role": "user", "content": "hi"}],
            )
            out["sdk_err_429"] = False
        except anthropic.RateLimitError as exc:
            out["sdk_err_429"] = (
                exc.status_code == 429 and exc.response.headers.get("retry-after") is not None
            )


async def _probe_tokens_no_channel(out: dict[str, Any]) -> None:
    import anthropic

    with _sdk_client(no_tokens=True) as (cl, _stub, _http):
        try:
            await cl.messages.count_tokens(
                model="fx1", messages=[{"role": "user", "content": "hi"}]
            )
            out["sdk_tokens_no_channel_501"] = False
        except anthropic.InternalServerError as exc:
            # 501 lands in the SDK's 5xx bucket — the claim is the typed
            # class fires, not a bare connection error
            out["sdk_tokens_no_channel_501"] = exc.status_code == 501


def anthropic_sdk_audit() -> dict[str, Any]:
    """Run the whole surface under the stock SDK — one event loop, one
    app per auth variant, probes accumulate as literal bools."""
    out: dict[str, Any] = {}
    loop = asyncio.new_event_loop()
    try:
        with _sdk_client() as (cl, stub, _http):
            loop.run_until_complete(_probe_messages(cl, stub, out))
            loop.run_until_complete(_probe_streaming(cl, out))
            loop.run_until_complete(_probe_count_tokens(cl, stub, out))
            loop.run_until_complete(_probe_models(cl, out))
            loop.run_until_complete(_probe_batches(cl, out))
        loop.run_until_complete(_probe_errors(out))
        loop.run_until_complete(_probe_tokens_no_channel(out))
    finally:
        loop.close()
    return out


def anthropic_sdk_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under anthropic_sdk_audit.v1."""
    r = anthropic_sdk_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "anthropic_sdk_audit",
        "schema": "anthropic_sdk_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "The unmodified anthropic AsyncAnthropic client drives the "
            "harness end-to-end in-process: messages.create (typed Message "
            "+ usage), both streaming surfaces (raw create(stream=True) "
            "SSE grammar and the stream() ctx manager's "
            "get_final_message), count_tokens incl. the no-channel 501, "
            "cursor-paginated models.list with typed 404 retrieve, "
            "message batches (create/retrieve/list/cancel/delete + JSONL "
            "results incl. canceled rows), params pass-through (system, "
            "metadata.user_id, stop_sequences, temperature/top_p), "
            "idempotent replay + typed conflicts, and every wire failure "
            "lands as the SDK's typed exception class carrying the "
            "anthropic error envelope."
            if ok
            else f"ANTHROPIC SDK AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
