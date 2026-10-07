"""SYNTHETIC independent checks of the Anthropic-compat translation leaf.

Deterministic, no network: request contract, block validation, the
wire-shape translation, envelope building on provider-verbatim payloads,
SSE grammar, and the message-batch surface.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from fx1.serve.anthropic_compat import (
    AnthropicBatchCreate,
    AnthropicBatchItem,
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
    anthropic_batch_object,
    anthropic_batch_result,
    anthropic_count_messages,
    anthropic_envelope,
    anthropic_error_body,
    anthropic_events,
    anthropic_model_object,
    anthropic_sse,
    anthropic_to_openai,
)
from fx1.serve.openai_compat import OpenAICompatError


def _msgs(**over: Any) -> AnthropicMessagesRequest:
    body: dict[str, Any] = {
        "model": "fx1",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(over)
    return AnthropicMessagesRequest.model_validate(body)


def test_error_type_map_covers_the_anthropic_kinds() -> None:
    for status, typ in (
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (408, "timeout_error"),
        (413, "request_too_large"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (502, "api_error"),
        (529, "overloaded_error"),
    ):
        body = anthropic_error_body("m", status)
        assert body["type"] == "error" and body["error"]["type"] == typ


def test_block_validation_refuses_unhonorable_shapes() -> None:
    for block in (
        {"type": "image", "source": {}},
        {"type": "document"},
        {"type": "thinking"},
        {"type": "mcp_tool_use"},
        {"type": "citations"},
        {"type": "mystery"},
        {"no_type": 1},
    ):
        with pytest.raises(ValidationError):
            _msgs(messages=[{"role": "user", "content": [block]}])
    # cache_control is a field Anthropic documents that this pipeline
    # can't honor — its presence refuses at validation, never drops
    with pytest.raises(ValidationError):
        _msgs(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
                    ],
                }
            ]
        )
    # tool_result is_error — the wire's tool messages have no channel for
    # the flag; dropping it would present a failure as a normal result
    with pytest.raises(ValidationError):
        _msgs(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "is_error": True,
                            "content": "boom",
                        }
                    ],
                },
            ]
        )
    ok = _msgs(
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "is_error": False,
                        "content": "fine",
                    }
                ],
            },
        ]
    )
    assert anthropic_to_openai(ok)["messages"][-1]["content"] == "fine"


def test_role_alternation_and_turn_contract() -> None:
    with pytest.raises(ValidationError):
        _msgs(messages=[{"role": "assistant", "content": "x"}])  # must open user
    with pytest.raises(ValidationError):
        _msgs(
            messages=[
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ]
        )
    with pytest.raises(ValidationError):
        _msgs(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "tool_use", "id": "t", "name": "f", "input": {}}],
                }
            ]
        )
    with pytest.raises(ValidationError):
        _msgs(
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}],
                },
            ]
        )


def test_system_and_tool_contract_refusals() -> None:
    with pytest.raises(ValidationError):
        _msgs(system=[{"type": "image", "source": {}}])
    ok = _msgs(system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert anthropic_to_openai(ok)["messages"][0] == {"role": "system", "content": "a\nb"}
    with pytest.raises(ValidationError):
        _msgs(
            tools=[
                {"name": "f", "input_schema": {"type": "object"}},
                {"name": "f", "input_schema": {"type": "object"}},
            ]
        )
    with pytest.raises(ValidationError):
        _msgs(tool_choice={"type": "tool", "name": "ghost"}, tools=[])
    with pytest.raises(ValidationError):
        _msgs(tool_choice={"type": "tool"})  # name required for type=tool


def test_unsupported_fields_refuse_named_knobs() -> None:
    for field in ("top_k", "thinking", "service_tier", "mcp_servers", "effort", "cache_control"):
        with pytest.raises(ValidationError):
            _msgs(**{field: 1})
    with pytest.raises(ValidationError):
        _msgs(stop_sequences=["x", ""])
    with pytest.raises(ValidationError):
        _msgs(stop_sequences=["a", "b", "c", "d", "e"])  # >4
    with pytest.raises(ValidationError):
        _msgs(temperature=1.5)


def test_anthropic_to_openai_wire_translation() -> None:
    body = _msgs(
        system="sys",
        stop_sequences=["END"],
        metadata={"user_id": "u-1"},
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "what is it"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me check"},
                    {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"q": "x"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                    },
                ],
            },
        ],
    )
    kw = anthropic_to_openai(body)
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    assert kw["messages"][2]["tool_calls"][0]["function"]["name"] == "lookup"
    assert kw["messages"][3] == {"role": "tool", "tool_call_id": "t1", "content": "a\nb"}
    assert kw["stop"] == ["END"] and kw["store"] is False
    assert kw["user"] == "u-1"
    for tc, want in (
        ({"type": "auto"}, "auto"),
        ({"type": "any"}, "required"),
        ({"type": "none"}, "none"),
        ({"type": "tool", "name": "lookup"}, {"type": "function", "function": {"name": "lookup"}}),
    ):
        tools = [{"name": "lookup", "input_schema": {"type": "object"}}]
        kw = anthropic_to_openai(
            _msgs(messages=[{"role": "user", "content": "x"}], tools=tools, tool_choice=tc)
        )
        assert kw["tool_choice"] == want
    kw = anthropic_to_openai(
        _msgs(
            tools=[{"name": "f", "input_schema": {"type": "object"}}],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
    )
    assert kw["parallel_tool_calls"] is False
    # non-serializable tool_use input refuses at translation — never ships
    # a fake args blob (the request model takes input loosely; the gate is
    # the translator)
    bad = AnthropicMessagesRequest.model_validate(
        {
            "model": "fx1",
            "max_tokens": 1,
            "messages": [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t", "name": "f", "input": {1, 2}}],
                },
            ],
        }
    )
    with pytest.raises(OpenAICompatError):
        anthropic_to_openai(bad)


def test_tool_result_before_text_preserves_block_order() -> None:
    # a result that precedes prose on the wire must not be reported to the
    # model as if the prose came first — block order is causal
    body = _msgs(
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
                    {"type": "text", "text": "after the tool"},
                ],
            },
        ]
    )
    msgs = anthropic_to_openai(body)["messages"]
    assert msgs[2]["role"] == "tool" and msgs[2]["content"] == "r1"
    assert msgs[3] == {"role": "user", "content": "after the tool"}


def test_tool_result_content_shape_refusals() -> None:
    for bad_content in (42, [{"type": "image", "source": {}}, {"type": "text", "text": "x"}]):
        body = AnthropicMessagesRequest.model_validate(
            {
                "model": "fx1",
                "max_tokens": 1,
                "messages": [
                    {"role": "user", "content": "go"},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "t1", "content": bad_content}
                        ],
                    },
                ],
            }
        )
        with pytest.raises(OpenAICompatError):
            anthropic_to_openai(body)


def test_count_tokens_shares_the_message_flattening() -> None:
    body = AnthropicCountTokensRequest.model_validate(
        {
            "model": "fx1",
            "system": "sys",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }
    )
    msgs = anthropic_count_messages(body)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1] == {"role": "user", "content": "hi"}
    with pytest.raises(ValidationError):
        AnthropicCountTokensRequest.model_validate(
            {"model": "fx1", "messages": [{"role": "assistant", "content": "x"}]}
        )


def _env(**over: Any) -> dict[str, Any]:
    env: dict[str, Any] = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }
    env.update(over)
    return env


def test_anthropic_envelope_shape_and_usage_sieve() -> None:
    out = anthropic_envelope(_env())
    assert out["id"] == "msg_abc" and out["type"] == "message" and out["role"] == "assistant"
    assert out["content"] == [{"type": "text", "text": "hello"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"]["input_tokens"] == 3 and out["usage"]["output_tokens"] == 5
    assert out["usage"]["cache_creation_input_tokens"] == 0
    for finish, want in (
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("content_filter", "refusal"),
        ("mystery", "end_turn"),
    ):
        out = anthropic_envelope(
            _env(choices=[{"index": 0, "message": {"content": "x"}, "finish_reason": finish}])
        )
        assert out["stop_reason"] == want
    with pytest.raises(OpenAICompatError) as exc:
        anthropic_envelope(_env(choices=[]))
    assert exc.value.status == 502


def test_anthropic_envelope_survives_hostile_provider_claims() -> None:
    # tool_calls ride verbatim from the provider — a non-dict member or a
    # non-dict `function` must never crash the envelope into a bare 500
    env = _env()
    env["choices"][0]["message"]["tool_calls"] = [
        "garbage",
        {"id": "c1", "type": "function", "function": 42},
        {"id": "c2", "type": "function", "function": {"name": "f", "arguments": "{bad"}},
        {"id": "c3", "type": "function", "function": {"name": "g", "arguments": [1, 2]}},
    ]
    out = anthropic_envelope(env)
    blocks = out["content"]
    # non-dict call skipped; dict calls keep their verbatim claims — even a
    # malformed `function` echoes (id preserved, unshapeable name "" visible)
    assert len(blocks) == 4
    assert blocks[1] == {"type": "tool_use", "id": "c1", "name": "", "input": {}}
    assert blocks[2]["type"] == "tool_use" and blocks[2]["input"] == {"_raw": "{bad"}
    assert blocks[3]["input"] == {"_value": [1, 2]}
    # malformed usage claims read as 0 — never a crash, never a bool
    env2 = _env(usage={"prompt_tokens": True, "completion_tokens": "abc"})
    out2 = anthropic_envelope(env2)
    assert out2["usage"]["input_tokens"] == 0 and out2["usage"]["output_tokens"] == 0
    env3 = _env(usage=None)
    assert anthropic_envelope(env3)["usage"]["input_tokens"] == 0


def test_anthropic_events_and_sse_grammar() -> None:
    env = _env()
    env["choices"][0]["message"]["tool_calls"] = [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}
    ]
    events = list(anthropic_events(env))
    names = [e["event"] for e in events]
    assert names[0] == "message_start" and names[1] == "ping"
    assert names[-2] == "message_delta" and names[-1] == "message_stop"
    # the tool_use input ships as ONE complete partial_json — never faked
    # per-delta token boundaries
    json_deltas = [
        e["data"]["delta"]
        for e in events
        if e["event"] == "content_block_delta" and e["data"]["delta"]["type"] == "input_json_delta"
    ]
    assert len(json_deltas) == 1 and json_deltas[0]["partial_json"] == '{"a":1}'
    frames = list(anthropic_sse(env, skip=2))
    assert all(f.startswith("id: ") for f in frames)
    assert len(list(anthropic_sse(env))) - len(frames) == 2
    md = [e for e in events if e["event"] == "message_delta"][0]
    assert md["data"]["usage"] == {"output_tokens": 5}


def test_model_object_and_batch_surfaces() -> None:
    obj = anthropic_model_object("fx1", created=0)
    assert obj["type"] == "model" and obj["created_at"] == "1970-01-01T00:00:00Z"
    rec = {
        "batch_id": "msgbatch_1",
        "status": "ended",
        "request_counts": {
            "processing": 0,
            "succeeded": 1,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
        "created_at": 0,
        "expires_at": 86400,
        "ended_at": 60,
    }
    b = anthropic_batch_object(rec)
    assert b["processing_status"] == "ended"
    assert b["results_url"] == "/v1/messages/batches/msgbatch_1/results"
    assert b["ended_at"] == "1970-01-01T00:01:00Z"
    live = anthropic_batch_object({**rec, "status": "in_progress"})
    assert live["results_url"] is None
    assert anthropic_batch_result("c", {"type": "succeeded", "message": {}})["custom_id"] == "c"


def test_batch_item_and_create_refusals() -> None:
    with pytest.raises(ValidationError):
        AnthropicBatchItem.model_validate(
            {
                "custom_id": "c",
                "params": {
                    "model": "fx1",
                    "max_tokens": 1,
                    "stream": True,
                    "messages": [{"role": "user", "content": "x"}],
                },
            }
        )
    with pytest.raises(ValidationError):
        AnthropicBatchItem.model_validate(
            {
                "custom_id": "",
                "params": {
                    "model": "fx1",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "x"}],
                },
            }
        )
    with pytest.raises(ValidationError):
        AnthropicBatchCreate.model_validate(
            {
                "requests": [
                    {
                        "custom_id": "dup",
                        "params": {
                            "model": "fx1",
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "x"}],
                        },
                    },
                    {
                        "custom_id": "dup",
                        "params": {
                            "model": "fx1",
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "y"}],
                        },
                    },
                ]
            }
        )
    with pytest.raises(ValidationError):
        AnthropicBatchCreate.model_validate({"requests": []})
    with pytest.raises(ValidationError):
        AnthropicBatchCreate.model_validate(
            {
                "requests": [
                    {
                        "custom_id": "a",
                        "params": {
                            "model": "fx1",
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "x"}],
                        },
                    }
                ],
                "callback_url": "ftp://h",
            }
        )
    with pytest.raises(ValidationError):
        AnthropicBatchCreate.model_validate(
            {
                "requests": [
                    {
                        "custom_id": "a",
                        "params": {
                            "model": "fx1",
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "x"}],
                        },
                    }
                ],
                "callback_secret": "s",
            }
        )
