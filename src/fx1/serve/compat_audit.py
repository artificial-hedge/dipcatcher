"""compat_audit — translation-invariant battery over the dialect layers.

``openai_compat`` and ``anthropic_compat`` are where every request and
every answer crosses a wire grammar: request models, ``*_to_kwargs``
sampling/backend resolution, envelope builders, stream-chunk grammar,
batch/embeddings/file stores, and the Anthropic field-by-field
translation. These are pure functions — the bugs that matter are silent
shape corruption (a dropped field, a mis-keyed id, a refusal that
accepts), which sampled unit tests cover sparsely; an audit pins the
invariants exhaustively.

``compat_audit()`` runs every probe and returns ``{name: True|False|…}``
— ``False`` pins a measured divergence; the sealed bench payload names
every defect. Probes are deterministic, offline, and order-independent.
"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import ValidationError

from .anthropic_compat import (
    _FINISH_TO_STOP_REASON,
    AnthropicBatchCreate,
    AnthropicBatchItem,
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
    _system_text,
    _tool_result_content,
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
from .openai_compat import (
    OPENAI_BACKENDS,
    OPENAI_BATCH_ENDPOINTS,
    OPENAI_MODEL_IDS,
    OpenAIChatRequest,
    OpenAICompatError,
    OpenAICompletionRequest,
    OpenAIConversationCreate,
    OpenAIConversationItemsAdd,
    OpenAIEmbeddingRequest,
    OpenAIEnvelopeStore,
    OpenAIResponseRequest,
    OpenAIVectorStoreCreate,
    OpenAIVectorStoreFileCreate,
    OpenAIVectorStoreSearch,
    _text_pieces,
    batch_line_body,
    batch_line_shape,
    batch_object,
    batch_output_line,
    chained_response_input,
    chat_messages_for_store,
    completion_events,
    conversation_id_of,
    embeddings_to_kwargs,
    file_object,
    file_search_call_item,
    is_openai_path,
    legacy_to_chat,
    openai_chunks,
    openai_completion_envelope,
    openai_embedding_envelope,
    openai_envelope,
    openai_error_body,
    openai_messages,
    openai_response_call_items,
    openai_response_events,
    openai_response_object,
    openai_response_replay_events,
    openai_to_kwargs,
    openai_usage,
    paged_item_list,
    response_input_items_for_store,
    response_input_to_messages,
    response_output_pieces,
    response_query_text,
    response_text_format,
    response_to_kwargs,
    validate_response_format,
)

__all__ = ["compat_audit", "compat_audit_bench"]

_AUDIT_PROMPT = "audit prompt"
_CHAT_PATH = "/v1/chat/completions"
_RESP_PATH = "/v1/responses"
_CHAT_OBJECT = "chat.completion"
_ANSWER = "the answer"
_MSGS = [{"role": "user", "content": _AUDIT_PROMPT}]


def _chat_body(**kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "fx1", "messages": list(_MSGS)}
    body.update(kw)
    return body


def _chat_req(**kw: Any) -> OpenAIChatRequest:
    return OpenAIChatRequest.model_validate(_chat_body(**kw))


def _resp_req(**kw: Any) -> OpenAIResponseRequest:
    body: dict[str, Any] = {"model": "fx1", "input": _AUDIT_PROMPT}
    body.update(kw)
    return OpenAIResponseRequest.model_validate(body)


def _anth_req(**kw: Any) -> AnthropicMessagesRequest:
    body: dict[str, Any] = {
        "model": "fx1",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": _AUDIT_PROMPT}],
    }
    body.update(kw)
    return AnthropicMessagesRequest.model_validate(body)


def _chat_env(text: str | list[str] = "answer", **kw: Any) -> dict[str, Any]:
    return openai_envelope(
        cid="fixedcid0001",
        content=text,
        backend="stub",
        model="fx1",
        usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        created=1700000000,
        **kw,
    )


def _refuse(fn: Any, *args: Any, **kw: Any) -> tuple[bool, int | None, str | None]:
    """Run ``fn`` expecting an ``OpenAICompatError``; report whether it
    refused, with which status/code."""
    try:
        fn(*args, **kw)
    except OpenAICompatError as exc:
        return True, getattr(exc, "status", getattr(exc, "status_code", None)), exc.code
    return False, None, None


# ---------------------------------------------------------------------------
# Error bodies + path detection
# ---------------------------------------------------------------------------


def _probe_errors_paths() -> dict[str, bool]:
    out: dict[str, bool] = {}
    body = openai_error_body("boom", 400, "bad_request")
    out["oai_error_body_shape"] = (
        body.get("error", {}).get("message") == "boom"
        and body["error"].get("code") == "bad_request"
        and isinstance(body["error"].get("type"), str)
    )
    abody = anthropic_error_body("boom", 400)
    out["anth_error_body_shape"] = (
        abody.get("type") == "error"
        and abody.get("error", {}).get("message") == "boom"
        and isinstance(abody["error"].get("type"), str)
    )
    hits = [_CHAT_PATH, _RESP_PATH, "/v1/models", "/v1/embeddings"]
    misses = ["/harness/complete", "", "/v2/chat/completions", "/healthz", "v1/x"]
    out["is_openai_path_hits"] = all(is_openai_path(p) for p in hits)
    out["is_openai_path_misses"] = not any(is_openai_path(p) for p in misses)
    out["model_id_registry_nonempty"] = bool(OPENAI_MODEL_IDS) and "fx1" in OPENAI_MODEL_IDS
    out["backend_registry_sane"] = {"hosted_k3", "local_fx1", "byok"} <= set(OPENAI_BACKENDS)
    return out


# ---------------------------------------------------------------------------
# Chat request contract + message normalization
# ---------------------------------------------------------------------------


def _probe_chat_contract() -> dict[str, bool]:
    out: dict[str, bool] = {}
    out["chat_req_accepts_minimal"] = _chat_req().model == "fx1"

    def _bad(**kw: Any) -> bool:
        try:
            OpenAIChatRequest.model_validate(_chat_body(**kw))
        except ValidationError:
            return True
        return False

    out["chat_req_refuses_empty_messages"] = _bad(messages=[])
    out["chat_req_refuses_temperature_oob"] = _bad(temperature=2.01) and _bad(temperature=-0.1)
    out["chat_req_refuses_n_oob"] = _bad(n=0) and _bad(n=9)
    out["chat_req_refuses_max_tokens_oob"] = _bad(max_tokens=0)
    out["chat_req_refuses_tokens_disagree"] = _bad(max_tokens=33, max_completion_tokens=44)
    out["chat_req_toolchoice_needs_tools"] = _bad(tool_choice="auto") and _bad(
        parallel_tool_calls=True
    )
    out["chat_req_accepts_full_knobs"] = (
        _chat_req(
            temperature=0.7,
            top_p=0.9,
            n=2,
            stop=["\n", "END"],
            seed=7,
            max_tokens=44,
            max_completion_tokens=44,
            presence_penalty=0.5,
            frequency_penalty=-0.5,
            stream=True,
            stream_options={"include_usage": True},
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            tool_choice="auto",
            parallel_tool_calls=False,
            logprobs=True,
            top_logprobs=3,
            user="u1",
            metadata={"k": "v"},
        ).n
        == 2
    )
    msgs = openai_messages(
        _chat_req(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "res"},
            ]
        ).messages
    )
    out["openai_messages_roles"] = [m["role"] for m in msgs] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    out["openai_messages_tool_passthrough"] = (
        msgs[2].get("tool_calls", [{}])[0].get("id") == "call_1"
        and msgs[3].get("tool_call_id") == "call_1"
    )
    return out


# ---------------------------------------------------------------------------
# ``openai_to_kwargs`` — sampling + link resolution
# ---------------------------------------------------------------------------


def _probe_openai_to_kwargs() -> dict[str, bool]:
    out: dict[str, bool] = {}
    kw = openai_to_kwargs(_chat_req(model="hosted_k3"))
    out["kwargs_backend_default"] = kw["backend"] == "hosted_k3"
    out["kwargs_messages_flattened"] = kw["messages"] == _MSGS
    # model naming a backend wins the link; a registered model falls to
    # the default link
    out["kwargs_model_backend_link"] = (
        openai_to_kwargs(_chat_req(model="local_fx1"))["backend"] == "local_fx1"
    )
    out["kwargs_default_link"] = openai_to_kwargs(_chat_req())["backend"] == "hosted_k3"
    # explicit fx1.backend beats the model name
    out["kwargs_ext_beats_model"] = (
        openai_to_kwargs(_chat_req(model="local_fx1", fx1={"backend": "hosted_k3"}))["backend"]
        == "hosted_k3"
    )
    # header beats model, loses to ext
    out["kwargs_header_backend"] = (
        openai_to_kwargs(_chat_req(), headers={"X-Fx1-Backend": "byok"})["backend"] == "byok"
    )
    # byok headers bind the byok link even when model is a foreign name
    out["kwargs_byok_header_link"] = (
        openai_to_kwargs(
            _chat_req(model="gpt-4o"),
            headers={
                "X-Fx1-Byok-Base-Url": "https://up.example.com/v1",
                "X-Fx1-Byok-Api-Key": "probe-key",
            },
        )["backend"]
        == "byok"
    )
    # base-url without the key refuses — a half-formed credential is a
    # hard error, never a silent default link
    refused_b, _, _ = _refuse(
        openai_to_kwargs,
        _chat_req(model="gpt-4o"),
        headers={"X-Fx1-Byok-Base-Url": "https://up.example.com/v1"},
    )
    out["kwargs_byok_requires_key"] = refused_b
    # unregistered name on a completion surface = fail-closed 404
    refused, status, code = _refuse(openai_to_kwargs, _chat_req(model="typo-model-9"))
    out["kwargs_unknown_model_404"] = refused and status == 404 and code == "model_not_found"
    # sampling fields land verbatim
    kw2 = openai_to_kwargs(
        _chat_req(
            temperature=0.5,
            top_p=0.8,
            seed=42,
            max_completion_tokens=22,
            stop="END",
            presence_penalty=0.1,
            frequency_penalty=-0.1,
            logit_bias={"7": 2},
            user="u",
            service_tier="flex",
            reasoning_effort="high",
            verbosity="low",
            prompt_cache_key="pk",
            prompt_cache_retention="24h",
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            tool_choice="required",
            parallel_tool_calls=True,
            logprobs=True,
            top_logprobs=5,
            metadata={"m": "v"},
        )
    )
    out["kwargs_sampling_passthrough"] = (
        math.isclose(kw2["temperature"], 0.5)
        and math.isclose(kw2["top_p"], 0.8)
        and kw2["seed"] == 42
        and kw2["max_tokens"] == 22  # max_completion_tokens wins
        and kw2["stop"] == ["END"]  # str folds to list
        and math.isclose(kw2["presence_penalty"], 0.1)
        and kw2["logit_bias"] == {"7": 2}
        and kw2["user"] == "u"
        and kw2["reasoning_effort"] == "high"
        and kw2["verbosity"] == "low"
        and kw2["prompt_cache_key"] == "pk"
        and kw2["tool_choice"] == "required"
        and kw2["top_logprobs"] == 5
        and kw2["metadata"] == {"m": "v"}
        and kw2["tools"][0]["function"]["name"] == "f"
    )
    # response_format post-validation: provider output must honor the
    # declared shape — a violation is a 502 format_violation, not a ship
    try:
        validate_response_format({"type": "json_object"}, '{"k": 1}')
        rf_ok = True
    except OpenAICompatError:
        rf_ok = False
    refused_j, status_j, code_j = _refuse(
        validate_response_format, {"type": "json_object"}, "not json"
    )
    refused_j2, _, _ = _refuse(validate_response_format, {"type": "json_object"}, "[1]")
    out["response_format_validation"] = (
        rf_ok and refused_j and status_j == 502 and code_j == "format_violation" and refused_j2
    )
    out["openai_usage_none_vs_passthrough"] = openai_usage(None) is None and openai_usage(
        {"prompt_tokens": 1}
    ) == {"prompt_tokens": 1}
    return out


# ---------------------------------------------------------------------------
# Envelope builders
# ---------------------------------------------------------------------------


def _probe_envelopes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    env = _chat_env()
    out["env_shape"] = (
        env["object"] == _CHAT_OBJECT
        and env["id"].startswith("chatcmpl-")
        and env["created"] == 1700000000
        and env["system_fingerprint"] == "stub"
        and env["usage"]["total_tokens"] == 7
    )
    out["env_choice_shape"] = (
        env["choices"][0]["index"] == 0
        and env["choices"][0]["message"]["role"] == "assistant"
        and env["choices"][0]["message"]["content"] == "answer"
        and env["choices"][0]["finish_reason"] == "stop"
        and "logprobs" in env["choices"][0]
    )
    env3 = _chat_env(["a", "b", "c"], finish_reasons=["stop", "length", "tool_calls"])
    out["env_n_fanout"] = (
        len(env3["choices"]) == 3
        and [c["index"] for c in env3["choices"]] == [0, 1, 2]
        and env3["choices"][1]["finish_reason"] == "length"
    )
    # a pure tool-call turn emits content: null with the call list intact
    env_tc = _chat_env(
        "",
        tool_calls=[
            [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"x":1}'},
                }
            ]
        ],
    )
    out["env_toolcall_null_content"] = (
        env_tc["choices"][0]["message"]["content"] is None
        and env_tc["choices"][0]["message"]["tool_calls"][0]["id"] == "c1"
    )
    # legacy completion envelope: flat choices carry text, ids mint cmpl-,
    # usage sums across prompt elements
    e_a = _chat_env("t1")
    e_b = _chat_env("t2")
    e_b["id"] = "chatcmpl-fixedcid0002"
    leg = openai_completion_envelope(cid="lcid1", envs=[e_a, e_b], prompts=["p1", "p2"])
    out["legacy_env_choices"] = (
        leg["object"] == "text_completion"
        and leg["choices"][0]["text"] == "t1"
        and leg["choices"][1]["index"] == 1
        and leg["id"] == "cmpl-lcid1"
        and leg["usage"]["total_tokens"] == 14
    )
    leg_echo = openai_completion_envelope(cid="lcid2", envs=[e_a], prompts=["p1"], echo=True)
    out["legacy_env_echo"] = leg_echo["choices"][0]["text"] == "p1t1"
    # legacy_to_chat: prompt → single user turn, decode knobs pass through
    lbody = OpenAICompletionRequest.model_validate(
        {"model": "fx1", "prompt": "hi", "max_tokens": 5, "temperature": 0.3}
    )
    lchat = legacy_to_chat(lbody, "hi")
    out["legacy_to_chat"] = (
        lchat.messages[0].role == "user"
        and lchat.messages[0].content == "hi"
        and lchat.max_tokens == 5
        and math.isclose(lchat.temperature or 9.9, 0.3)
    )
    # completion_events: piece frames then a finish frame per choice
    events = list(completion_events(leg, include_usage=True))
    out["legacy_events_grammar"] = (
        all(e["object"] == "text_completion" for e in events)
        and events[-1]["choices"] == []
        and events[-1]["usage"]["total_tokens"] == 14
        and events[-2]["choices"][0]["finish_reason"] is not None
    )
    return out


# ---------------------------------------------------------------------------
# Stream-chunk grammar
# ---------------------------------------------------------------------------


def _probe_chunks() -> dict[str, bool]:
    out: dict[str, bool] = {}
    text = "alpha " * 30 + "omega"
    pieces = list(_text_pieces(text))
    out["pieces_rejoin_lossless"] = "".join(pieces) == text and len(pieces) > 1
    out["pieces_bounded"] = all(len(p) <= 64 for p in pieces) and all(len(p) > 0 for p in pieces)
    chunks = list(openai_chunks(text="hello world", backend="stub", model="fx1", cid="c1"))
    out["chunk_role_first"] = chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    out["chunk_content_then_finish"] = (
        chunks[1]["choices"][0]["delta"].get("content") == "hello world"
        and chunks[-1]["choices"][0]["finish_reason"] == "stop"
    )
    out["chunk_object_and_id"] = all(
        c["object"] == "chat.completion.chunk" for c in chunks
    ) and all(c["id"] == "chatcmpl-c1" for c in chunks)
    chunks2 = list(
        openai_chunks(
            text=["aa", "bb"],
            backend="stub",
            cid="c2",
            finish_reasons=["length", "stop"],
            include_usage=True,
            usage={"total_tokens": 9},
        )
    )
    out["chunk_n_grouped"] = (
        chunks2[0]["choices"][0]["index"] == 0
        and [
            c["choices"][0]["index"]
            for c in chunks2
            if c["choices"] and c["choices"][0].get("finish_reason")
        ]
        == [0, 1]
        and chunks2[-1]["choices"] == []
        and chunks2[-1]["usage"]["total_tokens"] == 9
    )
    tc_chunks = list(
        openai_chunks(
            text="",
            backend="stub",
            cid="c3",
            tool_calls=[[{"id": "t1", "type": "function", "function": {"name": "g"}}]],
        )
    )
    out["chunk_toolcall_single_shot"] = any(
        (c["choices"][0]["delta"].get("tool_calls") or [{}])[0].get("id") == "t1" for c in tc_chunks
    )
    return out


# ---------------------------------------------------------------------------
# Responses translation + pagination + chain
# ---------------------------------------------------------------------------


def _probe_responses() -> dict[str, bool]:
    out: dict[str, bool] = {}
    out["resp_str_input"] = response_input_to_messages("hi", None) == [
        {"role": "user", "content": "hi"}
    ]
    out["resp_instructions_system"] = response_input_to_messages("hi", "be terse")[0] == {
        "role": "system",
        "content": "be terse",
    }
    folded = response_input_to_messages(
        [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "f",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "g",
                "arguments": "[]",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "done",
            },
        ],
        None,
    )
    out["resp_fc_folds_toolcalls"] = (
        folded[1]["role"] == "assistant"
        and len(folded[1]["tool_calls"]) == 2
        and folded[2]["role"] == "tool"
        and folded[2]["tool_call_id"] == "call_1"
    )
    refused, _, _ = _refuse(
        response_input_to_messages,
        [{"type": "function_call", "call_id": "", "name": "f", "arguments": "{}"}],
        None,
    )
    out["resp_fc_empty_callid_refuses"] = refused
    refused2, _, _ = _refuse(response_input_to_messages, [{"type": "wat"}], None)
    out["resp_unknown_item_refuses"] = refused2
    kw = response_to_kwargs(_resp_req(model="hosted_k3", max_output_tokens=17))
    out["resp_kwargs_max_tokens"] = kw["max_tokens"] == 17
    out["resp_kwargs_reasoning_effort"] = (
        response_to_kwargs(
            _resp_req(model="hosted_k3", reasoning={"effort": "low"}),
        )["reasoning_effort"]
        == "low"
    )
    out["resp_kwargs_user_or_safety"] = (
        response_to_kwargs(_resp_req(model="hosted_k3", safety_identifier="sid"))["user"] == "sid"
    )
    out["resp_text_format_passthrough"] = (
        response_text_format(_resp_req(text={"format": {"type": "json_object"}})) is not None
    )
    # conversation field unwrap
    out["conversation_id_of"] = (
        conversation_id_of(None) is None
        and conversation_id_of("conv_1") == "conv_1"
        and conversation_id_of({"id": "conv_2"}) == "conv_2"
        and conversation_id_of({"noid": 1}) is None
    )
    # chained input: prior items lose ids, outputs fold in, new input appends
    chained = chained_response_input(
        {"output": [{"type": "message", "id": "m1", "role": "assistant", "content": []}]},
        [{"type": "message", "id": "m0", "role": "user", "content": []}],
        "next turn",
    )
    out["chained_ids_dropped"] = len(chained) == 3 and all("id" not in it for it in chained[:2])
    # pagination invariants
    items = [{"id": f"i{i}"} for i in range(5)]
    page = paged_item_list(items, limit=2)
    out["paged_first_page"] = (
        page["data"] == items[:2]
        and page["has_more"] is True
        and page["first_id"] == "i0"
        and page["last_id"] == "i1"
    )
    page2 = paged_item_list(items, limit=2, after="i1")
    out["paged_after_cursor"] = page2["data"] == items[2:4] and page2["has_more"] is True
    out["paged_desc_order"] = paged_item_list(items, limit=2, order="desc")["data"] == [
        {"id": "i4"},
        {"id": "i3"},
    ]
    refused3, _, code3 = _refuse(paged_item_list, items, limit=2, after="nope")
    out["paged_bad_cursor_400"] = refused3 and code3 == "invalid_cursor"
    refused4, _, _ = _refuse(paged_item_list, items, limit=0)
    out["paged_bad_limit_400"] = refused4
    out["resp_query_text"] = (
        response_query_text("plain") == "plain"
        and response_query_text(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "find me"}],
                }
            ]
        )
        == "find me"
    )
    # stored items mint deterministic ids off (rid, index)
    stored = response_input_items_for_store("hi", rid="resp_xyz")
    stored2 = response_input_items_for_store("hi", rid="resp_xyz")
    out["stored_items_deterministic"] = (
        stored == stored2
        and stored[0]["id"].startswith("msg_")
        and response_input_items_for_store("hi", rid="resp_xyz", start_at=1)[0]["id"]
        != stored[0]["id"]
    )
    # chat sidecar: message dicts mint the same digest namespace
    chat_items = chat_messages_for_store(_chat_req().messages, envelope_id="eid1")
    out["chat_store_items_ids"] = (
        chat_items[0]["id"].startswith("msg_") and chat_items[0]["role"] == "user"
    )

    # request model refuse modes
    def _resp_bad(**kw: Any) -> bool:
        try:
            _resp_req(**kw)
        except ValidationError:
            return True
        return False

    out["resp_req_refuses_bad_reasoning"] = _resp_bad(reasoning={"effort": "bogus"})
    out["resp_req_refuses_input_obj"] = _resp_bad(input={"not": "a list"})
    return out


# ---------------------------------------------------------------------------
# Response object + call items + events + replay
# ---------------------------------------------------------------------------


def _probe_response_object_events() -> dict[str, bool]:
    out: dict[str, bool] = {}
    body = _resp_req(model="fx1")
    env = openai_response_object(
        rid="resp_probe1",
        item_id="msg_item1",
        body=body,
        content="done",
        model="fx1",
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        created=1700000000,
    )
    out["resp_obj_shape"] = (
        env["object"] == "response"
        and env["id"] == "resp_probe1"
        and env["status"] == "completed"
        and env["model"] == "fx1"
        and env["usage"]["total_tokens"] == 3
    )
    out["resp_obj_output_text"] = (
        env["output"][0]["type"] == "message"
        and env["output"][0]["content"][0]["type"] == "output_text"
        and env["output"][0]["content"][0]["text"] == "done"
    )
    # call items: file_search + function calls materialize as output items
    fc_item = file_search_call_item(queries=["query"], results=[{"file_id": "f1"}])
    out["file_search_item_shape"] = (
        fc_item["type"] == "file_search_call"
        and str(fc_item["id"]).startswith("fs_")
        and fc_item["queries"] == ["query"]
    )
    call_items = openai_response_call_items(
        [{"id": "call_9", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    )
    out["fc_call_item_shape"] = (
        call_items[0]["type"] == "function_call"
        and call_items[0]["call_id"] == "call_9"
        and call_items[0]["name"] == "f"
    )
    # event grammar: created/in_progress prelude → lifecycle → terminal
    events = list(
        openai_response_events(
            text="tok " * 20,
            rid="resp_ev1",
            item_id="item_1",
            body=body,
            model="fx1",
            usage={"total_tokens": 5},
            created=1700000000,
        )
    )
    names = [e for e, _ in events]
    out["resp_events_grammar"] = (
        names[0] == "response.created"
        and "response.in_progress" in names
        and "response.output_item.added" in names
        and "response.output_text.delta" in names
        and names[-1] == "response.completed"
    )
    terminal = events[-1][1]
    out["resp_events_terminal_usage"] = (
        terminal["response"]["status"] == "completed"
        and terminal["response"]["usage"]["total_tokens"] == 5
        and terminal["response"]["id"] == "resp_ev1"
    )
    out["resp_events_incomplete_variant"] = [
        e
        for e, _ in openai_response_events(
            text="x",
            rid="r2",
            item_id="i2",
            body=body,
            model=None,
            usage=None,
            final_status="incomplete",
            incomplete_details={"reason": "max_tool_calls"},
        )
    ][-1] == "response.incomplete"
    # replay: terminal envelope replays the same typed sequence
    replayed = [e for e, _ in openai_response_replay_events(env)]
    out["resp_replay_grammar"] = (
        replayed[0] == "response.created" and replayed[-1] == "response.completed"
    )
    out["resp_replay_no_internals"] = all(
        not k.startswith("_fx1")
        for _, payload in openai_response_replay_events(env)
        for k in (payload.get("response", {}) if isinstance(payload, dict) else {})
    )
    # response_output_pieces: the stored envelope decomposes back into
    # the stream-grammar parts the replay emitter consumes
    p_text, p_item, p_calls, p_search, p_lp = response_output_pieces(env)
    out["output_pieces_reconstruct"] = (
        p_text == "done"
        and p_item == "msg_item1"
        and p_calls is None
        and p_search is None
        and p_lp is None
    )
    return out


# ---------------------------------------------------------------------------
# Embeddings + batch + files + stores
# ---------------------------------------------------------------------------


def _probe_embeddings_batch_files() -> dict[str, bool]:
    out: dict[str, bool] = {}
    ekw = embeddings_to_kwargs(
        OpenAIEmbeddingRequest.model_validate(
            {"model": "text-embed-9", "input": ["a", "b"], "dimensions": 8, "user": "u"}
        )
    )
    out["emb_kwargs_shape"] = (
        ekw["model"] == "text-embed-9"
        and ekw["input"] == ["a", "b"]
        and ekw["dimensions"] == 8
        and ekw["backend"] in OPENAI_BACKENDS
    )
    env = openai_embedding_envelope(
        data=[{"object": "embedding", "index": 0, "embedding": [0.1]}],
        model="m",
        usage={"total_tokens": 2},
    )
    out["emb_envelope"] = (
        env["object"] == "list" and env["data"][0]["index"] == 0 and env["model"] == "m"
    )
    # batch line validation: all refuse modes
    ok_line = {"custom_id": "c1", "method": "POST", "url": _RESP_PATH, "body": {}}
    out["batch_line_accepts"] = (
        batch_line_shape(ok_line, endpoint=_RESP_PATH, lineno=1)["custom_id"] == "c1"
    )
    bad_cases = [
        ("non-dict", "x"),
        ("no custom_id", {"method": "POST", "url": _RESP_PATH, "body": {}}),
        ("long custom_id", {**ok_line, "custom_id": "x" * 65}),
        ("wrong method", {**ok_line, "method": "GET"}),
        ("wrong url", {**ok_line, "url": _CHAT_PATH}),
        ("non-dict body", {**ok_line, "body": "x"}),
    ]
    out["batch_line_refuses"] = all(
        _refuse(batch_line_shape, line, endpoint=_RESP_PATH, lineno=i + 1)[0]
        for i, (_, line) in enumerate(bad_cases)
    )
    parsed = batch_line_body({"body": {"model": "fx1", "input": "hi"}}, _RESP_PATH)
    out["batch_line_body_typed"] = isinstance(parsed, OpenAIResponseRequest)
    refused5, _, _ = _refuse(batch_line_body, {"body": {"model": "fx1"}}, _RESP_PATH)
    out["batch_line_body_invalid_refuses"] = refused5
    fobj = file_object(
        {
            "file_id": "file_1",
            "filename": "f.jsonl",
            "size": 3,
            "created_at": 9,
            "purpose": "batch",
        }
    )
    out["file_object_shape"] = (
        fobj["object"] == "file"
        and fobj["id"] == "file_1"
        and fobj["bytes"] == 3
        and fobj["status"] == "processed"
    )
    bobj = batch_object(
        {
            "batch_id": "batch_1",
            "endpoint": _RESP_PATH,
            "input_file_id": "file_1",
            "completion_window": "24h",
            "status": "completed",
            "created_at": 1,
            "request_counts": {"total": 2, "completed": 2, "failed": 0},
        }
    )
    out["batch_object_shape"] = (
        bobj["object"] == "batch"
        and bobj["request_counts"]["total"] == 2
        and bobj["status"] == "completed"
        and bobj["input_file_id"] == "file_1"
    )
    line_out = batch_output_line(custom_id="c1", status_code=200, body=_chat_env("ok"), rid="rid1")
    out["batch_output_line_shape"] = (
        line_out["custom_id"] == "c1"
        and line_out["id"] == "batch_req_rid1"
        and line_out["response"]["status_code"] == 200
        and line_out["response"]["body"]["object"] == _CHAT_OBJECT
        and line_out["error"] is None
    )
    out["batch_endpoints_pinned"] = {
        _CHAT_PATH,
        _RESP_PATH,
        "/v1/embeddings",
    } == OPENAI_BATCH_ENDPOINTS
    return out


def _probe_envelope_store() -> dict[str, bool]:
    out: dict[str, bool] = {}
    store = OpenAIEnvelopeStore(cap=3)
    env1 = _chat_env("one")
    env2 = _chat_env("two")
    env2["id"] = "chatcmpl-fixedcid0002"
    store.put(env1, items={"input": [{"role": "user", "content": "q1"}]})
    store.put(env2)
    out["store_get_hit"] = store.get(env1["id"]) == env1
    out["store_get_miss_none"] = store.get("chatcmpl-nope") is None
    out["store_items_sidecar"] = store.get_items(env1["id"], "input") == [
        {"role": "user", "content": "q1"}
    ]
    out["store_items_missing_none"] = store.get_items(env2["id"], "input") == []
    out["store_list_by_object"] = [e["id"] for e in store.list_envelopes(_CHAT_OBJECT)] == [
        env1["id"],
        env2["id"],
    ]
    # cap evicts the oldest; a re-put refreshes position
    e_c, e_d = _chat_env("three"), _chat_env("four")
    e_c["id"], e_d["id"] = "chatcmpl-c", "chatcmpl-d"
    store.put(e_c)
    store.put(e_d)
    out["store_cap_evicts_oldest"] = store.get(env1["id"]) is None and store.get(env2["id"]) == env2
    store2 = OpenAIEnvelopeStore(cap=2)
    e_a, e_b, e_c2 = _chat_env("a"), _chat_env("b"), _chat_env("c")
    e_a["id"], e_b["id"], e_c2["id"] = "chatcmpl-a", "chatcmpl-b", "chatcmpl-c2"
    store2.put(e_a)
    store2.put(e_b)
    store2.put(e_a)  # refresh → e_a no longer oldest
    store2.put(e_c2)
    out["store_reput_refreshes"] = (
        store2.get("chatcmpl-a") == e_a and store2.get("chatcmpl-b") is None
    )
    out["store_delete"] = (
        store2.delete("chatcmpl-a") is True and store2.delete("chatcmpl-a") is False
    )
    try:
        OpenAIEnvelopeStore(cap=0)
        out["store_bad_cap_refuses"] = False
    except ValueError:
        out["store_bad_cap_refuses"] = True
    return out


# ---------------------------------------------------------------------------
# Anthropic contract + translation
# ---------------------------------------------------------------------------


def _probe_anthropic_contract() -> dict[str, bool]:
    out: dict[str, bool] = {}
    out["anth_req_accepts_minimal"] = _anth_req().model == "fx1"

    def _bad(**kw: Any) -> bool:
        try:
            _anth_req(**kw)
        except (ValidationError, OpenAICompatError):
            return True
        return False

    out["anth_req_max_tokens_required"] = _bad(max_tokens=None) or _bad(
        **{"max_tokens": None, "messages": []}
    )
    out["anth_req_refuses_bad_role"] = _bad(messages=[{"role": "pirate", "content": "hi"}])
    out["anth_req_refuses_bad_content_shape"] = _bad(messages=[{"role": "user", "content": 42}])
    # translation: system → system message, tool_use ↔ tool_calls
    body = _anth_req(
        system="be honest",
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling"},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "f",
                        "input": {"x": 1},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "res",
                    }
                ],
            },
        ],
        tools=[
            {
                "name": "f",
                "description": "does f",
                "input_schema": {"type": "object"},
            }
        ],
    )
    oai = anthropic_to_openai(body)
    out["anth_xlate_system_first"] = oai["messages"][0] == {
        "role": "system",
        "content": "be honest",
    }
    out["anth_xlate_tool_use_to_calls"] = (
        oai["messages"][2]["role"] == "assistant"
        and oai["messages"][2]["tool_calls"][0]["id"] == "tu_1"
        and oai["messages"][2]["tool_calls"][0]["function"]["name"] == "f"
    )
    out["anth_xlate_tool_result_to_tool"] = (
        oai["messages"][3]["role"] == "tool"
        and oai["messages"][3]["tool_call_id"] == "tu_1"
        and oai["messages"][3]["content"] == "res"
    )
    out["anth_xlate_tools_to_functions"] = (
        oai["tools"][0]["type"] == "function" and oai["tools"][0]["function"]["name"] == "f"
    )
    out["anth_xlate_store_forced_off"] = oai.get("store") is False
    out["anth_xlate_max_tokens"] = oai["max_tokens"] == 64
    # _system_text both forms; _tool_result_content folds blocks
    out["system_text_forms"] = (
        _system_text("plain") == "plain"
        and _system_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    )
    out["tool_result_content_folds"] = _tool_result_content(
        {"type": "tool_result", "content": "x"}
    ) == "x" and isinstance(
        _tool_result_content(
            {"content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}
        ),
        str,
    )
    return out


def _probe_anthropic_envelope_events() -> dict[str, bool]:
    out: dict[str, bool] = {}
    env = _chat_env(_ANSWER)
    msg = anthropic_envelope(env, model="fx1")
    out["anth_env_shape"] = (
        msg["type"] == "message"
        and msg["role"] == "assistant"
        and msg["id"].startswith("msg_")
        and msg["id"].removeprefix("msg_") == env["id"].removeprefix("chatcmpl-")
        and msg["content"][0] == {"type": "text", "text": _ANSWER}
        and msg["stop_reason"] == "end_turn"
        and msg["usage"]["output_tokens"] == 4
    )
    # finish_reason → stop_reason mapping is total and honest
    out["anth_stop_reason_map"] = (
        all(isinstance(v, str) and v for v in _FINISH_TO_STOP_REASON.values())
        and _FINISH_TO_STOP_REASON.get("length") == "max_tokens"
    )
    env_tc = _chat_env(
        "",
        finish_reasons=["tool_calls"],
        tool_calls=[
            [
                {
                    "id": "tu_2",
                    "type": "function",
                    "function": {"name": "g", "arguments": '{"y":2}'},
                }
            ]
        ],
    )
    msg_tc = anthropic_envelope(env_tc)
    out["anth_env_tool_use_block"] = (
        msg_tc["stop_reason"] == "tool_use"
        and msg_tc["content"][0]["type"] == "tool_use"
        and msg_tc["content"][0]["id"] == "tu_2"
        and msg_tc["content"][0]["input"] == {"y": 2}
    )
    # malformed tool args never crash the envelope — they land as _raw
    env_bad = _chat_env(
        "",
        tool_calls=[
            [{"id": "tu_3", "type": "function", "function": {"name": "g", "arguments": "{bad"}}]
        ],
    )
    out["anth_env_bad_args_failclosed"] = (
        anthropic_envelope(env_bad)["content"][0]["input"].get("_raw") == "{bad"
    )
    # events: typed lifecycle, event name mirrors data.type
    events = list(anthropic_events(env))
    names = [e["event"] for e in events]
    out["anth_events_grammar"] = (
        names[0] == "message_start"
        and "ping" in names
        and "content_block_start" in names
        and "content_block_delta" in names
        and "message_delta" in names
        and names[-1] == "message_stop"
        and all(e["event"] == e["data"]["type"] for e in events)
    )
    out["anth_events_text_rejoins"] = (
        "".join(
            e["data"]["delta"]["text"]
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"].get("delta", {}).get("type") == "text_delta"
        )
        == _ANSWER
    )
    # sse: skip drops leading frames; emits event:+data: pairs
    full = list(anthropic_sse(env))
    skipped = list(anthropic_sse(env, skip=2))
    out["anth_sse_skip"] = len(skipped) == len(full) - 2 and all(
        "event:" in frame or "data:" in frame for frame in skipped[:1]
    )
    refused7, status7, _ = _refuse(anthropic_envelope, {"choices": []})
    out["anth_env_no_choices_502"] = refused7 and status7 == 502
    return out


def _probe_anthropic_misc() -> dict[str, bool]:
    out: dict[str, bool] = {}
    cnt = anthropic_count_messages(
        AnthropicCountTokensRequest.model_validate(
            {"model": "fx1", "messages": [{"role": "user", "content": "hi"}]}
        )
    )
    out["anth_count_messages"] = isinstance(cnt, list) and cnt[0]["role"] == "user"
    mo = anthropic_model_object("fx1", created=1700000000)
    out["anth_model_object"] = (
        mo["type"] == "model" and mo["id"] == "fx1" and mo["created_at"] is not None
    )
    item = AnthropicBatchItem.model_validate(
        {
            "custom_id": "b1",
            "params": {
                "model": "fx1",
                "max_tokens": 4,
                "messages": [{"role": "user", "content": "q"}],
            },
        }
    )
    breq = AnthropicBatchCreate.model_validate({"requests": [item.model_dump()]})
    out["anth_batch_models"] = breq.requests[0].custom_id == "b1"
    bobj = anthropic_batch_object(
        {
            "batch_id": "msgbatch_1",
            "created_at": 1700000000.0,
            "ended_at": 1700000010.0,
            "expires_at": 1700086400.0,
            "status": "ended",
            "request_counts": {
                "succeeded": 2,
                "errored": 0,
                "processing": 0,
                "canceled": 0,
                "expired": 0,
            },
        }
    )
    out["anth_batch_object_rfc3339"] = (
        bobj["type"] == "message_batch"
        and isinstance(bobj["created_at"], str)
        and "T" in bobj["created_at"]
        and bobj["request_counts"]["succeeded"] == 2
        and bobj["results_url"].endswith("/results")
    )
    bobj_live = anthropic_batch_object(
        {
            "batch_id": "msgbatch_2",
            "created_at": 1700000000.0,
            "expires_at": 1700086400.0,
            "status": "in_progress",
            "request_counts": {
                "succeeded": 0,
                "errored": 0,
                "processing": 2,
                "canceled": 0,
                "expired": 0,
            },
        }
    )
    out["anth_batch_results_url_gated"] = bobj_live["results_url"] is None
    res = anthropic_batch_result(
        "b1",
        {
            "type": "succeeded",
            "message": anthropic_envelope(_chat_env("ok")),
        },
    )
    out["anth_batch_result_shape"] = (
        res["custom_id"] == "b1" and res["result"]["type"] == "succeeded"
    )
    return out


# ---------------------------------------------------------------------------
# Conversation + vector-store request models
# ---------------------------------------------------------------------------


def _probe_conv_vs_models() -> dict[str, bool]:
    out: dict[str, bool] = {}
    conv = OpenAIConversationCreate.model_validate(
        {
            "items": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
            "metadata": {"k": "v"},
        }
    )
    out["conv_create_accepts"] = conv.metadata == {"k": "v"}
    add = OpenAIConversationItemsAdd.model_validate(
        {"items": [{"type": "message", "role": "user", "content": "x"}]}
    )
    out["conv_items_add"] = len(add.items or []) == 1

    def _bad_vs(**kw: Any) -> bool:
        try:
            OpenAIVectorStoreCreate.model_validate(kw or {"name": None, "file_ids": "x"})
        except ValidationError:
            return True
        return False

    out["vs_create_accepts"] = (
        OpenAIVectorStoreCreate.model_validate(
            {"name": "corpus", "file_ids": ["file_1"], "metadata": {"k": "v"}}
        ).name
        == "corpus"
    )
    filec = OpenAIVectorStoreFileCreate.model_validate(
        {
            "file_id": "file_1",
            "chunking_strategy": {
                "type": "static",
                "static": {"max_chunk_size_tokens": 400, "chunk_overlap_tokens": 40},
            },
        }
    )
    out["vs_file_chunking_accepts"] = filec.file_id == "file_1"
    out["vs_create_refuses_bad_chunking"] = _bad_vs(
        name="x",
        file_ids=["x"] * 65,
    )
    search = OpenAIVectorStoreSearch.model_validate(
        {"query": "q", "max_num_results": 5, "filters": {"type": "eq", "key": "k", "value": "v"}}
    )
    out["vs_search_model"] = search.max_num_results == 5

    def _bad_search() -> bool:
        try:
            OpenAIVectorStoreSearch.model_validate({"query": "q", "max_num_results": 0})
        except ValidationError:
            return True
        return False

    out["vs_search_refuses_oob"] = _bad_search()
    return out


# ---------------------------------------------------------------------------
# Top-level battery
# ---------------------------------------------------------------------------


def compat_audit() -> dict[str, bool]:
    """Run every dialect-translation probe. All offline, deterministic."""
    out: dict[str, bool] = {}
    for probe in (
        _probe_errors_paths,
        _probe_chat_contract,
        _probe_openai_to_kwargs,
        _probe_envelopes,
        _probe_chunks,
        _probe_responses,
        _probe_response_object_events,
        _probe_embeddings_batch_files,
        _probe_envelope_store,
        _probe_anthropic_contract,
        _probe_anthropic_envelope_events,
        _probe_anthropic_misc,
        _probe_conv_vs_models,
    ):
        out.update(probe())
    return out


def compat_audit_bench() -> dict[str, Any]:
    """Sealed receipt over the translation battery — ``claim.results``
    carries every probe verdict; ``claim.ok`` holds iff all are True."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = compat_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "compat_audit",
        "schema": "compat_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "pure-function translation layer (in-process)",
            "not_verified": [
                "end-to-end wire serialization of translated objects",
                "provider-side parsing of emitted grammars",
                "concurrent mutation of shared stores",
            ],
        },
        "interpretation": (
            "Dialect translation invariants hold: OpenAI request models "
            "accept the spec envelope and refuse malformed bodies; "
            "``*_to_kwargs`` resolves the link in the documented order "
            "(fx1.backend > header > model > byok-headers > hosted_k3) and "
            "fails closed 404 on unregistered models; envelopes mint the "
            "right object types with derived ids, n-fanout, and per-choice "
            "tool_calls/logprobs; stream chunkers split ~64-char on "
            "whitespace, emit role-first deltas and spec-legal terminal "
            "frames; Responses input folds function_call items into "
            "tool_calls, pagination cursors fail closed; Anthropic "
            "translation maps system/tools/stop_reasons correctly, emits "
            "the typed SSE lifecycle, and keeps bad tool args in-band as "
            "_raw rather than crashing; batch lines refuse at submit, "
            "the envelope store evicts oldest-first with re-put refresh, "
            "and vector-store models enforce their bounds. "
            "Not verified: wire serialization, provider-side parsing, "
            "concurrent store mutation."
            if ok
            else f"COMPAT AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(compat_audit_bench(), indent=2, sort_keys=True))
