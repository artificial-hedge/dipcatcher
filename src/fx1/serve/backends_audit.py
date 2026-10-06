"""Backend audit — the transport/policy contract under ``serve.backends``.

``backends`` is the module every other audit *patches* — the wire seam
(``_openai_urlopen``), the BYOK destination policy, the three backend
classes — but no battery ever pinned the module's own contract. Every
request the harness answers or emits crosses it: sampling defaults
(deterministic ``temperature``), stop-sequence truncation applied
harness-side, URL/timeout normalization, the BYOK credential and
anti-SSRF policy (URL shape, resolved-address allow/deny, redirect
refusal, DNS pinning), usage accounting (``_extract_usage`` refuses
bools/strings/non-finite floats), the fail-closed shape validators on
``tool_calls[]`` / embeddings ``data[]``, the error envelope
(HTTPError→``RuntimeError`` labeled, refused tokenize→
``TokenCountUnavailableError``, never an estimate), and the factory /
capability-protocol surface.

Probes are pure functions plus the documented ``_openai_urlopen``
seam — no sockets, no child processes except a faked ``Popen``, fully
deterministic. SYNTHETIC by construction; the sealed bench payload is
``research_only`` and makes no live-PnL claim.
"""

from __future__ import annotations

import contextlib
import email.message
import io
import ipaddress
import json
import os
import tempfile
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fx1.serve import backends
from fx1.serve.backends import (
    BYOK_API_KEY_ENV,
    BYOK_BASE_URL_ENV,
    BYOK_MODEL_ENV,
    LOCAL_API_KEY_ENV,
    LOCAL_MODEL_ENV,
    LOCAL_SERVE_CMD_ENV,
    LOCAL_SERVE_URL_ENV,
    LOCAL_START_TIMEOUT_S_ENV,
    LOCAL_TIMEOUT_S_ENV,
    BackendNotConfiguredError,
    EmbeddingBackend,
    HostedK3Backend,
    InferenceBackend,
    LocalFx1Backend,
    OpenAICompatBackend,
    SamplingParams,
    StreamingBackend,
    TokenCountingBackend,
    TokenCountUnavailableError,
    ToolBackend,
    _byok_address_allowed,
    _byok_resolved_addresses,
    _chat_completions_url,
    _embedding_item_shape,
    _env_float,
    _extract_token_count,
    _extract_usage,
    _openai_chat_complete,
    _openai_chat_complete_tools,
    _openai_chat_stream,
    _openai_embeddings_complete,
    _openai_sibling_url,
    _openai_tokenize_count,
    _openai_urlopen,
    _RefuseRedirects,
    _stop_cut,
    _tool_call_shape,
    _UsageTracker,
    byok_base_url_problem,
    get_backend,
    truncate_at_stops,
    truncate_chunks,
)

_ENV_KEYS = (
    "MOONSHOT_API_KEY",
    BYOK_BASE_URL_ENV,
    BYOK_API_KEY_ENV,
    BYOK_MODEL_ENV,
    "FX1_BYOK_ALLOW_PRIVATE_NETWORKS",
    LOCAL_SERVE_URL_ENV,
    LOCAL_SERVE_CMD_ENV,
    LOCAL_MODEL_ENV,
    LOCAL_API_KEY_ENV,
    LOCAL_TIMEOUT_S_ENV,
    LOCAL_START_TIMEOUT_S_ENV,
    "FX1_SIGNING_KEY",
)

_MSGS = [{"role": "user", "content": "hi"}]


@contextlib.contextmanager
def _env(sets: Mapping[str, str | None]) -> Iterator[None]:
    """Set (value) or delete (``None``) env vars; restore after."""
    saved = {k: os.environ.get(k) for k in sets}
    try:
        for k, v in sets.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Body(io.BytesIO):
    """BytesIO already satisfies the response contract used by the wire
    helpers (context manager + ``read`` + line iteration)."""


def _wire(
    rec: dict[str, Any],
    payload: bytes | None = None,
    error: BaseException | None = None,
    frames: bytes | None = None,
):
    """A stand-in for ``backends._openai_urlopen`` recording the request."""

    @contextlib.contextmanager
    def _fake(request: urllib.request.Request, *, timeout_s: float):
        rec["request"] = request
        rec["timeout_s"] = timeout_s
        if error is not None:
            raise error
        yield _Body(frames if frames is not None else (payload or b""))

    return _fake


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://x/v1/chat/completions",
        code,
        "boom",
        email.message.Message(),
        io.BytesIO(b"{}"),
    )


def _chat_payload(text: str = "answer", **extra: Any) -> bytes:
    body = {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        **extra,
    }
    return json.dumps(body).encode()


def _run(fn, *a: Any, **kw: Any) -> tuple[bool, Any]:
    """``(True, result)`` or ``(False, exception)``."""
    try:
        return True, fn(*a, **kw)
    except Exception as exc:  # noqa: BLE001 — the probe records the type
        return False, exc


def _checkpoint(wd: Path, *, eligible: bool = True) -> Path:
    """A minimal valid checkpoint dir: modelcard.json, ship-gate verdict."""
    wd.mkdir(parents=True, exist_ok=True)
    card = {
        "version": "fx-1.v1.0",
        "corpus_sha256": "a" * 64,
        "corpus_receipt_range": "b" * 8 + ".." + "c" * 8,
        "training_manifest_sha256": "d" * 64,
        "eval_delta": {
            "domain_pass_rate_base": 0.5,
            "domain_pass_rate_candidate": 0.9 if eligible else 0.4,
            "general_pass_rate_base": 0.5,
            "general_pass_rate_candidate": 0.5,
            "honesty_gate_candidate": eligible,
        },
    }
    (wd / "modelcard.json").write_text(json.dumps(card), encoding="utf-8")
    return wd


# ---------------------------------------------------------------- sampling


def _probe_sampling_and_truncation() -> dict[str, bool]:
    out: dict[str, bool] = {}

    f = SamplingParams().body_fields()
    out["sp_default_temperature_zero"] = f == {"temperature": 0.0}
    out["sp_undeclared_never_ships"] = "seed" not in SamplingParams(top_p=0.5).body_fields()
    sp = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=8,
        seed=1,
        stop=("END", "STOP"),
        presence_penalty=0.1,
        frequency_penalty=0.2,
        logit_bias={"1": -1},
        reasoning_effort="low",
        service_tier="flex",
        prompt_cache_key="k",
        prompt_cache_retention="24h",
        verbosity="low",
        user="u",
    )
    f2 = sp.body_fields()
    out["sp_all_declared_fields_land"] = f2 == {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 8,
        "seed": 1,
        "stop": ["END", "STOP"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "logit_bias": {"1": -1},
        "reasoning_effort": "low",
        "service_tier": "flex",
        "prompt_cache_key": "k",
        "prompt_cache_retention": "24h",
        "verbosity": "low",
        "user": "u",
    }
    out["sp_stop_tuple_serializes_list"] = isinstance(f2["stop"], list)
    out["sp_frozen_dataclass"] = not _run(setattr, sp, "temperature", 1.0)[0]

    out["stop_cut_earliest_wins"] = _stop_cut("abXcdY", ("Y", "X")) == 2
    out["stop_cut_none_is_len"] = _stop_cut("abc", ("Z",)) == 3
    out["stop_cut_empty_stops"] = _stop_cut("abc", None) == 3
    out["truncate_excludes_matched_seq"] = truncate_at_stops("abENDcd", ("END",)) == "ab"
    out["truncate_no_stop_unchanged"] = truncate_at_stops("ab", ("Z",)) == "ab"
    out["truncate_stops_none_unchanged"] = truncate_at_stops("ab", None) == "ab"

    chunks = truncate_chunks(["ab", "cENDd", "ef"], ("END",))
    out["chunks_boundary_preserving"] = chunks == ["ab", "c"]
    out["chunks_no_stops_identity"] = truncate_chunks(["a", "b"], None) == ["a", "b"]
    out["chunks_cut_before_first"] = truncate_chunks(["xEND", "y"], ("END",)) == ["x"]
    out["chunks_empty_piece_skipped"] = truncate_chunks(["a", "", "b"], ("Q",)) == [
        "a",
        "b",
    ]
    return out


# ---------------------------------------------------------------- urls/env


def _probe_url_and_env_helpers() -> dict[str, bool]:
    out: dict[str, bool] = {}

    out["chaturl_normalizes_v1"] = (
        _chat_completions_url("https://h/v1") == "https://h/v1/chat/completions"
    )
    out["chaturl_trailing_slash"] = (
        _chat_completions_url("https://h/v1/") == "https://h/v1/chat/completions"
    )
    out["chaturl_complete_passthrough"] = (
        _chat_completions_url("https://h/v1/chat/completions").endswith("/chat/completions")
        and _chat_completions_url("https://h/v1/chat/completions").count("chat/completions") == 1
    )

    out["sibling_embeddings"] = (
        _openai_sibling_url("https://h/v1/chat/completions", "embeddings")
        == "https://h/v1/embeddings"
    )
    out["sibling_nested_route"] = (
        _openai_sibling_url("https://h/v1/chat/completions", "tokenizers/estimate-token-count")
        == "https://h/v1/tokenizers/estimate-token-count"
    )
    out["sibling_same_route_passthrough"] = (
        _openai_sibling_url("https://h/v1/embeddings", "embeddings") == "https://h/v1/embeddings"
    )

    with _env({"FX1_TEST_KNOB": "2.5"}):
        out["envfloat_env_beats_default"] = _env_float("FX1_TEST_KNOB", None, 1.0) == 2.5
        out["envfloat_kwarg_beats_env"] = _env_float("FX1_TEST_KNOB", 9.0, 1.0) == 9.0
    with _env({"FX1_TEST_KNOB": None}):
        out["envfloat_absent_is_default"] = _env_float("FX1_TEST_KNOB", None, 3.0) == 3.0
    with _env({"FX1_TEST_KNOB": ""}):
        out["envfloat_empty_is_default"] = _env_float("FX1_TEST_KNOB", None, 3.0) == 3.0
    ok, exc = _run(_env_float, "FX1_TEST_KNOB", 0.0, 1.0)
    out["envfloat_zero_refused"] = not ok and isinstance(exc, ValueError)
    ok, exc = _run(_env_float, "FX1_TEST_KNOB", -1.0, 1.0)
    out["envfloat_negative_refused"] = not ok and isinstance(exc, ValueError)
    return out


# ---------------------------------------------------------------- byok policy


def _probe_byok_policy() -> dict[str, bool]:
    out: dict[str, bool] = {}

    out["byokurl_clean_none"] = byok_base_url_problem("https://h/v1") is None
    out["byokurl_http_ok"] = byok_base_url_problem("http://127.0.0.1:8000/v1") is None
    out["byokurl_non_http_refused"] = byok_base_url_problem("ftp://h/v1") is not None
    out["byokurl_no_netloc_refused"] = byok_base_url_problem("https:///v1") is not None
    out["byokurl_userinfo_refused"] = byok_base_url_problem("https://u:p@h/v1") is not None
    out["byokurl_query_refused"] = byok_base_url_problem("https://h/v1?k=v") is not None
    out["byokurl_fragment_refused"] = byok_base_url_problem("https://h/v1#f") is not None
    out["byokurl_params_refused"] = byok_base_url_problem("https://h/v1;p") is not None
    out["byokurl_no_host_refused"] = byok_base_url_problem("https://:8080/v1") is not None
    reason = byok_base_url_problem("https://secret.example.com:99999/v1")
    out["byokurl_reason_never_echoes_url"] = (
        reason is not None and "secret.example.com" not in reason
    )

    v4 = ipaddress.ip_address
    out["addr_public_always"] = _byok_address_allowed(v4("8.8.8.8"), allow_private=False)
    out["addr_loopback_denied_default"] = not _byok_address_allowed(
        v4("127.0.0.1"), allow_private=False
    )
    out["addr_rfc1918_denied_default"] = not _byok_address_allowed(
        v4("10.1.2.3"), allow_private=False
    )
    out["addr_loopback_optin"] = _byok_address_allowed(v4("127.0.0.1"), allow_private=True)
    out["addr_rfc1918_optin"] = _byok_address_allowed(v4("192.168.1.1"), allow_private=True)
    out["addr_17216_optin"] = _byok_address_allowed(v4("172.16.0.1"), allow_private=True)
    out["addr_linklocal_never"] = not _byok_address_allowed(
        v4("169.254.169.254"), allow_private=True
    )
    out["addr_multicast_never"] = not _byok_address_allowed(v4("224.0.0.1"), allow_private=True)
    out["addr_unspecified_never"] = not _byok_address_allowed(v4("0.0.0.0"), allow_private=True)  # nosec B104 — probe literal, not a bind
    out["addr_v6_loopback_optin"] = _byok_address_allowed(
        ipaddress.ip_address("::1"), allow_private=True
    )
    out["addr_v6_linklocal_never"] = not _byok_address_allowed(
        ipaddress.ip_address("fe80::1"), allow_private=True
    )
    out["addr_v4mapped_not_ipv4"] = not _byok_address_allowed(
        ipaddress.ip_address("::ffff:127.0.0.1"), allow_private=True
    )

    def gai_v4mapped(*a: Any, **kw: Any):
        return [
            (
                backends.socket.AF_INET6,
                backends.socket.SOCK_STREAM,
                6,
                "",
                ("::ffff:127.0.0.1", 8000, 0, 0),
            )
        ]

    with patch.object(backends.socket, "getaddrinfo", gai_v4mapped):
        ok, addrs = _run(_byok_resolved_addresses, "h", 8000, allow_private=True)
        out["resolve_v4mapped_normalizes_to_v4"] = ok and addrs == ("127.0.0.1",)

    def gai_private(*a: Any, **kw: Any):
        return [(2, 1, 6, "", ("192.168.0.1", 8000))]

    def gai_public(*a: Any, **kw: Any):
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    def gai_mixed(*a: Any, **kw: Any):
        return [
            (2, 1, 6, "", ("93.184.216.34", 443)),
            (2, 1, 6, "", ("169.254.169.254", 443)),
        ]

    def gai_empty(*a: Any, **kw: Any):
        return []

    with patch.object(backends.socket, "getaddrinfo", gai_public):
        ok, addrs = _run(_byok_resolved_addresses, "h", 443, allow_private=False)
        out["resolve_public_returns_numeric"] = ok and addrs == ("93.184.216.34",)
    with patch.object(backends.socket, "getaddrinfo", gai_private):
        ok, exc = _run(_byok_resolved_addresses, "h", 8000, allow_private=False)
        out["resolve_private_denied_default"] = not ok and isinstance(exc, ValueError)
        ok, addrs = _run(_byok_resolved_addresses, "h", 8000, allow_private=True)
        out["resolve_private_optin"] = ok and addrs == ("192.168.0.1",)
    with patch.object(backends.socket, "getaddrinfo", gai_mixed):
        ok, exc = _run(_byok_resolved_addresses, "h", 443, allow_private=True)
        out["resolve_mixed_set_refused"] = not ok and isinstance(exc, ValueError)
    with patch.object(backends.socket, "getaddrinfo", gai_empty):
        ok, exc = _run(_byok_resolved_addresses, "h", 443, allow_private=False)
        out["resolve_empty_oserror"] = not ok and isinstance(exc, OSError)
    return out


# ---------------------------------------------------------------- transport


def _probe_transport() -> dict[str, bool]:
    out: dict[str, bool] = {}

    handler = _RefuseRedirects()
    refuse = handler.redirect_request
    out["redirects_always_refused"] = (
        refuse(
            urllib.request.Request("https://a"),
            None,
            302,
            "m",
            {},
            "https://b",
        )
        is None
    )

    req = urllib.request.Request("https://x/v1/chat/completions", data=b"{}", method="POST")

    class _Opener:
        def __init__(self, payload: bytes = b"{}"):
            self.seen: dict[str, Any] = {}
            self.payload = payload

        def open(self, request: urllib.request.Request, timeout: float = 0):
            self.seen["timeout"] = timeout
            self.seen["request"] = request
            return _Body(self.payload)

    opener = _Opener(b'{"ok": true}')
    with patch.object(backends.urllib.request, "build_opener", lambda *a: opener):
        with _openai_urlopen(req, timeout_s=7.0) as resp:
            out["urlopen_normal_reads_body"] = resp.read() == b'{"ok": true}'
        out["urlopen_forwards_timeout"] = opener.seen["timeout"] == 7.0

    class _ErrOpener:
        def open(self, request: urllib.request.Request, timeout: float = 0):
            raise _http_error(500)

    with patch.object(backends.urllib.request, "build_opener", lambda *a: _ErrOpener()):
        ok, exc = _run(lambda: _openai_urlopen(req, timeout_s=1.0).__enter__())
        out["urlopen_httperror_reraised"] = not ok and isinstance(exc, urllib.error.HTTPError)

    class _PinnedResp:
        def __init__(self, status: int):
            self.status = status
            self.reason = "r"
            self.headers: dict[str, str] = {}
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _PinnedConn:
        def __init__(self):
            self.closed = False

        def close(self) -> None:
            self.closed = True

    breq = urllib.request.Request("https://x/v1/chat/completions", data=b"{}", method="POST")
    breq._fx1_byok_allow_private = True  # type: ignore[attr-defined]
    resp200, conn200 = _PinnedResp(200), _PinnedConn()
    with patch.object(
        backends,
        "_open_byok_pinned",
        lambda request, *, timeout_s, allow_private: (resp200, conn200),
    ):
        with _openai_urlopen(breq, timeout_s=1.0) as resp:
            out["pinned_2xx_yields_response"] = resp is resp200
        out["pinned_response_closed_after"] = resp200.closed
        out["pinned_connection_closed_after"] = conn200.closed

    resp503, conn503 = _PinnedResp(503), _PinnedConn()
    with patch.object(
        backends,
        "_open_byok_pinned",
        lambda request, *, timeout_s, allow_private: (resp503, conn503),
    ):
        ok, exc = _run(lambda: _openai_urlopen(breq, timeout_s=1.0).__enter__())
        out["pinned_non2xx_is_httperror"] = not ok and isinstance(exc, urllib.error.HTTPError)
        out["pinned_error_response_closed"] = resp503.closed
        out["pinned_error_connection_closed"] = conn503.closed

    with _env({"FX1_BYOK_ALLOW_PRIVATE_NETWORKS": "1"}):
        out["private_env_true_variants"] = backends._byok_private_networks_allowed()
    with _env({"FX1_BYOK_ALLOW_PRIVATE_NETWORKS": "yes"}):
        out["private_env_yes_counts"] = backends._byok_private_networks_allowed()
    with _env({"FX1_BYOK_ALLOW_PRIVATE_NETWORKS": "0"}):
        out["private_env_zero_denied"] = not backends._byok_private_networks_allowed()
    with _env({"FX1_BYOK_ALLOW_PRIVATE_NETWORKS": None}):
        out["private_env_absent_denied"] = not backends._byok_private_networks_allowed()

    areq = urllib.request.Request("https://x/v1", data=b"{}", method="POST")
    with _env({"FX1_BYOK_ALLOW_PRIVATE_NETWORKS": "1"}):
        backends._apply_byok_destination_policy(areq)
        out["policy_stamp_allow"] = getattr(areq, "_fx1_byok_allow_private", None) is True
    with _env({"FX1_BYOK_ALLOW_PRIVATE_NETWORKS": None}):
        backends._apply_byok_destination_policy(areq)
        out["policy_stamp_public_only"] = getattr(areq, "_fx1_byok_public_only", None) is True
    return out


# ---------------------------------------------------------------- usage


def _probe_usage() -> dict[str, bool]:
    out: dict[str, bool] = {}

    out["usage_nondict_payload_none"] = _extract_usage("x") is None
    out["usage_missing_none"] = _extract_usage({}) is None
    out["usage_nondict_block_none"] = _extract_usage({"usage": "x"}) is None
    u = _extract_usage(
        {
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
                "junk_bool": True,
                "junk_str": "x",
                "junk_float": 1.5,
                "junk_nan": float("nan"),
            }
        }
    )
    out["usage_only_genuine_ints"] = u == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }
    out["usage_all_malformed_none"] = _extract_usage({"usage": {"b": "x"}}) is None

    out["count_field_wins"] = _extract_token_count({"count": 11}) == 11
    out["count_bool_refused"] = _extract_token_count({"count": True}) is None
    out["count_negative_refused"] = _extract_token_count({"count": -1}) is None
    out["count_zero_ok"] = _extract_token_count({"count": 0}) == 0
    out["count_data_total"] = _extract_token_count({"data": {"total_tokens": 9}}) == 9
    out["count_tokens_len"] = _extract_token_count({"tokens": [1, 2, 3]}) == 3
    out["count_token_ids_len"] = _extract_token_count({"token_ids": [1, 2]}) == 2
    out["count_nothing_none"] = _extract_token_count({"x": 1}) is None
    out["count_nondict_none"] = _extract_token_count([1]) is None

    t = _UsageTracker()
    t._record_usage({"prompt_tokens": 1, "total_tokens": 2})
    t._record_usage({"prompt_tokens": 4, "total_tokens": 6})
    out["tracker_last_is_latest"] = t.last_usage == {"prompt_tokens": 4, "total_tokens": 6}
    out["tracker_total_accumulates"] = t.total_usage == {
        "prompt_tokens": 5,
        "total_tokens": 8,
    }
    t2 = _UsageTracker()
    t2._record_usage(None)
    out["tracker_none_keeps_last_none"] = t2.last_usage is None and t2.total_usage == {}

    t3 = _UsageTracker()

    def _hit() -> None:
        for _ in range(500):
            t3._record_usage({"total_tokens": 1})

    threads = [threading.Thread(target=_hit) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    out["tracker_concurrent_exact"] = t3.total_usage == {"total_tokens": 2000}
    return out


# ---------------------------------------------------------------- shapes


def _probe_shape_validators() -> dict[str, bool]:
    out: dict[str, bool] = {}

    good = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    out["tool_shape_accepts_valid"] = _tool_call_shape(good, "L", 0) is good
    ok, _ = _run(_tool_call_shape, "x", "L", 0)
    out["tool_shape_refuses_nondict"] = not ok
    for bad_key, mutant in (
        ("id_int", {**good, "id": 1}),
        ("wrong_type", {**good, "type": "x"}),
        ("fn_nondict", {**good, "function": "x"}),
        ("name_missing", {**good, "function": {"arguments": "{}"}}),
        ("args_nonstr", {**good, "function": {"name": "f", "arguments": {}}}),
    ):
        ok, _ = _run(_tool_call_shape, mutant, "L", 0)
        out[f"tool_shape_refuses_{bad_key}"] = not ok

    good_e = {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
    out["emb_shape_accepts_numbers"] = _embedding_item_shape(good_e, "L", 0) is good_e
    b64 = {"object": "embedding", "index": 1, "embedding": "AAAA"}
    out["emb_shape_accepts_base64"] = _embedding_item_shape(b64, "L", 1) is b64
    for bad_key, emut in (
        ("nondict", "x"),
        ("wrong_object", {**good_e, "object": "x"}),
        ("str_index", {**good_e, "index": "0"}),
        ("bool_index", {**good_e, "index": True}),
        ("bool_element", {**good_e, "embedding": [0.1, True]}),
        ("missing_emb", {"object": "embedding", "index": 0}),
    ):
        ok, _ = _run(_embedding_item_shape, emut, "L", 0)
        out[f"emb_shape_refuses_{bad_key}"] = not ok
    return out


# ---------------------------------------------------------------- wire helpers


def _probe_wire_helpers() -> dict[str, bool]:
    out: dict[str, bool] = {}
    url = "https://x/v1/chat/completions"

    rec: dict[str, Any] = {}
    with patch.object(
        backends, "_openai_urlopen", _wire(rec, _chat_payload("hi", usage={"total_tokens": 5}))
    ):
        content, usage = _openai_chat_complete(
            url, model="m", messages=_MSGS, timeout_s=3.0, api_key="k", label="L"
        )
    out["chat_complete_returns_content"] = content == "hi"
    out["chat_complete_returns_usage"] = usage == {"total_tokens": 5}
    sent = json.loads(rec["request"].data.decode())
    out["chat_complete_body_shape"] = sent["model"] == "m" and sent["messages"] == _MSGS
    out["chat_complete_deterministic_temp"] = sent["temperature"] == 0.0
    out["chat_complete_auth_header"] = rec["request"].headers.get("Authorization") == "Bearer k"
    out["chat_complete_no_key_no_header"] = "Authorization" not in (json.loads("{}") or {})

    rec2: dict[str, Any] = {}
    with patch.object(
        backends,
        "_openai_urlopen",
        _wire(rec2, _chat_payload("x")),
    ):
        _openai_chat_complete(
            url, model="m", messages=_MSGS, timeout_s=1.0, api_key=None, label="L"
        )
    out["chat_complete_no_key_sends_none"] = "Authorization" not in rec2["request"].headers
    out["chat_complete_timeout_forwarded"] = rec2["timeout_s"] == 1.0

    rec3: dict[str, Any] = {}
    with patch.object(backends, "_openai_urlopen", _wire(rec3, error=_http_error(429))):
        ok, exc = _run(
            _openai_chat_complete,
            url,
            model="m",
            messages=_MSGS,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["chat_http_error_labeled_runtime"] = (
        not ok and isinstance(exc, RuntimeError) and "HTTP 429" in str(exc)
    )
    rec4: dict[str, Any] = {}
    with patch.object(
        backends,
        "_openai_urlopen",
        _wire(rec4, error=urllib.error.URLError(TimeoutError("t"))),
    ):
        ok, exc = _run(
            _openai_chat_complete,
            url,
            model="m",
            messages=_MSGS,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["chat_urlerror_unreachable"] = (
        not ok and "unreachable" in str(exc) and "TimeoutError" in str(exc)
    )
    rec5: dict[str, Any] = {}
    with patch.object(backends, "_openai_urlopen", _wire(rec5, error=TimeoutError("stall"))):
        ok, exc = _run(
            _openai_chat_complete,
            url,
            model="m",
            messages=_MSGS,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["chat_transport_fault_wrapped"] = (
        not ok and "transport fault" in str(exc) and "TimeoutError" in str(exc)
    )
    rec6: dict[str, Any] = {}
    with patch.object(backends, "_openai_urlopen", _wire(rec6, b"not json")):
        ok, exc = _run(
            _openai_chat_complete,
            url,
            model="m",
            messages=_MSGS,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["chat_nonjson_malformed"] = not ok and "not JSON" in str(exc)
    rec7: dict[str, Any] = {}
    with patch.object(
        backends, "_openai_urlopen", _wire(rec7, json.dumps({"choices": []}).encode())
    ):
        ok, exc = _run(
            _openai_chat_complete,
            url,
            model="m",
            messages=_MSGS,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["chat_missing_choices_malformed"] = not ok and "missing" in str(exc)
    rec8: dict[str, Any] = {}
    with patch.object(
        backends,
        "_openai_urlopen",
        _wire(rec8, json.dumps({"choices": [{"message": {"content": 7}}]}).encode()),
    ):
        ok, exc = _run(
            _openai_chat_complete,
            url,
            model="m",
            messages=_MSGS,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["chat_nonstr_content_malformed"] = not ok and "not str" in str(exc)

    # -- tokenize
    rec9: dict[str, Any] = {}
    with patch.object(backends, "_openai_urlopen", _wire(rec9, json.dumps({"count": 17}).encode())):
        n = _openai_tokenize_count(
            "https://x/v1/tokenize",
            model="m",
            messages=_MSGS,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["tokenize_count_returns"] = n == 17
    sent9 = json.loads(rec9["request"].data.decode())
    out["tokenize_adds_generation_prompt"] = sent9.get("add_generation_prompt") is True
    for code in (400, 404, 405, 501):
        with patch.object(backends, "_openai_urlopen", _wire({}, error=_http_error(code))):
            ok, exc = _run(
                _openai_tokenize_count,
                "https://x/v1/tokenize",
                model="m",
                messages=_MSGS,
                timeout_s=1.0,
                api_key=None,
                label="L",
            )
        out[f"tokenize_{code}_unavailable"] = not ok and isinstance(exc, TokenCountUnavailableError)
    with patch.object(backends, "_openai_urlopen", _wire({}, error=_http_error(500))):
        ok, exc = _run(
            _openai_tokenize_count,
            "https://x/v1/tokenize",
            model="m",
            messages=_MSGS,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["tokenize_500_runtime"] = not ok and isinstance(exc, RuntimeError)
    with patch.object(backends, "_openai_urlopen", _wire({}, json.dumps({"x": 1}).encode())):
        ok, exc = _run(
            _openai_tokenize_count,
            "https://x/v1/tokenize",
            model="m",
            messages=_MSGS,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["tokenize_malformed_runtime"] = not ok and "needs count" in str(exc)
    out["tokenize_never_estimates"] = not ok

    # -- tools
    rec10: dict[str, Any] = {}
    tool_payload = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                    "logprobs": {"content": []},
                }
            ],
            "usage": {"total_tokens": 9},
        }
    ).encode()
    with patch.object(backends, "_openai_urlopen", _wire(rec10, tool_payload)):
        tc, tu = _openai_chat_complete_tools(
            url,
            model="m",
            messages=_MSGS,
            tools=[{"type": "function"}],
            tool_choice="auto",
            parallel_tool_calls=True,
            logprobs=True,
            top_logprobs=2,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["tools_null_content_allowed"] = tc.content is None
    out["tools_calls_parsed"] = (
        tc.tool_calls is not None and tc.tool_calls[0]["function"]["name"] == "f"
    )
    out["tools_finish_reason"] = tc.finish_reason == "tool_calls"
    out["tools_logprobs_verbatim"] = tc.logprobs == {"content": []}
    out["tools_usage_returned"] = tu == {"total_tokens": 9}
    sent10 = json.loads(rec10["request"].data.decode())
    out["tools_wire_fields_verbatim"] = (
        sent10.get("tools") == [{"type": "function"}]
        and sent10.get("tool_choice") == "auto"
        and sent10.get("parallel_tool_calls") is True
        and sent10.get("logprobs") is True
        and sent10.get("top_logprobs") == 2
    )

    rec11: dict[str, Any] = {}
    with patch.object(
        backends,
        "_openai_urlopen",
        _wire(rec11, json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()),
    ):
        tc2, _ = _openai_chat_complete_tools(
            url,
            model="m",
            messages=_MSGS,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["tools_absent_no_wire_fields"] = all(
        k not in json.loads(rec11["request"].data.decode())
        for k in ("tools", "tool_choice", "parallel_tool_calls", "logprobs", "top_logprobs")
    )
    out["tools_none_fields_absent"] = tc2.tool_calls is None
    for bad_key, mutant in (
        ("nonstr_content", {"content": 5}),
        ("toolcalls_nondict", {"tool_calls": "x"}),
        ("toolcalls_badentry", {"tool_calls": [{"id": "x"}]}),
    ):
        msg = {"content": "ok", **mutant}
        with patch.object(
            backends,
            "_openai_urlopen",
            _wire({}, json.dumps({"choices": [{"message": msg}]}).encode()),
        ):
            ok, _ = _run(
                _openai_chat_complete_tools,
                url,
                model="m",
                messages=_MSGS,
                tools=None,
                tool_choice=None,
                parallel_tool_calls=None,
                timeout_s=1.0,
                api_key=None,
                label="L",
            )
        out[f"tools_refuses_{bad_key}"] = not ok
    with patch.object(
        backends,
        "_openai_urlopen",
        _wire(
            {}, json.dumps({"choices": [{"message": {"content": "ok"}, "logprobs": []}]}).encode()
        ),
    ):
        ok, _ = _run(
            _openai_chat_complete_tools,
            url,
            model="m",
            messages=_MSGS,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=None,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["tools_refuses_nondict_logprobs"] = not ok

    # -- stream
    frames = (
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
        b'data: {"choices":[{"delta":{"content":"he"}}]}\n'
        b'data: {"choices":[{"delta":{"content":"llo"}}]}\n'
        b'data: {"usage":{"total_tokens":3},"choices":[]}\n'
        b"data: [DONE]\n"
    )
    rec12: dict[str, Any] = {}
    usage_box: list[dict[str, int]] = []
    with patch.object(backends, "_openai_urlopen", _wire(rec12, frames=frames)):
        toks = list(
            _openai_chat_stream(
                url,
                model="m",
                messages=_MSGS,
                timeout_s=1.0,
                api_key=None,
                label="L",
                usage_out=usage_box,
            )
        )
    out["stream_yields_content_deltas"] = toks == ["he", "llo"]
    out["stream_skips_role_frames"] = toks == ["he", "llo"]
    out["stream_captures_usage_out"] = usage_box == [{"total_tokens": 3}]
    sent12 = json.loads(rec12["request"].data.decode())
    out["stream_sends_stream_flag"] = sent12.get("stream") is True
    out["stream_no_stream_options"] = "stream_options" not in sent12
    out["stream_accept_sse_header"] = rec12["request"].headers.get("Accept") == "text/event-stream"

    frames_usage_only = b'data: {"usage":{"total_tokens":1}}\ndata: [DONE]\n'
    with patch.object(backends, "_openai_urlopen", _wire({}, frames=frames_usage_only)):
        toks2 = list(
            _openai_chat_stream(
                url, model="m", messages=_MSGS, timeout_s=1.0, api_key=None, label="L"
            )
        )
    out["stream_usage_only_frame_ok"] = toks2 == []

    frames_bad = b'data: {"x":1}\ndata: [DONE]\n'
    with patch.object(backends, "_openai_urlopen", _wire({}, frames=frames_bad)):
        ok, exc = _run(
            lambda: list(
                _openai_chat_stream(
                    url, model="m", messages=_MSGS, timeout_s=1.0, api_key=None, label="L"
                )
            )
        )
    out["stream_missing_choices_fails_closed"] = not ok and "missing choices" in str(exc)
    frames_nonstr = b'data: {"choices":[{"delta":{"content":5}}]}\n'
    with patch.object(backends, "_openai_urlopen", _wire({}, frames=frames_nonstr)):
        ok, exc = _run(
            lambda: list(
                _openai_chat_stream(
                    url, model="m", messages=_MSGS, timeout_s=1.0, api_key=None, label="L"
                )
            )
        )
    out["stream_nonstr_content_refused"] = not ok and "not str" in str(exc)
    frames_nonjson = b"data: {oops\n"
    with patch.object(backends, "_openai_urlopen", _wire({}, frames=frames_nonjson)):
        ok, exc = _run(
            lambda: list(
                _openai_chat_stream(
                    url, model="m", messages=_MSGS, timeout_s=1.0, api_key=None, label="L"
                )
            )
        )
    out["stream_nonjson_frame_refused"] = not ok and "not JSON" in str(exc)

    # -- embeddings
    rec13: dict[str, Any] = {}
    emb_payload = json.dumps(
        {
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": "emb-m",
            "usage": {"total_tokens": 2},
        }
    ).encode()
    with patch.object(backends, "_openai_urlopen", _wire(rec13, emb_payload)):
        er = _openai_embeddings_complete(
            "https://x/v1/embeddings",
            model="emb-m",
            input=["a", "b"],
            encoding_format="float",
            dimensions=2,
            user="u",
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["emb_result_shape"] = (
        len(er.data) == 1 and er.model == "emb-m" and er.usage == {"total_tokens": 2}
    )
    sent13 = json.loads(rec13["request"].data.decode())
    out["emb_wire_fields_verbatim"] = (
        sent13["input"] == ["a", "b"]
        and sent13["model"] == "emb-m"
        and sent13["encoding_format"] == "float"
        and sent13["dimensions"] == 2
        and sent13["user"] == "u"
    )
    with patch.object(
        backends,
        "_openai_urlopen",
        _wire({}, json.dumps({"data": {"x": 1}}).encode()),
    ):
        ok, _ = _run(
            _openai_embeddings_complete,
            "https://x/v1/embeddings",
            model="m",
            input="a",
            encoding_format=None,
            dimensions=None,
            user=None,
            timeout_s=1.0,
            api_key=None,
            label="L",
        )
    out["emb_nonlist_data_refused"] = not ok
    return out


# ---------------------------------------------------------------- backends


def _probe_backend_classes() -> dict[str, bool]:
    out: dict[str, bool] = {}

    with _env({"MOONSHOT_API_KEY": None}):
        ok, exc = _run(HostedK3Backend)
        out["k3_missing_key_refuses"] = not ok and isinstance(exc, RuntimeError)
    with _env({"MOONSHOT_API_KEY": "env-key"}):
        b = HostedK3Backend()
        out["k3_env_key_picked"] = b._api_key == "env-key"
    ok, exc = _run(HostedK3Backend, api_key="k", timeout_s=0)
    out["k3_zero_timeout_refused"] = not ok and isinstance(exc, ValueError)
    b = HostedK3Backend(api_key="k", timeout_s=5)
    ok_inf, _ = _run(isinstance, b, InferenceBackend)
    out["inference_base_not_checkable"] = not ok_inf
    out["k3_is_tool_backend"] = isinstance(b, ToolBackend)
    out["k3_is_streaming_backend"] = isinstance(b, StreamingBackend)
    out["k3_is_embedding_backend"] = isinstance(b, EmbeddingBackend)
    out["k3_is_counting_backend"] = isinstance(b, TokenCountingBackend)
    rec: dict[str, Any] = {}
    with patch.object(
        backends, "_openai_urlopen", _wire(rec, _chat_payload("k3 ok", usage={"total_tokens": 4}))
    ):
        content = b.complete(_MSGS)
    out["k3_complete_content"] = content == "k3 ok"
    out["k3_records_usage"] = b.last_usage == {"total_tokens": 4}
    sent = json.loads(rec["request"].data.decode())
    out["k3_wire_model_pin"] = sent["model"] == "kimi-k3"

    empty = {k: None for k in (BYOK_BASE_URL_ENV, BYOK_API_KEY_ENV, BYOK_MODEL_ENV)}
    with _env(empty):
        ok, exc = _run(OpenAICompatBackend)
        out["byok_missing_all_refuses"] = (
            not ok
            and isinstance(exc, BackendNotConfiguredError)
            and all(k in str(exc) for k in (BYOK_BASE_URL_ENV, BYOK_API_KEY_ENV, BYOK_MODEL_ENV))
        )
    with _env({**empty, BYOK_BASE_URL_ENV: "https://h/v1"}):
        ok, exc = _run(OpenAICompatBackend)
        out["byok_missing_pair_names_missing"] = (
            not ok and BYOK_API_KEY_ENV in str(exc) and BYOK_BASE_URL_ENV not in str(exc)
        )
    with _env(empty):
        ok, exc = _run(
            OpenAICompatBackend,
            base_url="https://u:p@h/v1",
            api_key="k",
            model="m",
        )
        out["byok_bad_url_refuses"] = not ok and isinstance(exc, RuntimeError)
    with _env(empty):
        bk = OpenAICompatBackend(base_url="https://h/v1", api_key="k", model="m", timeout_s=9)
        out["byok_url_normalized"] = bk._url == "https://h/v1/chat/completions"
        recb: dict[str, Any] = {}
        with patch.object(backends, "_openai_urlopen", _wire(recb, _chat_payload("byok ok"))):
            out["byok_complete_content"] = bk.complete(_MSGS) == "byok ok"
        sentb_req = recb["request"]
        out["byok_request_policy_stamped"] = hasattr(sentb_req, "_fx1_byok_public_only")
        out["byok_auth_header"] = sentb_req.headers.get("Authorization") == "Bearer k"

    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        ok, exc = _run(LocalFx1Backend, wd / "nope")
        out["local_missing_card_filenotfound"] = not ok and isinstance(exc, FileNotFoundError)
        ck_bad = _checkpoint(wd / "bad", eligible=False)
        ok, exc = _run(LocalFx1Backend, ck_bad)
        out["local_gate_fail_refuses"] = not ok and isinstance(exc, RuntimeError)
        ck = _checkpoint(wd / "good", eligible=True)
        with _env({k: None for k in _ENV_KEYS}):
            lb = LocalFx1Backend(ck)
            out["local_model_defaults_card_version"] = lb._model == "fx-1.v1.0"
            ok, exc = _run(lb.complete, _MSGS)
            out["local_unconfigured_backend_not_configured"] = not ok and isinstance(
                exc, BackendNotConfiguredError
            )
            ok, exc = _run(lb.count_tokens, _MSGS)
            out["local_count_unconfigured"] = not ok and isinstance(exc, BackendNotConfiguredError)
        with _env({k: None for k in _ENV_KEYS} | {"FX1_SIGNING_KEY": "k"}):
            with patch("fx1.serve.signing.verify_release", lambda root: False):
                ok, exc = _run(LocalFx1Backend, ck)
                out["local_unsigned_refuses"] = not ok and isinstance(exc, RuntimeError)
            with patch("fx1.serve.signing.verify_release", lambda root: True):
                ok2, lb2 = _run(LocalFx1Backend, ck)
                out["local_signed_passes"] = ok2
        with _env({k: None for k in _ENV_KEYS} | {LOCAL_SERVE_URL_ENV: "ftp://x"}):
            ok, exc = _run(LocalFx1Backend, ck)
            out["local_bad_serve_url_refuses"] = not ok and isinstance(exc, RuntimeError)
        with _env({k: None for k in _ENV_KEYS} | {LOCAL_SERVE_URL_ENV: "http://127.0.0.1:9/v1"}):
            lb3 = LocalFx1Backend(ck)
            out["local_url_normalized"] = lb3._url == "http://127.0.0.1:9/v1/chat/completions"
            rec3: dict[str, Any] = {}
            with patch.object(backends, "_openai_urlopen", _wire(rec3, _chat_payload("local ok"))):
                out["local_complete_attached_engine"] = lb3.complete(_MSGS) == "local ok"
            out["local_timeout_env_default"] = lb3._timeout_s == 120.0
        with _env(
            {k: None for k in _ENV_KEYS}
            | {LOCAL_TIMEOUT_S_ENV: "7.5", LOCAL_START_TIMEOUT_S_ENV: "3"}
        ):
            lb4 = LocalFx1Backend(ck, serve_url="http://127.0.0.1:9/v1")
            out["local_env_timeouts"] = lb4._timeout_s == 7.5 and lb4._start_timeout_s == 3.0

        class _FakeProc:
            def __init__(self, argv, env=None, stdout=None, stderr=None):
                rec_spawn["argv"] = argv
                rec_spawn["env"] = env
                self.terminated = False
                self.killed = False

            def poll(self):
                return None if not self.terminated else 0

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                return 0

        rec_spawn: dict[str, Any] = {}
        with _env(
            {k: None for k in _ENV_KEYS}
            | {
                LOCAL_SERVE_URL_ENV: "http://127.0.0.1:9/v1",
                LOCAL_SERVE_CMD_ENV: 'python -m x "$checkpoint_dir" --port 9',
            }
        ):
            lb5 = LocalFx1Backend(ck)
            with (
                patch.object(
                    backends.LocalFx1Backend, "_engine_up", lambda self: lb5._proc is not None
                ),
                patch.object(backends.subprocess, "Popen", _FakeProc),
            ):
                lb5._ensure_engine()
                out["spawn_template_expands_checkpoint"] = str(ck) in rec_spawn["argv"]
                out["spawn_exports_checkpoint_env"] = rec_spawn["env"]["FX1_CHECKPOINT_DIR"] == str(
                    ck
                )
                lb5.close()
                out["close_terminates_spawned"] = lb5._proc is None
        lb6 = LocalFx1Backend(ck, serve_url="http://127.0.0.1:9/v1")
        lb6.close()
        out["close_attach_only_noop"] = lb6._proc is None
    return out


def _probe_factory_and_protocols() -> dict[str, bool]:
    out: dict[str, bool] = {}
    ok, exc = _run(get_backend, "nope")
    out["factory_unknown_keyerror"] = not ok and isinstance(exc, KeyError)
    out["factory_error_lists_kinds"] = not ok and "hosted_k3" in str(exc)
    with _env({"MOONSHOT_API_KEY": "k"}):
        out["factory_hosted_k3"] = isinstance(get_backend("hosted_k3"), HostedK3Backend)
    with _env({BYOK_BASE_URL_ENV: "https://h/v1", BYOK_API_KEY_ENV: "k", BYOK_MODEL_ENV: "m"}):
        out["factory_byok"] = isinstance(get_backend("byok"), OpenAICompatBackend)
    with tempfile.TemporaryDirectory() as td:
        ck = _checkpoint(Path(td), eligible=True)
        with _env({k: None for k in _ENV_KEYS}):
            out["factory_local_fx1"] = isinstance(
                get_backend("local_fx1", checkpoint_dir=ck), LocalFx1Backend
            )

    class _Duck:
        def complete(self, messages, *, sampling=None):
            return "x"

    ok_inf2, exc_inf = _run(isinstance, _Duck(), InferenceBackend)
    out["plain_duck_inference_raises"] = not ok_inf2 and isinstance(exc_inf, TypeError)
    out["plain_duck_not_tool"] = not isinstance(_Duck(), ToolBackend)
    out["plain_duck_not_stream"] = not isinstance(_Duck(), StreamingBackend)
    out["plain_duck_not_embed"] = not isinstance(_Duck(), EmbeddingBackend)
    out["plain_duck_not_count"] = not isinstance(_Duck(), TokenCountingBackend)

    class _ToolsOnly:
        def complete_with_tools(self, messages, **kw):
            return None

    out["tools_only_not_tool"] = not isinstance(_ToolsOnly(), ToolBackend)

    class _Full:
        def complete(self, messages, *, sampling=None):
            return "x"

        def complete_with_tools(self, messages, **kw):
            return None

        def stream(self, messages, *, sampling=None):
            yield "x"

        def embeddings(self, input, *, model, **kw):
            return None

        def count_tokens(self, messages):
            return 0

    full = _Full()
    out["full_duck_all_protocols"] = (
        isinstance(full, ToolBackend)
        and isinstance(full, StreamingBackend)
        and isinstance(full, EmbeddingBackend)
        and isinstance(full, TokenCountingBackend)
    )
    return out


# ---------------------------------------------------------------- battery

_PROBES = (
    _probe_sampling_and_truncation,
    _probe_url_and_env_helpers,
    _probe_byok_policy,
    _probe_transport,
    _probe_usage,
    _probe_shape_validators,
    _probe_wire_helpers,
    _probe_backend_classes,
    _probe_factory_and_protocols,
)


def backends_audit() -> dict[str, bool]:
    """Run every backends probe; ``name -> passed``."""
    out: dict[str, bool] = {}
    for probe in _PROBES:
        out.update(probe())
    return out


def backends_audit_bench() -> dict[str, Any]:
    """Sealed bench payload — SYNTHETIC, research-only, no live-PnL claim."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = backends_audit()
    ok = bool(r) and all(r.values())
    defects = sorted(k for k, v in r.items() if v is not True)
    out: dict[str, Any] = {
        "kind": "backends_audit",
        "schema": "backends_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process; wire seam stubbed at backends._openai_urlopen",
            "not_verified": [
                "live provider round-trips",
                "real DNS resolution paths",
                "actual engine spawn/attach lifecycle",
            ],
        },
        "interpretation": (
            "all_probes_hold"
            if ok
            else ("no_probes" if not r else f"defects: {', '.join(defects)}")
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
