"""Surface-parity audit — the SDK, the HTTP API, and HarnessClient are one contract.

``fx1.sdk.Fx1Harness`` (in-process), ``fx1.serve.api`` (the wire), and
``fx1.serve.client.HarnessClient`` (the remote caller) expose the same
harness over three transports. This audit drives the SAME injected backend
through all three and asserts byte-identical results — a drift between
what an in-process caller, the server, and a remote client get is a
product defect, not a transport detail.

- *Complete parity* — ``Fx1Harness.complete`` output equals
  ``POST /harness/complete``: identical content (evidence footer included),
  identical ``model``/``backend``/``receipt_hashes`` envelope fields.
- *Batch parity* — ``complete_many`` results equal
  ``POST /harness/complete/batch`` items in submission order, one shared
  backend per batch on both surfaces (close counts prove it).
- *Stream parity* — SSE ``token`` frames concatenate to exactly the
  ``complete`` content; ``stream_complete`` returns the identical chunk
  list; the ``final`` envelope matches the completion envelope.
- *Error parity* — the same fault surfaces with the same class on both:
  gate refusal (SDK raises ``Fx1HonestyError`` / API 502 JSON with the
  refusal text), non-streaming backend (NotImplementedError / 501),
  checkpoint_dir on a non-local backend (ValueError / 422), unknown
  backend (KeyError / 422 literal rejection), malformed receipts.
- *Verifier parity* — ``verify_receipt`` and ``POST /receipts/verify``
  agree on valid/errors/warnings/schema for a sealed receipt, a tampered
  digest, and a non-receipt dict.
- *Registry/health parity* — command list and backend presence booleans
  are identical over both surfaces.
- *OpenAI ingress parity* — ``/v1/chat/completions`` and
  ``Fx1Harness.openai_chat`` translate through the same module
  (``fx1.serve.openai_compat``): identical envelopes, identical chunk
  sequences on the streaming surface, identical rejections under each
  surface's own exception class, identical model inventory and
  completion-log linkage.
- *Remote-client parity* — ``HarnessClient`` (urllib transport injectable)
  returns the SDK's own result types; wire status codes map back to the
  SDK's exception classes (KeyError/ValueError/BackendNotConfiguredError/
  NotImplementedError/Fx1HonestyError), auth faults are
  ``HarnessAuthError``, transport faults ``HarnessTransportError``.
- *In-flight cap* — ``max_inflight`` (env ``FX1_API_MAX_INFLIGHT``, default
  16) bounds concurrent heavy requests: saturation is an honest 503 with
  ``Retry-After``, cheap routes stay responsive, the slot releases
  cleanly, ``max_inflight<1`` fails app construction.

Sealed ``parity_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import dataclasses
import json
import os
import urllib.parse
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from fx1.serve.backends import EmbeddingResult, SamplingParams, ToolCompletion

__all__ = ["parity_audit", "parity_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_BYOK_ENVS = ("FX1_BYOK_BASE_URL", "FX1_BYOK_API_KEY", "FX1_BYOK_MODEL")
_LOCAL_ENVS = (
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_CHECKPOINT_DIR",
)


class _ParityBackend:
    """Deterministic streaming backend — one instance per resolution."""

    _closed_total = 0

    def __init__(self) -> None:
        self._model = "parity-v0"

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        return f"echo:{messages[-1]['content']}"

    def stream(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> Iterator[str]:
        text = self.complete(messages)
        yield text[:3]
        yield text[3:]

    def close(self) -> None:
        _ParityBackend._closed_total += 1


class _DirtyBackend(_ParityBackend):
    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        return "total Sharpe 4.2 on NAV"  # forbidden headline

    def stream(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> Iterator[str]:
        yield "total Sharpe 4.2 on NAV"


class _NonStreamingBackend:
    def __init__(self) -> None:
        self._model = "ns-v0"

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        return "ans"

    def close(self) -> None:
        return None


class _ParityToolBackend(_ParityBackend):
    """Tool-capable parity backend — answers one canned function call,
    plus a canned logprobs payload when the request asks for scores."""

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion:
        from fx1.serve.backends import ToolCompletion  # noqa: PLC0415

        lp: dict[str, Any] | None = None
        if logprobs:
            lp = {
                "content": [
                    {
                        "token": "x",
                        "logprob": -0.5,
                        "bytes": [120],
                        "top_logprobs": (
                            [{"token": "x", "logprob": -0.5, "bytes": [120]}]
                            if top_logprobs is not None
                            else []
                        ),
                    }
                ]
            }
        return ToolCompletion(
            content=None,
            tool_calls=(
                {
                    "id": "call_p",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"x": 1}'},
                },
            ),
            finish_reason="tool_calls",
            logprobs=lp,
        )


class _ParityLpBackend(_ParityBackend):
    """Structured-channel stub answering a text turn with scores — no
    tool_calls, so the message part exists to carry the logprobs."""

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: SamplingParams | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        parallel_tool_calls: bool | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
    ) -> ToolCompletion:
        from fx1.serve.backends import ToolCompletion  # noqa: PLC0415

        lp: dict[str, Any] | None = None
        if logprobs:
            lp = {
                "content": [
                    {
                        "token": "x",
                        "logprob": -0.5,
                        "bytes": [120],
                        "top_logprobs": (
                            [{"token": "x", "logprob": -0.5, "bytes": [120]}]
                            if top_logprobs is not None
                            else []
                        ),
                    }
                ]
            }
        return ToolCompletion(
            content=self.complete(messages),
            tool_calls=None,
            finish_reason="stop",
            logprobs=lp,
        )


class _ParityEmbedBackend(_ParityBackend):
    """Embedding-capable parity stub — deterministic canned vectors."""

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the wire field's own name
        *,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResult:
        from fx1.serve.backends import EmbeddingResult  # noqa: PLC0415

        n = (
            len(input)
            if isinstance(input, list) and input and isinstance(input[0], (str, list))
            else 1
        )
        return EmbeddingResult(
            data=tuple(
                {"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i in range(n)
            ),
            model=f"{model}-v1",
            usage={"prompt_tokens": 4, "total_tokens": 4},
        )


class _CallFailBackend(_ParityBackend):
    """Availability fault on every call — the retriable link."""

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        raise RuntimeError("backend exploded")

    def stream(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> Iterator[str]:
        raise RuntimeError("backend exploded")
        yield


class _FlakyBackend(_ParityBackend):
    """Refuses (gate-tripping output) only on prompts containing 'bad'."""

    def complete(
        self, messages: list[dict[str, str]], *, sampling: SamplingParams | None = None
    ) -> str:
        if "bad" in messages[-1]["content"]:
            return "total Sharpe 4.2 on NAV"
        return super().complete(messages)


def _raises(fn: Any) -> tuple[str, str]:
    """(exception class name, str(exc)) — ("", "") when no raise."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — probe captures the class
        return type(exc).__name__, str(exc)
    return "", ""


def _surfaces(
    backend: Any,
    *,
    ft_runner: Any = None,
) -> tuple[Any, TestClient]:
    """(Fx1Harness, TestClient) wired to one resolver returning `backend`'s class."""
    from fastapi.testclient import TestClient as _TC

    import fx1.serve.api as api_mod
    from fx1.harness import Harness
    from fx1.sdk import Fx1Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, f"ran:{' '.join(argv)}", ""

    def resolver(name: str, **kw: Any) -> Any:
        if name not in {"hosted_k3", "local_fx1", "byok"}:
            raise KeyError(name)
        return backend()

    sdk = Fx1Harness(
        harness=Harness(runner=fake_runner), backend_resolver=resolver, ft_runner=ft_runner
    )
    app = api_mod.create_app(
        harness=Harness(runner=fake_runner),
        backend_resolver=resolver,
        ft_runner=ft_runner,
        # no receipt store -> receipt_hashes citations stay advisory, so these
        # probes can use synthetic hashes deterministically from any cwd.
        receipts_dir="/nonexistent-parity-store",
    )
    return sdk, _TC(app)


def _tc_transport(client: TestClient) -> Any:
    """Adapt HarnessClient's transport contract to a TestClient."""

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        p = urllib.parse.urlparse(url)
        path = p.path + (f"?{p.query}" if p.query else "")
        if method == "GET":
            resp = client.get(path, headers=headers)
        elif method == "DELETE":
            resp = client.delete(path, headers=headers)
        elif isinstance(payload, bytes):
            resp = client.post(path, content=payload, headers=headers)
        else:
            resp = client.post(path, json=payload, headers=headers)
        return resp.status_code, dict(resp.headers), resp.content

    return send


def parity_audit() -> dict[str, bool]:
    out: dict[str, bool] = {}
    saved = {
        k: os.environ.get(k) for k in (_API_KEY_ENV, *_BYOK_ENVS, *_LOCAL_ENVS, "MOONSHOT_API_KEY")
    }
    try:
        os.environ.pop(_API_KEY_ENV, None)

        sdk, client = _surfaces(_ParityBackend)
        msg = [{"role": "user", "content": "ping"}]
        receipt = "a" * 64
        sdk_out = sdk.complete(msg, backend="byok", receipt_hashes=[receipt])
        api_out = client.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": msg,
                "receipt_hashes": [receipt],
            },
        )
        out["complete_status_200"] = api_out.status_code == 200
        api_json = api_out.json()
        out["complete_content_identical"] = api_json["content"] == sdk_out.content
        out["complete_envelope_identical"] = (
            api_json["backend"] == sdk_out.backend
            and api_json["model"] == sdk_out.model
            and tuple(api_json["receipt_hashes"]) == sdk_out.receipt_hashes
        )
        out["complete_footer_identical"] = (
            sdk_out.content.endswith(
                f"Evidence: `{receipt[:16]}…` — verify with `dipcatcher verify-research`."
            )
            and api_json["content"] == sdk_out.content
        )

        # --- batch parity -----------------------------------------------------
        _ParityBackend._closed_total = 0
        batch = [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ]
        sdk_many = sdk.complete_many(batch, backend="byok", receipt_hashes=[receipt], max_workers=2)
        api_many = client.post(
            "/harness/complete/batch",
            json={
                "backend": "byok",
                "batch": batch,
                "receipt_hashes": [receipt],
                "max_workers": 2,
            },
        )
        out["batch_status_200"] = api_many.status_code == 200
        api_items = api_many.json()
        out["batch_content_identical"] = [r.content for r in sdk_many] == [
            i["content"] for i in api_items["results"]
        ] and all(i["ok"] for i in api_items["results"])
        out["batch_envelope_identical"] = (
            api_items["backend"] == sdk_many[0].backend
            and api_items["model"] == sdk_many[0].model
            and tuple(api_items["receipt_hashes"]) == sdk_many[0].receipt_hashes
        )
        out["batch_one_backend_each_side"] = _ParityBackend._closed_total == 2

        # --- stream parity ----------------------------------------------------
        sdk_chunks = sdk.stream_complete(msg, backend="byok", receipt_hashes=[receipt])
        api_stream = client.post(
            "/harness/complete/stream",
            json={
                "backend": "byok",
                "messages": msg,
                "receipt_hashes": [receipt],
            },
        )
        out["stream_status_200_sse"] = api_stream.status_code == 200 and api_stream.headers[
            "content-type"
        ].startswith("text/event-stream")
        frames = [
            ln[len("data: ") :] for ln in api_stream.text.splitlines() if ln.startswith("data: ")
        ]
        payloads = [json.loads(f) for f in frames if f.strip() != "[DONE]"]
        token_chunks = [p["content"] for p in payloads if p["type"] == "token"]
        final = [p for p in payloads if p["type"] == "final"]
        out["stream_chunks_identical"] = token_chunks == sdk_chunks
        out["stream_joined_eq_complete"] = (
            "".join(token_chunks) == sdk_out.content and "".join(sdk_chunks) == sdk_out.content
        )
        out["stream_done_terminal"] = frames[-1].strip() == "[DONE]"
        out["stream_final_envelope"] = (
            len(final) == 1
            and final[0]["model"] == sdk_out.model
            and final[0]["receipt_hashes"] == [receipt]
        )

        # --- error parity -----------------------------------------------------
        dirty_sdk, dirty_api = _surfaces(_DirtyBackend)
        sdk_err_cls, sdk_err_txt = _raises(lambda: dirty_sdk.complete(msg, backend="byok"))
        api_err = dirty_api.post("/harness/complete", json={"backend": "byok", "messages": msg})
        out["gate_sdk_raises_api_502"] = (
            sdk_err_cls == "Fx1HonestyError"
            and api_err.status_code == 502
            and "honesty gate" in api_err.json()["detail"]
        )
        # the API surfaces the SDK's exception text verbatim after the prefix
        out["gate_error_text_shared"] = (
            api_err.status_code == 502
            and api_err.json()["detail"] == f"honesty gate refused model output: {sdk_err_txt}"
        )
        sdk_serr, _ = _raises(lambda: dirty_sdk.stream_complete(msg, backend="byok"))
        api_serr = dirty_api.post(
            "/harness/complete/stream", json={"backend": "byok", "messages": msg}
        )
        out["stream_gate_parity"] = (
            sdk_serr == "Fx1HonestyError"
            and api_serr.status_code == 502
            and "total Sharpe" not in api_serr.text
        )

        ns_sdk, ns_api = _surfaces(_NonStreamingBackend)
        out["stream_unsupported_parity"] = (
            _raises(lambda: ns_sdk.stream_complete(msg, backend="byok"))[0] == "NotImplementedError"
            and ns_api.post(
                "/harness/complete/stream",
                json={"backend": "byok", "messages": msg},
            ).status_code
            == 501
        )
        out["checkpoint_dir_non_local_parity"] = (
            _raises(lambda: sdk.complete(msg, backend="byok", checkpoint_dir="/x"))[0]
            == "ValueError"
            and client.post(
                "/harness/complete",
                json={
                    "backend": "byok",
                    "messages": msg,
                    "checkpoint_dir": "/x",
                },
            ).status_code
            == 422
        )
        # Resolver KeyError is unreachable over the wire: CompleteRequest's
        # backend Literal rejects unknown names with 422 before resolution;
        # the SDK raises KeyError on the same name. Both fail closed.
        out["unknown_backend_fail_closed"] = (
            _raises(lambda: sdk.complete(msg, backend="bogus"))[0] == "KeyError"
            and client.post(
                "/harness/complete",
                json={"backend": "bogus", "messages": msg},
            ).status_code
            == 422
        )
        out["empty_batch_contract"] = (
            sdk.complete_many([], backend="byok") == []
            and client.post(
                "/harness/complete/batch",
                json={"backend": "byok", "batch": []},
            ).status_code
            == 422
        )

        # --- OpenAI ingress parity ----------------------------------------------
        # One translation module (fx1.serve.openai_compat) serves both surfaces:
        # the same request body over the wire and in-process must produce the
        # same envelope, the same chunk stream, and the same rejections.
        oai_body = {
            "model": "hosted_k3",
            "messages": msg,
            "fx1": {"backend": "byok"},
        }
        oai_wire = client.post("/v1/chat/completions", json=oai_body)
        oai_sdk, oai_cid = sdk.openai_chat(oai_body)
        oai_json = oai_wire.json()
        out["openai_envelope_parity"] = (
            oai_wire.status_code == 200
            and oai_json["object"] == "chat.completion" == oai_sdk.object
            and oai_json["choices"][0]["message"]["content"]
            == oai_sdk.choices[0].message["content"]
            and oai_json["model"] == oai_sdk.model
            and oai_json["system_fingerprint"] == oai_sdk.system_fingerprint
            and oai_json["usage"] == oai_sdk.usage
        )
        oai_wire_cid = oai_wire.headers.get("X-Fx1-Completion-Id")
        out["openai_completion_log_parity"] = bool(
            oai_cid
            and oai_wire_cid
            and sdk.completion(oai_cid).backend == "byok"
            and client.get(f"/harness/completions/{oai_wire_cid}").status_code == 200
        )
        # stream parity: identical chunk-payload sequences (sans id/created)
        oai_stream_req = {**oai_body, "stream": True, "stream_options": {"include_usage": True}}
        oai_stream_wire = client.post("/v1/chat/completions", json=oai_stream_req)
        wire_frames = [
            json.loads(ln[len("data: ") :])
            for ln in oai_stream_wire.text.splitlines()
            if ln.startswith("data: ") and ln[len("data: ") :].strip() != "[DONE]"
        ]
        sdk_frames, _ = sdk.openai_chat_stream(
            {**oai_body, "stream_options": {"include_usage": True}}
        )

        def _strip_meta(payload: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in payload.items() if k not in ("id", "created")}

        out["openai_stream_parity"] = [_strip_meta(f) for f in wire_frames] == [
            _strip_meta(f) for f in sdk_frames
        ]
        out["openai_stream_done_terminal"] = oai_stream_wire.text.rstrip().endswith("data: [DONE]")
        out["openai_stream_usage_chunk"] = (
            bool(wire_frames)
            and wire_frames[-1]["choices"] == []
            and sdk_frames[-1]["choices"] == []
        )
        # models parity
        wire_models = client.get("/v1/models")
        sdk_models = sdk.openai_models()
        out["openai_models_parity"] = (
            wire_models.status_code == 200
            and wire_models.json()["object"] == "list" == sdk_models.object
            and [m["id"] for m in wire_models.json()["data"]] == [m.id for m in sdk_models.data]
        )
        # models.retrieve parity — one card per surface; unknown id is a
        # 404 on the wire and an OpenAICompatError in the SDK
        wire_rm = client.get("/v1/models/fx1")
        sdk_rm = sdk.openai_model("fx1")
        out["openai_retrieve_parity"] = (
            wire_rm.status_code == 200
            and wire_rm.json()["id"] == sdk_rm.id == "fx1"
            and wire_rm.json()["object"] == sdk_rm.object == "model"
            and isinstance(wire_rm.json()["created"], int)
            and isinstance(sdk_rm.created, int)
        )
        wire_rm_404 = client.get("/v1/models/nope")
        sdk_rm_err = _raises(lambda: sdk.openai_model("nope"))[0]
        out["openai_retrieve_404_parity"] = (
            wire_rm_404.status_code == 404
            and wire_rm_404.json()["error"]["code"] == "model_not_found"
            and sdk_rm_err == "OpenAICompatError"
        )

        # response_format parity — the post-validation gate runs inside
        # the shared compat layer, so wire and SDK hand back the same
        # verdicts: schema-conforming output ships on both; a violation
        # is 502 format_violation on the wire and OpenAICompatError
        # in-process.
        class _JsonB:
            def __init__(self, text: str) -> None:
                self._t = text

            def complete(
                self,
                messages: list[dict[str, str]],
                *,
                sampling: SamplingParams | None = None,
            ) -> str:
                return self._t

        _rf = {"type": "json_object"}
        j_sdk, j_wire = _surfaces(lambda: _JsonB('{"a": 1}'))
        _req = {**oai_body, "response_format": _rf}
        jw = j_wire.post("/v1/chat/completions", json=_req)
        js, _ = j_sdk.openai_chat(_req)
        out["openai_json_object_parity"] = (
            jw.status_code == 200
            and jw.json()["choices"][0]["message"]["content"] == '{"a": 1}'
            and js.choices[0].message["content"] == '{"a": 1}'
        )
        b_sdk, b_wire = _surfaces(lambda: _JsonB("not json"))
        bw = b_wire.post("/v1/chat/completions", json=_req)
        bs_err = _raises(lambda: b_sdk.openai_chat(_req))[0]
        out["openai_format_violation_parity"] = (
            bw.status_code == 502
            and bw.json()["error"]["code"] == "format_violation"
            and bs_err == "OpenAICompatError"
        )
        # stream surfaces run the same post-validation — a violation on
        # stream:true 502s on the wire and raises in the SDK, no chunks.
        bsw = b_wire.post("/v1/chat/completions", json={**_req, "stream": True})
        bst_err = _raises(lambda: b_sdk.openai_chat_stream({**_req, "stream_options": {}}))[0]
        out["openai_format_violation_stream_parity"] = (
            bsw.status_code == 502 and bst_err == "OpenAICompatError"
        )
        # Last-Event-ID resume parity: the wire drops frames <= the given
        # index when replaying a keyed stream; the SDK's last_event_id
        # filter returns the identical chunk suffix.
        _rs_body = {**_req, "stream": True}
        rs1 = j_wire.post(
            "/v1/chat/completions",
            json=_rs_body,
            headers={"Idempotency-Key": "pa-rs1"},
        )
        rs2 = j_wire.post(
            "/v1/chat/completions",
            json=_rs_body,
            headers={"Idempotency-Key": "pa-rs1", "Last-Event-ID": "1"},
        )
        wire_resume = [
            json.loads(ln[len("data: ") :])
            for ln in rs2.text.splitlines()
            if ln.startswith("data: ") and ln[len("data: ") :].strip() != "[DONE]"
        ]
        sdk_resume, _ = j_sdk.openai_chat_stream(_req, last_event_id=1)
        out["openai_resume_parity"] = (
            rs1.status_code == 200
            and [_strip_meta(f) for f in wire_resume] == [_strip_meta(f) for f in sdk_resume]
            and rs2.headers.get("X-Fx1-Idempotent-Replay") == "true"
            and "id: 0" not in rs2.text
        )
        # a bad resume cursor fails closed on both surfaces — 400 on the
        # wire, ValueError in-process
        rs_wire_bad = j_wire.post(
            "/v1/chat/completions",
            json=_rs_body,
            headers={"Idempotency-Key": "pa-rs1", "Last-Event-ID": "-1"},
        )
        rs_sdk_bad = _raises(lambda: j_sdk.openai_chat_stream(_req, last_event_id=-1))[0]
        out["openai_resume_bad_id_parity"] = (
            rs_wire_bad.status_code == 400 and rs_sdk_bad == "ValueError"
        )
        # rejection parity: same verdict, each surface's own exception class
        oai_bad_n = {**oai_body, "n": 9}
        wire_n_err = client.post("/v1/chat/completions", json=oai_bad_n)
        sdk_n_err = _raises(lambda: sdk.openai_chat(oai_bad_n))[0]
        out["openai_validation_parity"] = (
            wire_n_err.status_code == 422
            and "error" in wire_n_err.json()
            and sdk_n_err == "ValidationError"
        )
        # decode-contract parity: stop / n / declared params behave the
        # same over the wire and in-process
        oai_stop = {**oai_body, "stop": "pi", "user": "u-p", "metadata": {"t": "p"}}
        wire_stop = client.post("/v1/chat/completions", json=oai_stop)
        sdk_stop_env, sdk_stop_cid = sdk.openai_chat(oai_stop)
        out["openai_stop_parity"] = (
            wire_stop.status_code == 200
            and wire_stop.json()["choices"][0]["message"]["content"]
            == sdk_stop_env.choices[0].message["content"]
            == "echo:"
            and bool(sdk_stop_cid)
            and sdk.completion(sdk_stop_cid or "").user == "u-p"
            and sdk.completion(sdk_stop_cid or "").metadata == {"t": "p"}
            and client.get(
                f"/harness/completions/{wire_stop.headers['X-Fx1-Completion-Id']}"
            ).json()["user"]
            == "u-p"
        )
        oai_n2 = {**oai_body, "n": 2}
        wire_n2 = client.post("/v1/chat/completions", json=oai_n2)
        sdk_n2_env, _ = sdk.openai_chat(oai_n2)
        out["openai_n_parity"] = (
            wire_n2.status_code == 200
            and [c["index"] for c in wire_n2.json()["choices"]] == [0, 1]
            and [c.index for c in sdk_n2_env.choices] == [0, 1]
            and wire_n2.json()["choices"][1]["message"]["content"]
            == sdk_n2_env.choices[1].message["content"]
        )
        oai_pen = {**oai_body, "presence_penalty": 3.0}
        wire_pen_err = client.post("/v1/chat/completions", json=oai_pen)
        sdk_pen_err = _raises(lambda: sdk.openai_chat(oai_pen))[0]
        out["openai_penalty_range_parity"] = (
            wire_pen_err.status_code == 422 and sdk_pen_err == "ValidationError"
        )
        # n>1 stream parity — identical per-index frame sequences
        oai_n2s = {**oai_body, "n": 2, "stream": True}
        wire_n2s = client.post("/v1/chat/completions", json=oai_n2s)
        wire_n2f = [
            json.loads(ln[len("data: ") :])
            for ln in wire_n2s.text.splitlines()
            if ln.startswith("data: ") and ln[len("data: ") :].strip() != "[DONE]"
        ]
        sdk_n2f, _ = sdk.openai_chat_stream({**oai_body, "n": 2})
        out["openai_n_stream_parity"] = (
            wire_n2s.status_code == 200
            and [_strip_meta(f) for f in wire_n2f] == [_strip_meta(f) for f in sdk_n2f]
            and {f["choices"][0]["index"] for f in wire_n2f if f.get("choices")} == {0, 1}
        )
        oai_tools = {
            "model": "hosted_k3",
            "messages": [{"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]}],
        }
        wire_tool_err = client.post("/v1/chat/completions", json=oai_tools)
        sdk_tool_err = _raises(lambda: sdk.openai_chat(oai_tools))[0]
        out["openai_tool_reject_parity"] = (
            wire_tool_err.status_code == 400
            and wire_tool_err.json()["error"]["type"] == "invalid_request_error"
            and sdk_tool_err == "OpenAICompatError"
        )

        # tools channel parity — a capable link carries the same
        # tool_calls envelope, finish_reason, and null content on both
        # surfaces; a link without the channel is 501 / NotImplementedError
        sdk_t, client_t = _surfaces(_ParityToolBackend)
        tool_body = {
            "model": "fx1",
            "messages": [{"role": "user", "content": "calc"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "calc", "parameters": {"type": "object"}},
                }
            ],
            "tool_choice": "auto",
        }
        wire_te = client_t.post("/v1/chat/completions", json=tool_body)
        sdk_te, _te_cid = sdk_t.openai_chat(tool_body)
        wire_tm = wire_te.json()["choices"][0]["message"]
        sdk_tm = sdk_te.choices[0].message
        out["openai_tools_parity"] = (
            wire_te.status_code == 200
            and wire_tm.get("tool_calls") == sdk_tm.get("tool_calls")
            and wire_tm.get("content") is None
            and sdk_tm.get("content") is None
            and wire_te.json()["choices"][0]["finish_reason"]
            == sdk_te.choices[0].finish_reason
            == "tool_calls"
        )
        wire_tstream = client_t.post("/v1/chat/completions", json={**tool_body, "stream": True})
        wire_tframes = [
            json.loads(ln[len("data: ") :])
            for ln in wire_tstream.text.splitlines()
            if ln.startswith("data: ") and ln[len("data: ") :].strip() != "[DONE]"
        ]
        sdk_tframes, _ = sdk_t.openai_chat_stream(tool_body)
        out["openai_tools_stream_parity"] = [_strip_meta(f) for f in wire_tframes] == [
            _strip_meta(f) for f in sdk_tframes
        ] and any(
            f["choices"][0]["delta"].get("tool_calls") for f in sdk_tframes if f.get("choices")
        )
        wire_nc = client.post("/v1/chat/completions", json=tool_body)
        sdk_nc = _raises(lambda: sdk.openai_chat(tool_body))[0]
        out["openai_tools_501_parity"] = (
            wire_nc.status_code == 501
            and wire_nc.json()["error"]["type"] == "server_error"
            and sdk_nc == "NotImplementedError"
        )
        # logprobs channel parity — request fields reach the provider
        # verbatim on both surfaces and the payload lands on the same
        # envelope slot; a plain link is 501 / NotImplementedError
        lp_body = {
            "model": "fx1",
            "messages": [{"role": "user", "content": "calc"}],
            "logprobs": True,
            "top_logprobs": 2,
        }
        wire_lp = client_t.post("/v1/chat/completions", json=lp_body)
        sdk_lp, _ = sdk_t.openai_chat(lp_body)
        out["openai_logprobs_parity"] = (
            wire_lp.status_code == 200
            and wire_lp.json()["choices"][0]["logprobs"] == sdk_lp.choices[0].logprobs
        )
        wire_lps = client_t.post("/v1/chat/completions", json={**lp_body, "stream": True})
        wire_lp_frames = [
            json.loads(ln[len("data: ") :])
            for ln in wire_lps.text.splitlines()
            if ln.startswith("data: ") and ln[len("data: ") :].strip() != "[DONE]"
        ]
        sdk_lp_frames, _ = sdk_t.openai_chat_stream(lp_body)
        out["openai_logprobs_stream_parity"] = [_strip_meta(f) for f in wire_lp_frames] == [
            _strip_meta(f) for f in sdk_lp_frames
        ] and any(
            f["choices"][0]["delta"].get("logprobs") for f in sdk_lp_frames if f.get("choices")
        )
        wire_lnc = client.post("/v1/chat/completions", json=lp_body)
        sdk_lnc = _raises(lambda: sdk.openai_chat(lp_body))[0]
        out["openai_logprobs_501_parity"] = (
            wire_lnc.status_code == 501 and sdk_lnc == "NotImplementedError"
        )
        out["openai_toplogprobs_guard_parity"] = (
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "fx1",
                    "messages": [{"role": "user", "content": "h"}],
                    "top_logprobs": 2,
                },
            ).status_code
            == 422
            and _raises(
                lambda: sdk.openai_chat({"model": "fx1", "messages": msg, "top_logprobs": 2})
            )[0]
            == "ValidationError"
        )
        # agent history passes through verbatim on both surfaces
        tool_hist = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "2", "tool_call_id": "call_9"},
        ]
        wire_th = client_t.post(
            "/v1/chat/completions",
            json={"model": "fx1", "messages": tool_hist},
        )
        sdk_th = _raises(lambda: sdk_t.openai_chat({"model": "fx1", "messages": tool_hist}))
        out["openai_tool_history_parity"] = wire_th.status_code == 200 and sdk_th[0] == ""
        # header BYOK parity: gpt-4o model + X-Fx1-Byok-* headers bind byok on both
        byok_headers = {
            "X-Fx1-Byok-Base-Url": "https://api.invalid.test/v1",
            "X-Fx1-Byok-Api-Key": "test-key",
            "X-Fx1-Byok-Model": "gpt-4o",
        }
        oai_byok_req = {"model": "gpt-4o", "messages": msg}
        wire_byok = client.post("/v1/chat/completions", json=oai_byok_req, headers=byok_headers)
        sdk_byok_env, _ = sdk.openai_chat(oai_byok_req, headers=byok_headers)
        out["openai_byok_header_parity"] = (
            wire_byok.status_code == 200
            and wire_byok.json()["system_fingerprint"] == "byok"
            and sdk_byok_env.system_fingerprint == "byok"
        )

        # /v1/responses parity — one gated completion through either
        # surface carries the same text, the same echoed object fields,
        # the same fail-closed refusals, and the same completion log
        resp_body = {
            "model": "fx1",
            "instructions": "be terse",
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": "d"}]},
                {"role": "user", "content": "ping"},
            ],
            "max_output_tokens": 32,
            "reasoning": {"effort": "low"},
            "metadata": {"lane": "78"},
        }
        wire_resp = client.post("/v1/responses", json=resp_body)
        sdk_resp, sdk_resp_cid = sdk.openai_response(resp_body)
        wire_rd = wire_resp.json()
        out["openai_response_parity"] = (
            wire_resp.status_code == 200
            and wire_rd["object"] == "response"
            and wire_rd["output"][0]["content"][0]["text"]
            == sdk_resp["output"][0]["content"][0]["text"]
            == "echo:ping"
            and wire_rd["status"] == sdk_resp["status"] == "completed"
            and wire_rd["metadata"] == sdk_resp["metadata"] == {"lane": "78"}
            and wire_rd["reasoning"] == sdk_resp["reasoning"] == {"effort": "low"}
            and bool(sdk_resp_cid)
            and sdk.completion(sdk_resp_cid or "").metadata == {"lane": "78"}
        )
        # refused fields refuse identically on both surfaces
        resp_tools = {**resp_body, "truncation": "auto"}
        wire_rt = client.post("/v1/responses", json=resp_tools)
        sdk_rt = _raises(lambda: sdk.openai_response(resp_tools))[0]
        out["openai_response_refusal_parity"] = (
            wire_rt.status_code == 422
            and wire_rt.json()["error"]["type"] == "invalid_request_error"
            and sdk_rt == "ValidationError"
        )
        resp_item_refuse = {
            **resp_body,
            "input": [{"type": "computer_call", "role": "user", "content": "x"}],
        }
        wire_rf = client.post("/v1/responses", json=resp_item_refuse)
        sdk_rf = _raises(lambda: sdk.openai_response(resp_item_refuse))[0]
        out["openai_response_item_refusal_parity"] = (
            wire_rf.status_code == 400
            and wire_rf.json()["error"]["type"] == "invalid_request_error"
            and sdk_rf == "OpenAICompatError"
        )
        # stream parity — identical (event, payload) sequences modulo the
        # server-minted ids/timestamp
        resp_stream = {**resp_body, "stream": True}
        wire_rs = client.post("/v1/responses", json=resp_stream)
        wire_revents = [
            json.loads(ln[len("data: ") :])
            for ln in wire_rs.text.splitlines()
            if ln.startswith("data: ")
        ]
        sdk_revents, _ = sdk.openai_response_stream(resp_body)
        _volatile = ("id", "item_id", "created_at")

        def _resp_norm(p: dict[str, Any]) -> Any:
            p_ = {k: v for k, v in p.items() if k not in _volatile}
            if "response" in p_:
                r_ = {k: v for k, v in p_["response"].items() if k not in _volatile}
                r_["output"] = [
                    {k: v for k, v in i.items() if k != "id"} for i in p_["response"]["output"]
                ]
                p_["response"] = r_
            if "item" in p_:
                p_["item"] = {k: v for k, v in p_["item"].items() if k != "id"}
            return p_

        out["openai_response_stream_parity"] = (
            wire_rs.status_code == 200
            and [_resp_norm(p) for p in wire_revents] == [_resp_norm(p) for _e, p in sdk_revents]
            and [e for e, _p in sdk_revents] == [p["type"] for p in wire_revents]
        )

        # lane 82: the tool channel on /v1/responses — envelope, echo,
        # item folding, refusal shape, and stream grammar all identical
        # across the wire/SDK surfaces
        rtool_body = {
            "model": "fx1",
            "input": "calc one",
            "tools": [
                {
                    "type": "function",
                    "name": "calc",
                    "description": "arithmetic",
                    "parameters": {"type": "object"},
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": True,
        }
        wire_rtc = client_t.post("/v1/responses", json=rtool_body)
        sdk_rtc, _rtc_cid = sdk_t.openai_response(rtool_body)
        wire_rtc_o = wire_rtc.json()["output"]
        sdk_rtc_o = sdk_rtc["output"]
        out["openai_responses_tools_parity"] = (
            wire_rtc.status_code == 200
            and [{k: v for k, v in it.items() if k != "id"} for it in wire_rtc_o]
            == [{k: v for k, v in it.items() if k != "id"} for it in sdk_rtc_o]
            == [
                {
                    "type": "function_call",
                    "call_id": "call_p",
                    "name": "calc",
                    "arguments": '{"x": 1}',
                    "status": "completed",
                }
            ]
            and wire_rtc.json()["tool_choice"] == sdk_rtc["tool_choice"] == "required"
            and wire_rtc.json()["parallel_tool_calls"] == sdk_rtc["parallel_tool_calls"] is True
        )
        # fc/fco input items fold to the same shared history on both
        # surfaces — the tool-capable stub would see identical messages
        hist_body = {
            "model": "fx1",
            "input": [
                {"role": "user", "content": "q"},
                {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "calc",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_a",
                    "output": "2",
                },
                {"role": "user", "content": "and?"},
            ],
        }
        wire_hist = client_t.post("/v1/responses", json=hist_body)
        sdk_hist = _raises(lambda: sdk_t.openai_response(hist_body))[0]
        out["openai_responses_tool_items_parity"] = (
            wire_hist.status_code == 200
            and sdk_hist == ""
            and wire_hist.json()["output"][0]["type"] == "function_call"
        )
        # bounds refuse identically on both surfaces
        tool_bad = {**rtool_body, "tools": []}
        wire_tb = client_t.post("/v1/responses", json=tool_bad)
        sdk_tb = _raises(lambda: sdk_t.openai_response(tool_bad))[0]
        out["openai_responses_tool_bounds_parity"] = (
            wire_tb.status_code == 422 and sdk_tb == "ValidationError"
        )
        # a link without the channel 501s identically
        wire_tn = client.post("/v1/responses", json=rtool_body)
        sdk_tn = _raises(lambda: sdk.openai_response(rtool_body))[0]
        out["openai_responses_tools_501_parity"] = (
            wire_tn.status_code == 501 and sdk_tn == "NotImplementedError"
        )
        # stream: identical (event, payload) sequences modulo minted ids
        wire_rts = client_t.post("/v1/responses", json={**rtool_body, "stream": True})
        wire_rtevents = [
            json.loads(ln[len("data: ") :])
            for ln in wire_rts.text.splitlines()
            if ln.startswith("data: ")
        ]
        sdk_rtevents, _ = sdk_t.openai_response_stream(rtool_body)
        out["openai_responses_tools_stream_parity"] = (
            wire_rts.status_code == 200
            and [_resp_norm(p) for p in wire_rtevents] == [_resp_norm(p) for _e, p in sdk_rtevents]
            and [e for e, _p in sdk_rtevents] == [p["type"] for p in wire_rtevents]
            and "response.function_call_arguments.delta" in [p["type"] for p in wire_rtevents]
        )
        # lane 83: the logprobs channel on /v1/responses — include +
        # top_logprobs carry identically and the provider's array lands on
        # the output_text part on both surfaces (a message stub — a
        # call-only turn has no output_text part for scores to ride on)
        sdk_l, client_l = _surfaces(_ParityLpBackend)
        rlp_body = {
            "model": "fx1",
            "input": "calc one",
            "include": ["message.output_text.logprobs"],
            "top_logprobs": 2,
        }
        wire_rlp = client_l.post("/v1/responses", json=rlp_body)
        sdk_rlp, _ = sdk_l.openai_response(rlp_body)
        wire_rlp_lp = wire_rlp.json()["output"][0]["content"][0].get("logprobs")
        sdk_rlp_lp = sdk_rlp["output"][0]["content"][0].get("logprobs")
        out["openai_responses_logprobs_parity"] = (
            wire_rlp.status_code == 200
            and wire_rlp_lp == sdk_rlp_lp
            and isinstance(wire_rlp_lp, list)
            and wire_rlp_lp[0].get("token") == "x"
            and wire_rlp.json()["include"] == sdk_rlp["include"] == ["message.output_text.logprobs"]
        )
        # the include member is pinned to the logprobs channel on both
        # surfaces; top_logprobs without it refuses identically
        bad_inc = {**rlp_body, "include": ["bogus.member"]}
        wire_bi = client_l.post("/v1/responses", json=bad_inc)
        sdk_bi = _raises(lambda: sdk_l.openai_response(bad_inc))[0]
        out["openai_responses_include_parity"] = (
            wire_bi.status_code == 422 and sdk_bi == "ValidationError"
        )
        no_inc = {"model": "fx1", "input": "x", "top_logprobs": 1}
        wire_ni = client_l.post("/v1/responses", json=no_inc)
        sdk_ni = _raises(lambda: sdk_l.openai_response(no_inc))[0]
        out["openai_responses_toplogprobs_guard_parity"] = (
            wire_ni.status_code == 422 and sdk_ni == "ValidationError"
        )

        # ---- lane 84: the embeddings channel carries the same parity ----
        # contract: the canned provider answer arrives byte-identical on
        # the wire and in-process, the non-embedding link refuses 501 on
        # both, and the guards fire identically.
        sdk_e, client_e = _surfaces(_ParityEmbedBackend)
        eb_body = {
            "model": "emb-m",
            "input": ["a", "b"],
            "encoding_format": "float",
            "dimensions": 2,
            "user": "u",
        }
        wire_eb = client_e.post("/v1/embeddings", json=eb_body)
        sdk_eb, sdk_eb_cid = sdk_e.openai_embeddings(eb_body)
        out["openai_embeddings_parity"] = (
            wire_eb.status_code == 200
            and wire_eb.json() == sdk_eb
            and wire_eb.json()["object"] == "list"
            and len(wire_eb.json()["data"]) == 2
            and wire_eb.json()["model"] == "emb-m-v1"
            and isinstance(sdk_eb_cid, str)
            and isinstance(wire_eb.headers.get("X-Fx1-Completion-Id"), str)
        )
        # digests land in each surface's own completion log identically
        _rec_we = client_e.get(
            f"/harness/completions/{wire_eb.headers.get('X-Fx1-Completion-Id')}"
        ).json()
        _rec_se = sdk_e.completion(sdk_eb_cid or "")
        out["openai_embeddings_log_parity"] = (
            _rec_we.get("prompt_sha256") == _rec_se.prompt_sha256
            and _rec_we.get("output_sha256") == _rec_se.output_sha256
            and _rec_we.get("model") == _rec_se.model == "emb-m-v1"
            and _rec_we.get("ok") is _rec_se.ok is True
        )
        sdk_ne, client_ne = _surfaces(_NonStreamingBackend)
        wire_ne = client_ne.post("/v1/embeddings", json={"model": "m", "input": "x"})
        sdk_ne_err = _raises(lambda: sdk_ne.openai_embeddings({"model": "m", "input": "x"}))[0]
        out["openai_embeddings_no_channel_parity"] = (
            wire_ne.status_code == 501 and sdk_ne_err == "NotImplementedError"
        )
        eb_bad = {"model": "emb-m", "input": []}
        wire_eg = client_e.post("/v1/embeddings", json=eb_bad)
        sdk_eg = _raises(lambda: sdk_e.openai_embeddings(eb_bad))[0]
        out["openai_embeddings_guard_parity"] = (
            wire_eg.status_code == 422 and sdk_eg == "ValidationError"
        )

        from fx1.serve.byok_audit import byok_audit_bench

        good = byok_audit_bench()
        sdk_v = sdk.verify_receipt(good)
        api_v = client.post("/receipts/verify", json={"receipt": good}).json()
        out["verify_valid_parity"] = (
            sdk_v.valid is True
            and api_v["valid"] is True
            and api_v["schema_tag"] == sdk_v.schema_tag
            and api_v["kind"] == sdk_v.kind
            and api_v["verdict"] == sdk_v.verdict
            and tuple(api_v["errors"]) == sdk_v.errors
            and tuple(api_v["warnings"]) == sdk_v.warnings
        )
        tampered = dict(good)
        tampered["receipt_sha256"] = "0" * 64
        sdk_tv = sdk.verify_receipt(tampered)
        api_tv = client.post("/receipts/verify", json={"receipt": tampered}).json()
        out["verify_tamper_parity"] = (
            sdk_tv.valid is False
            and api_tv["valid"] is False
            and tuple(api_tv["errors"]) == sdk_tv.errors
        )
        sdk_mv = sdk.verify_receipt({"not": "a receipt"})
        api_mv = client.post("/receipts/verify", json={"receipt": {"not": "a receipt"}}).json()
        out["verify_malformed_parity"] = (
            sdk_mv.valid is False
            and api_mv["valid"] is False
            and tuple(api_mv["errors"]) == sdk_mv.errors
        )

        # --- sealed-receipt store parity: SDK index vs HTTP fetch ------------
        # One content-addressed store, three transports: the SDK reads it
        # in-process, the API serves it, and HarnessClient fetches over the wire.
        import tempfile as _tfd  # noqa: PLC0415
        from pathlib import Path as _Ptd  # noqa: PLC0415

        with _tfd.TemporaryDirectory() as _td:
            _rdir = _Ptd(_td)
            _sha = good["receipt_sha256"]
            (_rdir / "sealed.json").write_text(json.dumps(good))
            from fastapi.testclient import TestClient as _TCr  # noqa: PLC0415

            import fx1.serve.api as _api_mod_r  # noqa: PLC0415
            from fx1.sdk import Fx1Harness as _FHr  # noqa: PLC0415
            from fx1.serve.client import HarnessClient as _HCr  # noqa: PLC0415

            _sdk_r = _FHr(receipts_dir=_rdir)
            _api_r = _TCr(_api_mod_r.create_app(receipts_dir=_rdir))
            _rem_r = _HCr("http://harness.test", transport=_tc_transport(_api_r))

            _idx = _api_r.get("/receipts").json()
            _expect = ((_sha, "sealed.json"),)
            out["receipts_store_index_parity"] = (
                _idx["count"] == 1
                and _idx["items"][0]["sha256"] == _sha
                and tuple((r.sha256, r.name) for r in _sdk_r.receipts()) == _expect
                and tuple((r.sha256, r.name) for r in _rem_r.receipts()) == _expect
            )
            _sdoc = _sdk_r.receipt(_sha)
            _rdoc = _rem_r.receipt(_sha)
            out["receipts_store_fetch_parity"] = (
                _sdoc.document == good
                and _sdoc.valid is True
                and _rdoc.document == good
                and _rdoc.valid is True
                and _api_r.get(f"/receipts/{_sha}").json() == good
            )
            _miss = "f" * 64
            out["receipts_store_error_parity"] = (
                _raises(lambda: _sdk_r.receipt(_miss))[0] == "KeyError"
                and _raises(lambda: _rem_r.receipt(_miss))[0] == "KeyError"
                and _raises(lambda: _sdk_r.receipt("zz"))[0] == "ValueError"
                and _raises(lambda: _rem_r.receipt("zz"))[0] == "ValueError"
            )
            out["receipts_store_conditional_304"] = (
                _api_r.get(f"/receipts/{_sha}", headers={"if-none-match": f'"{_sha}"'}).status_code
                == 304
                and _api_r.get(f"/receipts/{_sha}", headers={"if-none-match": "*"}).status_code
                == 304
                and _api_r.get(
                    f"/receipts/{_sha}",
                    headers={"if-none-match": f'"{"b" * 64}"'},
                ).status_code
                == 200
            )
        out["receipts_store_absent_sdk"] = (
            _raises(lambda: _FHr(receipts_dir="/nonexistent-zzz").receipts())[0]
            == "FileNotFoundError"
        )

        # --- registry / run / health parity -----------------------------------
        api_cmds = client.get("/harness/commands").json()
        out["commands_parity"] = sorted(sdk.commands()) == sorted(
            c["name"] for c in api_cmds["items"]
        ) and api_cmds["total"] == len(api_cmds["items"])
        name = sorted(sdk.commands())[0]
        sdk_run = sdk.run(name)
        api_run = client.post("/harness/runs", json={"command": name}).json()
        out["run_parity"] = (
            api_run["exit_code"] == sdk_run.exit_code
            and api_run["stdout"] == sdk_run.stdout
            and api_run["ok"] == sdk_run.ok
        )
        api_health = client.get("/health").json()
        sdk_health = sdk.health()
        out["health_parity"] = (
            api_health["backends"] == sdk_health.backends
            and api_health["version"] == sdk_health.version
            and api_health["registered_commands"] == sdk_health.registered_commands
        )

        # --- per-request BYOK parity --------------------------------------------------
        # SDK ``byok=`` and wire ``byok`` must hand the backend factory the
        # identical kwargs — and a wrong-backend override must fail with the
        # same class on both surfaces.
        cap_kwargs: list[dict[str, Any]] = []

        def _cap_res(name: str, **kw: Any) -> Any:
            cap_kwargs.append(kw)
            return _ParityBackend()

        from fastapi.testclient import TestClient as _TCb  # noqa: PLC0415

        import fx1.serve.api as api_mod2  # noqa: PLC0415
        from fx1.harness import Harness as _Hb  # noqa: PLC0415
        from fx1.sdk import Fx1Harness as _FHb  # noqa: PLC0415

        def _fake_run(argv: list[str], t: int) -> tuple[int, str, str]:
            return 0, "ran", ""

        sdk_byok = _FHb(harness=_Hb(runner=_fake_run), backend_resolver=_cap_res)
        app_byok = _TCb(
            api_mod2.create_app(harness=_Hb(runner=_fake_run), backend_resolver=_cap_res)
        )
        ovr = {
            "base_url": "https://llm.example.com/v1",
            "api_key": "sk-x",
            "model": "m1",
        }
        sdk_byok.complete(msg, backend="byok", byok=ovr)
        sdk_wire_kwargs = dict(cap_kwargs[-1])
        cap_kwargs.clear()
        r_byok = app_byok.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": msg,
                "byok": ovr,
            },
        )
        out["byok_override_kwargs_identical"] = (
            r_byok.status_code == 200 and cap_kwargs[-1] == sdk_wire_kwargs
        )
        # Per-request backend deadline: SDK ``timeout_s=`` and the wire
        # ``timeout_s`` must reach the resolver as the same kwarg.
        sdk_byok.complete(msg, backend="byok", byok=ovr, timeout_s=2.5)
        sdk_to_kwargs = dict(cap_kwargs[-1])
        cap_kwargs.clear()
        r_to = app_byok.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": msg,
                "byok": ovr,
                "timeout_s": 2.5,
            },
        )
        out["timeout_s_kwargs_identical"] = (
            r_to.status_code == 200 and cap_kwargs[-1] == sdk_to_kwargs
        )
        sdk_err, _ = _raises(lambda: sdk_byok.complete(msg, backend="hosted_k3", byok=ovr))
        api_code = app_byok.post(
            "/harness/complete",
            json={
                "backend": "hosted_k3",
                "messages": msg,
                "byok": ovr,
            },
        ).status_code
        out["byok_override_error_parity"] = sdk_err == "ValueError" and api_code == 422
        out["byok_override_bad_url_parity"] = (
            _raises(
                lambda: sdk_byok.complete(msg, backend="byok", byok={**ovr, "base_url": "ftp://x"})
            )[0]
            == "ValueError"
        )
        # the typed ``byok`` param is exactly the ``backend_kwargs`` merge
        sdk_byok.complete(msg, backend="byok", backend_kwargs=ovr)
        out["byok_backend_kwargs_equivalent"] = cap_kwargs[-1] == sdk_wire_kwargs

        # --- HarnessClient: the remote-caller surface --------------------------------
        from fx1.serve.client import (
            HarnessAuthError,
            HarnessClient,
            HarnessCompatError,
            HarnessJobError,
            HarnessTransportError,
        )

        remote = HarnessClient("http://harness.test", transport=_tc_transport(client))
        rem_out = remote.complete(msg, backend="byok", receipt_hashes=[receipt])
        # completion_id is a per-surface minted handle — everything else must match.
        out["client_complete_identical"] = dataclasses.replace(
            rem_out, completion_id=None
        ) == dataclasses.replace(sdk_out, completion_id=None)
        # both surfaces logged a fetchable record carrying the same hashes
        rec_wire = remote.completion(rem_out.completion_id or "")
        rec_sdk = sdk.completion(sdk_out.completion_id or "")
        out["completion_log_parity"] = (
            rec_wire.prompt_sha256 == rec_sdk.prompt_sha256
            and rec_wire.output_sha256 == rec_sdk.output_sha256
            and rec_wire.ok is rec_sdk.ok is True
            and rec_wire.backend == rec_sdk.backend == "byok"
        )
        out["completion_log_list_parity"] = (
            len(remote.completions(limit=1)) == 1 and len(sdk.completions(limit=1)) == 1
        )
        # each surface seals its own record — the docs verify on their own
        # surface's verifier and carry the same cross-surface hashes.
        from quant_fund.research.receipt_v2 import (  # noqa: PLC0415
            verify_receipt_payload as _vrp,
        )

        doc_wire = remote.completion_receipt(rem_out.completion_id or "")
        doc_sdk = sdk.completion_receipt(sdk_out.completion_id or "")
        out["completion_receipt_parity"] = (
            doc_wire["schema"] == doc_sdk["schema"] == "fx1_completion_record.v1"
            and doc_wire["record"]["prompt_sha256"] == doc_sdk["record"]["prompt_sha256"]
            and doc_wire["record"]["output_sha256"] == doc_sdk["record"]["output_sha256"]
            and _vrp(doc_wire)["valid"] is True
            and _vrp(doc_sdk)["valid"] is True
            and remote.verify_receipt(doc_wire).valid is True
            and sdk.verify_receipt(doc_sdk).valid is True
        )
        # async jobs seal the same way: the wire's fx1_job_record.v1 export
        # embeds the digested run result, which must equal the SDK's own
        # run_result seal on the shared fields.
        import time as _time  # noqa: PLC0415

        j_id = client.post("/harness/jobs", json={"command": name}).json()["job_id"]
        for _ in range(500):
            if client.get(f"/harness/jobs/{j_id}").json()["status"] in (
                "succeeded",
                "failed",
                "cancelled",
            ):
                break
            _time.sleep(0.01)
        doc_j = remote.job_receipt(j_id)
        doc_r = sdk.run_receipt(sdk_run)
        shared_keys = ("command", "exit_code", "ok", "stdout_sha256", "stderr_sha256")
        out["job_receipt_parity"] = (
            doc_j["schema"] == "fx1_job_record.v1"
            and doc_r["schema"] == "fx1_run_result.v1"
            and {k: doc_j["record"]["result"][k] for k in shared_keys}
            == {k: doc_r["record"][k] for k in shared_keys}
            and _vrp(doc_j)["valid"] is True
            and _vrp(doc_r)["valid"] is True
        )
        # fallback chain parity: the same ordered failover contract runs
        # in-process — primary faults, the next link serves, and both
        # surfaces record the same attempt trace.
        import fx1.serve.api as api_mod2  # noqa: PLC0415
        from fx1 import sdk as sdk_mod  # noqa: PLC0415

        _boom = _CallFailBackend()

        def _chain_resolver(name: str, **kw: Any) -> Any:
            if name == "hosted_k3":
                return _boom
            if name == "byok":
                return _ParityBackend()
            raise KeyError(name)

        sdk_chain = sdk_mod.Fx1Harness(
            harness=_Hb(runner=_fake_run), backend_resolver=_chain_resolver
        )
        app_chain = api_mod2.create_app(
            harness=_Hb(runner=_fake_run), backend_resolver=_chain_resolver
        )
        client_chain = _TCb(app_chain)
        remote_chain = HarnessClient("http://harness.test", transport=_tc_transport(client_chain))
        s_fb = sdk_chain.complete(msg, backend="hosted_k3", fallbacks=["byok"])
        w_fb = remote_chain.complete(msg, backend="hosted_k3", fallbacks=["byok"])

        def _norm(atts: Iterable[dict[str, Any]]) -> list[tuple[Any, ...]]:
            return [(a["backend"], a["ok"], a.get("error_class")) for a in atts]

        out["fallback_chain_parity"] = (
            s_fb.backend == w_fb.backend == "byok"
            and s_fb.content == w_fb.content
            and _norm(s_fb.attempts)
            == _norm(w_fb.attempts)
            == [("hosted_k3", False, "RuntimeError"), ("byok", True, None)]
        )
        out["fallback_exhaust_parity"] = (
            _raises(
                lambda: sdk_mod.Fx1Harness(
                    harness=_Hb(runner=_fake_run),
                    backend_resolver=lambda *a, **k: _boom,
                ).complete(msg, backend="hosted_k3", fallbacks=["byok"])
            )[0]
            == "RuntimeError"
        )
        out["client_stream_identical"] = (
            remote.stream_complete(msg, backend="byok", receipt_hashes=[receipt]) == sdk_chunks
        )
        out["client_batch_identical"] = [
            r.content
            for r in remote.complete_many(
                batch, backend="byok", receipt_hashes=[receipt], max_workers=2
            )
        ] == [r.content for r in sdk_many]
        rem_v = remote.verify_receipt(good)
        out["client_verify_parity"] = (
            rem_v.valid is True
            and rem_v.schema_tag == sdk_v.schema_tag
            and rem_v.errors == sdk_v.errors
        )
        rem_tv = remote.verify_receipt(tampered)
        out["client_verify_tamper_parity"] = (
            rem_tv.valid is False and rem_tv.errors == sdk_tv.errors
        )
        # wire-only flag: a keyed retry replays instead of re-billing
        remote.complete(msg, backend="byok", receipt_hashes=[receipt], idempotency_key="pk1")
        replayed = remote.complete(
            msg, backend="byok", receipt_hashes=[receipt], idempotency_key="pk1"
        )
        out["client_complete_idem_replay"] = replayed.replayed is True
        rem_batch = remote.verify_receipts([good, tampered])
        sdk_batch = sdk.verify_receipts([good, tampered])
        out["client_verify_batch_parity"] = (
            len(rem_batch) == 2
            and rem_batch[0].valid is True
            and rem_batch[1].valid is False
            and rem_batch[0].errors == sdk_batch[0].errors
            and rem_batch[1].errors == sdk_batch[1].errors
        )
        out["client_commands_parity"] = sorted(remote.commands()) == sorted(sdk.commands())
        rem_run = remote.run(name)
        out["client_run_parity"] = (
            rem_run.exit_code == sdk_run.exit_code
            and rem_run.stdout == sdk_run.stdout
            and rem_run.ok == sdk_run.ok
        )
        rem_health = remote.health()
        out["client_health_parity"] = (
            rem_health.backends == sdk_health.backends and rem_health.version == sdk_health.version
        )
        filtered = remote.commands(role="evaluation")
        out["client_role_filter"] = 0 < len(filtered) <= len(remote.commands()) and set(
            filtered
        ) <= set(remote.commands())

        # usage accounting: the same endpoint counts surface identically on
        # the in-process SDK, the wire JSON, and the remote client.
        class _UsageBackend(_ParityBackend):
            def __init__(self) -> None:
                super().__init__()
                self.last_usage: dict[str, int] | None = None
                self.total_usage: dict[str, int] = {}

            def complete(
                self,
                messages: list[dict[str, str]],
                *,
                sampling: SamplingParams | None = None,
            ) -> str:
                u = {"prompt_tokens": 2, "total_tokens": 5}
                self.last_usage = u
                for k, v in u.items():
                    self.total_usage[k] = self.total_usage.get(k, 0) + v
                return super().complete(messages, sampling=sampling)

        sdk_u, uclient = _surfaces(_UsageBackend)
        sdk_usage = sdk_u.complete(msg, backend="byok").usage
        wire_usage = uclient.post(
            "/harness/complete",
            json={"backend": "byok", "messages": msg},
        ).json()["usage"]
        remote_u = HarnessClient("http://harness.test", transport=_tc_transport(uclient))
        out["usage_identical"] = (
            sdk_usage == {"prompt_tokens": 2, "total_tokens": 5}
            and wire_usage == sdk_usage
            and remote_u.complete(msg, backend="byok").usage == sdk_usage
        )

        # sampling controls: identical resolved set on all three surfaces —
        # SDK kwargs, wire JSON fields, and the remote client all land the
        # same body_fields dict, and the default stays temperature-pinned.
        want_s = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 16, "seed": 7}
        sdk_s = sdk_u.complete(
            msg, backend="byok", temperature=0.7, top_p=0.9, max_tokens=16, seed=7
        ).sampling
        wire_s = uclient.post(
            "/harness/complete",
            json={
                "backend": "byok",
                "messages": msg,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 16,
                "seed": 7,
            },
        ).json()["sampling"]
        remote_s = remote_u.complete(
            msg, backend="byok", temperature=0.7, top_p=0.9, max_tokens=16, seed=7
        ).sampling
        out["sampling_parity"] = (
            sdk_s == want_s
            and wire_s == want_s
            and remote_s == want_s
            and sdk_u.complete(msg, backend="byok").sampling == {"temperature": 0.0}
            and remote_u.complete(msg, backend="byok").sampling == {"temperature": 0.0}
        )

        # deep-health probe: in-process verdict mirrors the wire verdict —
        # latency differs per call so only the semantic fields are pinned.
        sdk_pr = sdk_u.probe_backend("byok")
        wire_pr = remote_u.probe_backend("byok")
        out["probe_parity"] = (
            sdk_pr.ok is True
            and wire_pr.ok is True
            and wire_pr.backend == sdk_pr.backend == "byok"
            and wire_pr.model == sdk_pr.model == "parity-v0"
            and wire_pr.error is None
            and sdk_pr.error is None
        )

        # gate pre-flight: one in-process check and the wire endpoint agree
        gc_clean_s = sdk_u.check_text("bootstrap intervals were used")
        gc_bad_s = sdk_u.check_text("we report Sharpe 2.1")
        gc_bad_w = remote_u.check_text("we report Sharpe 2.1")
        out["gate_check_parity"] = (
            gc_clean_s.ok is True
            and gc_bad_s.ok is False
            and gc_bad_w.ok is False
            and gc_bad_w.error == gc_bad_s.error
        )

        # reward-score surface: in-process breakdowns are byte-identical to
        # the wire's — total/components/violations per item, same input
        # contract (str or list) and the same guard faults.
        score_in = [
            "verify-research pins receipt 0123456789abcdef",
            "",
            "Sharpe 3.2 live trading NAV up",
        ]
        out["score_parity"] = (
            sdk_u.score(score_in) == remote_u.score(score_in)
            and sdk_u.score("x") == remote_u.score("x")
            and len(sdk_u.score(score_in)) == 3
        )
        out["score_guard_parity"] = (
            _raises(lambda: sdk_u.score([]))[0] == "ValidationError"
            and _raises(lambda: remote_u.score([]))[0] == "ValueError"
        )

        # moderation surface: the SDK's in-process classification is
        # byte-identical to the wire payload — flagged/categories/scores per
        # input plus the content-derived modr- id, same input contract and
        # the same guard faults.
        mod_in = [
            "clean text",
            "Sharpe 3.2 live trading NAV up",
            "the synthetic results show accuracy 0.99",
        ]
        out["moderate_parity"] = (
            sdk_u.moderate(mod_in) == remote_u.moderate(mod_in)
            and sdk_u.moderate("x") == remote_u.moderate("x")
            and len(sdk_u.moderate(mod_in)["results"]) == 3
        )
        out["moderate_guard_parity"] = (
            _raises(lambda: sdk_u.moderate([]))[0] == "ValidationError"
            and _raises(lambda: remote_u.moderate([]))[0] == "ValueError"
        )

        # eval-diff surface: the promotion-gate primitive is the same
        # contract in-process and over the wire — same-suite same-bank
        # records are comparable and verdict-classified identically; the
        # unknown-id fault maps KeyError on both surfaces.
        ev_a = sdk_u.run_eval("tooluse", backend="byok", seed=0)
        ev_b = sdk_u.run_eval("tooluse", backend="byok", seed=0)
        sdk_diff = sdk_u.eval_diff(ev_a.eval_id, ev_b.eval_id)
        rv_a = remote_u.submit_eval("tooluse", backend="byok", seed=0)
        rv_b = remote_u.submit_eval("tooluse", backend="byok", seed=0)
        remote_u.wait_eval(rv_a["eval_id"], timeout_s=120)
        remote_u.wait_eval(rv_b["eval_id"], timeout_s=120)
        wire_diff = remote_u.diff_evals(rv_a["eval_id"], rv_b["eval_id"])

        def _diff_norm(d: Any) -> dict[str, Any]:
            dd = d.model_dump(mode="json") if hasattr(d, "model_dump") else dict(d)
            dd.pop("base_eval_id", None)
            dd.pop("candidate_eval_id", None)
            return dd

        out["eval_diff_parity"] = (
            _diff_norm(sdk_diff) == _diff_norm(wire_diff)
            and sdk_diff.same_suite
            and sdk_diff.same_seed
            and sdk_diff.comparable
            and sdk_diff.verdict == "unchanged"
        )
        out["eval_diff_unknown_parity"] = (
            _raises(lambda: sdk_u.eval_diff("nope", ev_b.eval_id))[0] == "KeyError"
            and _raises(lambda: remote_u.diff_evals("nope", rv_b["eval_id"]))[0] == "KeyError"
        )

        # fine-tuning surface: the in-process twin takes the corpus inline
        # and runs the same runner contract synchronously; the wire twin
        # uploads a file, submits, and polls. Both land terminal-succeeded
        # with the same ft: model name and an event feed; the model guard,
        # unknown-id, and terminal-cancel faults map identically (400/409 →
        # HarnessTransportError on the wire, ValueError in-process).
        from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

        def _ft_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
            emit("info", "runner working")
            art = spec.work_dir / "receipt.json"
            art.write_text("{}")
            return FTJobOutcome(
                fine_tuned_model=spec.ft_model_name, artifacts={"receipt.json": art}
            )

        sdk_ft, ft_wire = _surfaces(_UsageBackend, ft_runner=_ft_runner)
        remote_ft = HarnessClient("http://harness.test", transport=_tc_transport(ft_wire))
        _corpus = (
            b'{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"}]}\n'
        )
        sjob = sdk_ft.create_finetune_job(model="fx1", training_jsonl=_corpus, suffix="pp")
        fid_w = remote_ft.upload_file(_corpus, filename="c.jsonl", purpose="fine-tune")["id"]
        wjob = remote_ft.create_finetune_job(model="fx1", training_file=fid_w, suffix="pp")
        wfin = remote_ft.wait_finetune_job(wjob["id"], timeout_s=30)
        out["ft_parity"] = (
            sjob.status == "succeeded"
            and sjob.object == "fine_tuning.job"
            and wfin["status"] == "succeeded"
            and wfin["fine_tuned_model"] == f"ft:fx1:pp:{wjob['id'].split('-', 1)[1][:12]}"
            and sjob.fine_tuned_model == f"ft:fx1:pp:{sjob.id.split('-', 1)[1][:12]}"
            and len(wfin["result_files"]) == 1
            and sdk_ft.finetune_job(sjob.id).id == sjob.id
        )
        out["ft_events_parity"] = (
            len(remote_ft.finetune_job_events(wjob["id"])["data"]) >= 2
            and len(sdk_ft.finetune_job_events(sjob.id)) >= 2
        )
        out["ft_list_parity"] = wjob["id"] in {
            j["id"] for j in remote_ft.finetune_jobs()["data"]
        } and sjob.id in {j.id for j in sdk_ft.finetune_jobs()}
        out["ft_guards_parity"] = (
            _raises(lambda: sdk_ft.create_finetune_job(model="byok", training_jsonl=_corpus))[0]
            == "ValueError"
            and _raises(lambda: remote_ft.create_finetune_job(model="byok", training_file=fid_w))[0]
            == "HarnessTransportError"
            and _raises(lambda: sdk_ft.finetune_job("ftjob-nope"))[0] == "KeyError"
            and _raises(lambda: remote_ft.finetune_job("ftjob-nope"))[0] == "KeyError"
            and _raises(lambda: remote_ft.cancel_finetune_job(wjob["id"]))[0]
            == "HarnessTransportError"
        )
        # error mapping: the wire's codes map back to the SDK's classes
        dirty_remote = HarnessClient("http://harness.test", transport=_tc_transport(dirty_api))
        out["client_gate_maps_fx1honesty"] = (
            _raises(lambda: dirty_remote.complete(msg, backend="byok"))[0] == "Fx1HonestyError"
        )
        out["client_stream_gate_maps"] = (
            _raises(lambda: dirty_remote.stream_complete(msg, backend="byok"))[0]
            == "Fx1HonestyError"
        )
        ns_remote = HarnessClient("http://harness.test", transport=_tc_transport(ns_api))
        out["client_501_maps"] = (
            _raises(lambda: ns_remote.stream_complete(msg, backend="byok"))[0]
            == "NotImplementedError"
        )
        out["client_404_maps_keyerror"] = (
            _raises(lambda: remote.run("no-such-command"))[0] == "KeyError"
        )
        out["client_422_maps_valueerror"] = (
            _raises(lambda: remote.complete(msg, backend="bogus"))[0] == "ValueError"
        )
        # per-item batch failure: lowest-index gate refusal raises on the client
        flaky_sdk, flaky_api = _surfaces(_FlakyBackend)
        flaky_remote = HarnessClient("http://harness.test", transport=_tc_transport(flaky_api))
        mixed = [
            [{"role": "user", "content": "fine"}],
            [{"role": "user", "content": "bad"}],
        ]
        out["client_batch_gate_maps"] = (
            _raises(lambda: flaky_remote.complete_many(mixed, backend="byok"))[0]
            == "Fx1HonestyError"
        )
        out["sdk_batch_gate_raises"] = (
            _raises(lambda: flaky_sdk.complete_many(mixed, backend="byok"))[0] == "Fx1HonestyError"
        )

        # transport-level faults
        def _dead_transport(*a: Any) -> Any:
            raise HarnessTransportError("connection refused")

        out["client_transport_unreachable"] = (
            _raises(
                lambda: HarnessClient("http://harness.test", transport=_dead_transport).health()
            )[0]
            == "HarnessTransportError"
        )
        out["client_auth_error_maps"] = (
            _raises(
                lambda: HarnessClient(
                    "http://harness.test",
                    transport=lambda *a: (401, {}, b'{"detail":"denied"}'),
                ).health()
            )[0]
            == HarnessAuthError.__name__
        )
        out["client_base_url_fail_closed"] = all(
            _raises(lambda u=u: HarnessClient(u))[0] == "ValueError"
            for u in ("not-a-url", "ftp://x", "http://")
        )
        out["client_timeout_fail_closed"] = (
            _raises(lambda: HarnessClient("http://h.test", timeout_s=0))[0] == "ValueError"
        )
        captured: list[dict[str, str]] = []

        def _capture(method: str, url: str, payload: Any, headers: dict[str, str], t: float) -> Any:
            captured.append(headers)
            return (200, {}, b'{"status":"ok","version":"v","registered_commands":0,"backends":{}}')

        HarnessClient("http://harness.test", api_key="probe-key", transport=_capture).health()
        out["client_api_key_header"] = (
            bool(captured) and captured[0].get("X-API-Key") == "probe-key"
        )
        # stream without terminal [DONE] is a transport fault, never data
        out["client_stream_no_done_fails"] = (
            _raises(
                lambda: HarnessClient(
                    "http://harness.test",
                    transport=lambda *a: (
                        200,
                        {},
                        b'data: {"type":"token","content":"x"}\n\n',
                    ),
                ).stream_complete(msg, backend="byok")
            )[0]
            == "HarnessTransportError"
        )

        # --- in-flight cap: saturation is an honest 503, never a queue hang -----
        import fx1.serve.api as api_mod
        from fx1.harness import Harness

        def fake_runner2(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
            return 0, "ran", ""

        def resolver2(name: str, **kw: Any) -> Any:
            return _ParityBackend()

        capped_app = api_mod.create_app(
            harness=Harness(runner=fake_runner2),
            backend_resolver=resolver2,
            max_inflight=2,
        )
        from fastapi.testclient import TestClient as _TC

        capped = _TC(capped_app)
        slots = capped_app.state.inflight_slots
        out["cap_normal_request_passes"] = (
            capped.post(
                "/harness/complete",
                json={"backend": "byok", "messages": msg},
            ).status_code
            == 200
        )
        held1 = slots.acquire(blocking=False)
        held2 = slots.acquire(blocking=False)
        saturated = capped.post(
            "/harness/complete",
            json={"backend": "byok", "messages": msg},
        )
        out["cap_saturated_503"] = (
            held1
            and held2
            and saturated.status_code == 503
            and "max_inflight" in saturated.json()["detail"]
        )
        out["cap_client_maps_503"] = (
            _raises(
                lambda: HarnessClient(
                    "http://harness.test", transport=_tc_transport(capped)
                ).complete(msg, backend="byok")
            )[0]
            == "BackendNotConfiguredError"
        )
        out["cap_runs_also_capped"] = (
            capped.post("/harness/runs", json={"command": name}).status_code == 503
        )
        out["cap_cheap_routes_respond"] = (
            capped.get("/health").status_code == 200
            and capped.get("/harness/commands").status_code == 200
        )
        if held1:
            slots.release()
        if held2:
            slots.release()
        out["cap_released_recovers"] = (
            capped.post(
                "/harness/complete",
                json={"backend": "byok", "messages": msg},
            ).status_code
            == 200
        )
        out["cap_invalid_config_fails"] = (
            _raises(
                lambda: api_mod.create_app(harness=Harness(runner=fake_runner2), max_inflight=0)
            )[0]
            == "ValueError"
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # --- resilience: bounded retries + Retry-After honoring -----------------
    from fx1.serve.client import HarnessTransportError  # noqa: PLC0415

    def _scripted(
        seq: list[Any],
    ) -> tuple[Any, dict[str, int]]:
        calls = {"n": 0}

        def transport(
            method: str,
            url: str,
            payload: Any,
            headers: Any,
            timeout_s: float,
        ) -> tuple[int, Mapping[str, str], bytes]:
            item = seq[min(calls["n"], len(seq) - 1)]
            calls["n"] += 1
            if item == "RAISE":
                raise HarnessTransportError("boom")
            status, hdrs, body = item
            return status, hdrs, body

        return transport, calls

    import json as _json_mod  # noqa: PLC0415

    def _health_body() -> bytes:
        return _json_mod.dumps(
            {"status": "ok", "version": "v", "registered_commands": 1, "backends": {}}
        ).encode()

    sleeps: list[float] = []
    tr, calls = _scripted(["RAISE", "RAISE", (200, {}, _health_body())])
    resilient = HarnessClient(
        "http://h.test",
        transport=tr,
        max_retries=2,
        sleep=sleeps.append,
    )
    out["retry_get_recovers"] = (
        resilient.health().status == "ok" and calls["n"] == 3 and len(sleeps) == 2
    )

    sleeps.clear()
    tr2, calls2 = _scripted(
        [
            (503, {"Retry-After": "0.25"}, _json_mod.dumps({"detail": "cap"}).encode()),
            (200, {}, _health_body()),
        ]
    )
    resilient2 = HarnessClient("http://h.test", transport=tr2, max_retries=2, sleep=sleeps.append)
    out["retry_after_honored"] = (
        resilient2.health().status == "ok" and calls2["n"] == 2 and sleeps[0] == 0.25
    )

    tr429, calls429 = _scripted(
        [
            (
                429,
                {"Retry-After": "0.2"},
                _json_mod.dumps(
                    {"detail": "rate limit exceeded", "code": "too_many_requests"}
                ).encode(),
            ),
            (200, {}, _health_body()),
        ]
    )
    sleeps429: list[float] = []
    resilient429 = HarnessClient(
        "http://h.test", transport=tr429, max_retries=2, sleep=sleeps429.append
    )
    out["retry_429_rate_limited"] = (
        resilient429.health().status == "ok" and calls429["n"] == 2 and sleeps429 == [0.2]
    )

    tr3, calls3 = _scripted(
        [(503, {"Retry-After": "999"}, _json_mod.dumps({"detail": "cap"}).encode())]
    )
    resilient3 = HarnessClient("http://h.test", transport=tr3, max_retries=3, sleep=lambda s: None)
    out["retry_over_budget_fails_fast"] = (
        _raises(lambda: resilient3.health())[0] == "HarnessTransportError" and calls3["n"] == 1
    )

    tr4, calls4 = _scripted([(401, {}, _json_mod.dumps({"detail": "bad key"}).encode())])
    resilient4 = HarnessClient("http://h.test", transport=tr4, max_retries=3, sleep=lambda s: None)
    out["no_retry_on_auth"] = (
        _raises(lambda: resilient4.health())[0] == "HarnessAuthError" and calls4["n"] == 1
    )

    tr5, calls5 = _scripted(["RAISE"])
    resilient5 = HarnessClient("http://h.test", transport=tr5, max_retries=3, sleep=lambda s: None)
    out["writes_never_retry_by_default"] = (
        _raises(lambda: resilient5.complete([{"role": "user", "content": "hi"}]))[0]
        == "HarnessTransportError"
        and calls5["n"] == 1
    )

    verdict_body = _json_mod.dumps(
        {
            "valid": True,
            "path": "<remote>",
            "schema_tag": "x",
            "kind": "k",
            "verdict": "ok",
            "digest_convention": None,
            "errors": [],
            "warnings": [],
        }
    ).encode()
    tr6, calls6 = _scripted(["RAISE", (200, {}, verdict_body)])
    resilient6 = HarnessClient("http://h.test", transport=tr6, max_retries=1, sleep=lambda s: None)
    out["verify_is_idempotent_retried"] = (
        resilient6.verify_receipt({"x": 1}).valid is True and calls6["n"] == 2
    )

    complete_body = _json_mod.dumps(
        {"backend": "byok", "model": "m", "content": "c", "receipt_hashes": []}
    ).encode()
    tr6b, calls6b = _scripted(["RAISE", (200, {}, complete_body)])
    resilient6b = HarnessClient(
        "http://h.test",
        transport=tr6b,
        max_retries=1,
        retry_writes=True,
        sleep=lambda s: None,
    )
    out["writes_retry_when_opted_in"] = (
        resilient6b.complete([{"role": "user", "content": "hi"}]).content == "c"
        and calls6b["n"] == 2
    )

    # A keyed complete dedupes server-side — retries without the opt-in flag.
    tr6c, calls6c = _scripted(["RAISE", (200, {}, complete_body)])
    resilient6c = HarnessClient(
        "http://h.test",
        transport=tr6c,
        max_retries=1,
        sleep=lambda s: None,
    )
    out["keyed_complete_retries"] = (
        resilient6c.complete(
            [{"role": "user", "content": "hi"}],
            idempotency_key="k1",
        ).content
        == "c"
        and calls6c["n"] == 2
    )
    tr6d, calls6d = _scripted(
        [
            "RAISE",
            (
                200,
                {},
                _json_mod.dumps(
                    {"results": [], "backend": "byok", "model": "m", "receipt_hashes": []}
                ).encode(),
            ),
        ]
    )
    resilient6d = HarnessClient(
        "http://h.test",
        transport=tr6d,
        max_retries=1,
        sleep=lambda s: None,
    )
    out["keyed_batch_retries"] = (
        resilient6d.complete_many(
            [[{"role": "user", "content": "hi"}]],
            idempotency_key="k2",
        )
        == []
        and calls6d["n"] == 2
    )

    tr7, calls7 = _scripted(
        [(503, {}, _json_mod.dumps({"detail": "backend not configured"}).encode())]
    )
    resilient7 = HarnessClient("http://h.test", transport=tr7, max_retries=3, sleep=lambda s: None)
    out["unconfigured_503_not_retried"] = (
        _raises(lambda: resilient7.health())[0] == "BackendNotConfiguredError" and calls7["n"] == 1
    )

    out["retry_validation_failclosed"] = all(
        _raises(lambda kw=kw: HarnessClient("http://h.test", **kw))[0] == "ValueError"
        for kw in (
            {"max_retries": -1},
            {"retry_backoff_s": 0.0},
            {"max_retry_wait_s": 0.0},
            {"circuit_breaker_threshold": -1},
            {"circuit_reset_s": 0.0},
        )
    )

    # circuit breaker: transport faults open it; half-open probe closes it
    tr8, calls8 = _scripted(["RAISE"])
    t8 = [0.0]
    cb = HarnessClient(
        "http://h.test",
        transport=tr8,
        circuit_breaker_threshold=2,
        circuit_reset_s=30.0,
        clock=lambda: t8[0],
        sleep=lambda s: None,
    )
    for _ in range(2):
        _raises(lambda: cb.health())
    out["circuit_opens_and_fails_fast"] = (
        _raises(lambda: cb.health())[0] == "HarnessTransportError"
        and calls8["n"] == 2  # the third call never reached the wire
        and "circuit open" in str(_raises(lambda: cb.health())[1])
    )
    # half-open probe: advance the clock past the reset window; a healthy
    # transport closes the circuit
    tr9, calls9 = _scripted(["RAISE", "RAISE", (200, {}, _health_body())])
    t9 = [0.0]
    cb9 = HarnessClient(
        "http://h.test",
        transport=tr9,
        circuit_breaker_threshold=2,
        circuit_reset_s=30.0,
        clock=lambda: t9[0],
        sleep=lambda s: None,
    )
    for _ in range(2):
        _raises(lambda: cb9.health())
    t9[0] = 31.0
    out["circuit_half_open_closes"] = cb9.health().status == "ok" and calls9["n"] == 3
    # a failed half-open probe re-opens the window
    tr10, calls10 = _scripted(["RAISE"])
    t10 = [0.0]
    cb10 = HarnessClient(
        "http://h.test",
        transport=tr10,
        circuit_breaker_threshold=1,
        circuit_reset_s=30.0,
        clock=lambda: t10[0],
        sleep=lambda s: None,
    )
    _raises(lambda: cb10.health())  # trips (threshold 1)
    t10[0] = 31.0
    _raises(lambda: cb10.health())  # half-open probe fails -> re-open
    out["circuit_half_open_reopens"] = calls10["n"] == 2 and "circuit open" in str(
        _raises(lambda: cb10.health())[1]
    )
    # disabled by default: every call reaches the wire
    tr11, calls11 = _scripted(["RAISE"])
    cb11 = HarnessClient("http://h.test", transport=tr11, sleep=lambda s: None)
    for _ in range(4):
        _raises(lambda: cb11.health())
    out["circuit_disabled_by_default"] = calls11["n"] == 4

    # keepalive comments are skipped; in-band error frames map through the
    # same exception table as HTTP errors (keepalive-mode wire contract)
    ka_body = (
        b": keepalive\n\n"
        b'data: {"type":"token","content":"ka-tok"}\n\n'
        b'data: {"type":"final","model":null,"receipt_hashes":[]}\n\n'
        b"data: [DONE]\n\n"
    )
    trka, _ = _scripted([(200, {}, ka_body)])
    out["client_stream_skips_comments"] = HarnessClient(
        "http://harness.test", transport=trka
    ).stream_complete(msg, backend="byok") == ["ka-tok"]
    inband_body = (
        b": keepalive\n\n"
        b'data: {"type":"error","status":503,"detail":"no backend"}\n\n'
        b"data: [DONE]\n\n"
    )
    trin, _ = _scripted([(200, {}, inband_body)])
    out["client_stream_inband_error_maps"] = (
        _raises(
            lambda: HarnessClient("http://harness.test", transport=trin).stream_complete(
                msg, backend="byok"
            )
        )[0]
        == "BackendNotConfiguredError"
    )
    honesty_body = (
        b'data: {"type":"error","status":502,'
        b'"detail":"honesty gate refused model output: sharpe"}\n\n'
        b"data: [DONE]\n\n"
    )
    trhon, _ = _scripted([(200, {}, honesty_body)])
    out["client_stream_inband_honesty_maps"] = (
        _raises(
            lambda: HarnessClient("http://harness.test", transport=trhon).stream_complete(
                msg, backend="byok"
            )
        )[0]
        == "Fx1HonestyError"
    )

    # idempotency: run() sends Idempotency-Key, stable across retries;
    # caller-supplied keys pass through verbatim.
    sent_headers: list[dict[str, str]] = []
    run_body = _json_mod.dumps(
        {
            "command": "doctor",
            "exit_code": 0,
            "stdout": "s",
            "stderr": "",
            "ok": True,
            "timeout_s": 1,
            "replayed": True,
        }
    ).encode()

    def _cap_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        sent_headers.append(dict(headers))
        if len(sent_headers) == 1:
            raise HarnessTransportError("boom")
        return 200, {}, run_body

    c_idem = HarnessClient(
        "http://harness.test",
        transport=_cap_transport,
        max_retries=1,
        sleep=lambda _s: None,
    )
    c_idem.run("doctor")
    out["client_run_idem_stable_across_retry"] = (
        len(sent_headers) == 2
        and bool(sent_headers[0].get("Idempotency-Key"))
        and sent_headers[0]["Idempotency-Key"] == sent_headers[1]["Idempotency-Key"]
    )
    sent_headers.clear()
    c_idem.run("doctor", idempotency_key="explicit-k")
    out["client_run_idem_explicit_key"] = sent_headers[0].get("Idempotency-Key") == "explicit-k"

    # ---- async jobs ----------------------------------------------------------
    submit_body = _json_mod.dumps(
        {"job_id": "abc123", "status": "queued", "replayed": False}
    ).encode()
    status_body = _json_mod.dumps(
        {
            "job_id": "abc123",
            "status": "succeeded",
            "created_at": 1.0,
            "finished_at": 2.0,
            "result": {
                "command": "doctor",
                "exit_code": 0,
                "stdout": "s",
                "stderr": "",
                "ok": True,
                "timeout_s": 1,
                "replayed": False,
            },
            "error": None,
        }
    ).encode()
    failed_body = _json_mod.dumps(
        {
            "job_id": "abc123",
            "status": "failed",
            "created_at": 1.0,
            "finished_at": 2.0,
            "result": None,
            "error": "RuntimeError: boom",
        }
    ).encode()
    job_calls: list[tuple[str, str, dict[str, str]]] = []

    def _job_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        job_calls.append((method, url, dict(headers)))
        if url.endswith("/harness/jobs"):
            return 202, {}, submit_body
        return 200, {}, status_body

    c_jobs = HarnessClient(
        "http://harness.test",
        transport=_job_transport,
        sleep=lambda _s: None,
    )
    jid = c_jobs.submit_run("doctor", idempotency_key="jk")
    result = c_jobs.wait_run(jid, poll_s=0.01)
    out["client_submit_returns_job_id"] = jid == "abc123"
    out["client_wait_run_polls_to_result"] = result.command == "doctor" and result.ok
    out["client_job_idem_sent"] = job_calls[0][2].get("Idempotency-Key") == "jk"
    out["client_job_status_get"] = any(
        m == "GET" and "/harness/jobs/abc123" in u for m, u, _h in job_calls
    )
    job_calls.clear()

    def _fail_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        return 200, {}, failed_body

    c_fail = HarnessClient(
        "http://harness.test",
        transport=_fail_transport,
        sleep=lambda _s: None,
    )
    try:
        c_fail.wait_run("abc123", poll_s=0.01)
        out["client_wait_run_failed_raises"] = False
    except HarnessJobError:
        out["client_wait_run_failed_raises"] = True

    def _queued_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        return (
            200,
            {},
            _json_mod.dumps(
                {
                    "job_id": "abc123",
                    "status": "queued",
                    "created_at": 1.0,
                    "finished_at": None,
                    "result": None,
                    "error": None,
                }
            ).encode(),
        )

    _ticks = iter([t * 0.03 for t in range(200)])
    c_wait = HarnessClient(
        "http://harness.test",
        transport=_queued_transport,
        sleep=lambda _s: None,
        clock=lambda: next(_ticks),
    )
    try:
        c_wait.wait_run("abc123", poll_s=0.01, timeout_s=0.05)
        out["client_wait_run_timeout_raises"] = False
    except HarnessTransportError:
        out["client_wait_run_timeout_raises"] = True

    # ---- readiness + blocking drain ---------------------------------------
    ops_calls: list[tuple[str, str]] = []

    def _ops_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        ops_calls.append((method, url))
        if "/harness/version" in url:
            return (
                200,
                {"X-Fx1-Api-Version": "1"},
                b'{"api_version": "1", "fx1_version": "0.4.0"}',
            )
        if "/harness/capabilities" in url:
            return (
                200,
                {},
                b'{"api_version": "1", "fx1_version": "0.4.0", '
                b'"features": {"idempotency": true, "sse": true}, '
                b'"limits": {"max_inflight": 4.0, "job_batch_max": 64.0}, '
                b'"backends": {"byok": false}, "roles": ["evaluation"]}',
            )
        if "/ready" in url:
            return 200, {}, b'{"ready": true, "inflight": 2}'
        return (
            200,
            {},
            _json_mod.dumps({"draining": True, "inflight": 0, "drained": True}).encode(),
        )

    c_ops = HarnessClient("http://harness.test", transport=_ops_transport)
    rd = c_ops.ready()
    out["client_ready_get"] = rd == {"ready": True, "inflight": 2}
    dr = c_ops.drain(wait_s=12.5)
    out["client_drain_wait_s_sent"] = any("wait_s=12.5" in u for _m, u in ops_calls)
    out["client_drain_reports_drained"] = dr["drained"] is True
    out["client_server_version"] = c_ops.server_version() == {
        "api_version": "1",
        "fx1_version": "0.4.0",
    }
    out["client_last_api_version"] = c_ops.last_api_version == "1"
    cabcaps = c_ops.capabilities()
    out["client_capabilities_get"] = (
        cabcaps["features"]["idempotency"] is True
        and cabcaps["limits"]["job_batch_max"] == 64.0
        and cabcaps["roles"] == ["evaluation"]
    )

    # ---- wire-contract negotiation --------------------------------------
    compat = c_ops.check_compat()
    out["client_check_compat_ok"] = (
        compat["compatible"] is True
        and compat["server_api_version"] == "1"
        and compat["server_fx1_version"] == "0.4.0"
        and compat["client_api_version"] == compat["server_api_version"]
    )

    def _stale_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        return 200, {"X-Fx1-Api-Version": "99"}, b'{"api_version": "99", "fx1_version": "9.9"}'

    c_stale = HarnessClient("http://harness.test", transport=_stale_transport)
    try:
        c_stale.check_compat()
        out["client_check_compat_mismatch_raises"] = False
    except HarnessCompatError as exc:
        out["client_check_compat_mismatch_raises"] = exc.code == "incompatible_contract"
    rep_stale = c_stale.check_compat(strict=False)
    out["client_check_compat_mismatch_report"] = (
        rep_stale["compatible"] is False and rep_stale["server_api_version"] == "99"
    )
    c_old = HarnessClient(
        "http://harness.test",
        transport=lambda *a, **k: (404, {}, b'{"detail": "Not Found"}'),
    )
    rep_old = c_old.check_compat(strict=False)
    out["client_check_compat_unversioned"] = (
        rep_old["compatible"] is False and rep_old["server_api_version"] is None
    )
    try:
        c_old.check_compat()
        out["client_check_compat_unversioned_strict"] = False
    except HarnessCompatError:
        out["client_check_compat_unversioned_strict"] = True

    def _metrics_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        return (
            200,
            {},
            _json_mod.dumps(
                {
                    "uptime_s": 3.0,
                    "requests_total": 9,
                    "errors_total": 1,
                    "by_status": {"200": 8, "429": 1},
                    "inflight": 0,
                    "inflight_watermark": 2,
                    "max_inflight": 16,
                    "draining": False,
                    "rate_limited_total": 4,
                }
            ).encode(),
        )

    c_m = HarnessClient("http://harness.test", transport=_metrics_transport)
    om = c_m.metrics()
    out["client_metrics_rate_limited"] = om.rate_limited_total == 4 and om.by_status == {
        "200": 8,
        "429": 1,
    }

    # metrics_text negotiates the Prometheus view: the client must send
    # Accept: text/plain and return the raw exposition verbatim.
    seen_accept: list[str] = []

    def _prom_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        seen_accept.append(str(headers.get("Accept", "")))
        return (200, {}, b"fx1_requests_total 7\n")

    c_p = HarnessClient("http://harness.test", transport=_prom_transport)
    out["client_metrics_text"] = c_p.metrics_text() == "fx1_requests_total 7\n" and seen_accept == [
        "text/plain"
    ]

    def _coded_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        return (
            503,
            {},
            b'{"detail": "harness is draining - no new work", "code": "draining"}',
        )

    from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415
    from fx1.serve.client import HarnessJobError  # noqa: PLC0415

    coded = HarnessClient("http://harness.test", transport=_coded_transport)
    try:
        coded.ready()
        out["client_error_code_carried"] = False
    except BackendNotConfiguredError as exc:
        out["client_error_code_carried"] = exc.code == "draining"

    # ---- job lifecycle: list / cancel / wait-on-cancelled ------------------
    lifecycle_calls: list[tuple[str, str]] = []

    def _jobs_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        lifecycle_calls.append((method, url))
        if method == "DELETE":
            return (
                200,
                {},
                b'{"job_id": "j1", "status": "cancelled", "created_at": 1.0,'
                b' "finished_at": 2.0, "result": null, "error": null}',
            )
        if url.startswith("http://harness.test/harness/jobs?"):
            return (
                200,
                {},
                b'{"jobs": [{"job_id": "j1", "status": "queued", "created_at": 1.0,'
                b' "finished_at": null, "result": null, "error": null}], "total": 1}',
            )
        return (
            200,
            {},
            b'{"job_id": "j1", "status": "cancelled", "created_at": 1.0,'
            b' "finished_at": 2.0, "result": null, "error": null}',
        )

    cj = HarnessClient("http://harness.test", transport=_jobs_transport)
    page = cj.list_jobs(status="queued", limit=5, offset=10)
    out["client_list_jobs"] = page["total"] == 1 and any(
        "status=queued" in u and "limit=5" in u and "offset=10" in u for _m, u in lifecycle_calls
    )
    out["client_cancel_job"] = (
        cj.cancel_job("j1")["status"] == "cancelled"
        and lifecycle_calls[-1][0] == "DELETE"
        and "/harness/jobs/j1" in lifecycle_calls[-1][1]
    )
    try:
        cj.wait_run("j1", poll_s=0.01, timeout_s=5.0)
        out["client_wait_cancelled_raises"] = False
    except HarnessJobError:
        out["client_wait_cancelled_raises"] = True

    # submit_run passes callback_url through the request body verbatim
    cb_seen: list[dict[str, Any]] = []

    def _cb_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        cb_seen.append(dict(payload))
        return (202, {}, b'{"job_id": "cb1", "status": "queued", "replayed": false}')

    c_cb = HarnessClient("http://harness.test", transport=_cb_transport)
    jid_cb = c_cb.submit_run(
        "doctor", callback_url="https://hooks.test/x", callback_secret="whsec-t"
    )
    out["client_submit_callback_url"] = (
        jid_cb == "cb1"
        and cb_seen[0].get("callback_url") == "https://hooks.test/x"
        and cb_seen[0].get("callback_secret") == "whsec-t"
    )

    # --- job SSE stream: client parses event frames --------------------------
    sse_body = (
        b"event: job\n"
        b'data: {"job_id": "j1", "status": "queued"}\n\n'
        b": keepalive\n\n"
        b"event: job\n"
        b'data: {"job_id": "j1", "status": "succeeded", "result": '
        b'{"command": "doctor", "exit_code": 0, "stdout": "ok", "stderr": ""}}\n\n'
    )

    def _sse_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        cb_seen.append({"_url": url, "_method": method})
        return (200, {"Content-Type": "text/event-stream"}, sse_body)

    c_sse = HarnessClient("http://harness.test", transport=_sse_transport)
    job_frames = c_sse.stream_job("j1", timeout_s=42.0)
    out["client_stream_job_frames"] = (
        len(job_frames) == 2
        and job_frames[0]["status"] == "queued"
        and job_frames[-1]["status"] == "succeeded"
        and "/harness/jobs/j1/events?timeout_s=42.0" in str(cb_seen[-1].get("_url"))
    )
    out["client_wait_run_stream_result"] = (
        c_sse.wait_run_stream("j1", timeout_s=42.0).stdout == "ok"
    )
    empty_sse = HarnessClient(
        "http://harness.test",
        transport=lambda m, u, p, h, t: (200, {}, b": keepalive\n\n"),
    )
    try:
        empty_sse.stream_job("j1")
        out["client_stream_job_empty_raises"] = False
    except HarnessTransportError:
        out["client_stream_job_empty_raises"] = True

    # --- batch submit: one POST carrying the item list -----------------------
    def _batch_transport(
        method: str,
        url: str,
        payload: Any,
        headers: Any,
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        cb_seen.append({"_url": url, "_payload": payload})
        return (
            202,
            {},
            b'{"jobs": [{"index": 0, "job_id": "jb0", "status": "queued",'
            b' "replayed": false}], "submitted": 1, "failed": 0}',
        )

    c_batch = HarnessClient("http://harness.test", transport=_batch_transport)
    bout = c_batch.submit_batch([{"command": "doctor", "idempotency_key": "b1"}])
    out["client_submit_batch"] = (
        bout["submitted"] == 1
        and bout["jobs"][0]["job_id"] == "jb0"
        and cb_seen[-1]["_url"].endswith("/harness/jobs/batch")
        and cb_seen[-1]["_payload"]["jobs"][0]["idempotency_key"] == "b1"
    )

    # /v1/files + /v1/batches parity: the wire client uploads, submits,
    # polls, and fetches the output file; the SDK's openai_batch runs the
    # same lines through the same gated cores in-process — per-line
    # verdicts must agree (ids/timestamps differ by construction).
    sdk_b, client_b = _surfaces(_ParityBackend)
    c_b = HarnessClient("http://parity.local", transport=_tc_transport(client_b))
    batch_lines = [
        {
            "custom_id": "p1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "fx1", "messages": msg},
        },
        {
            "custom_id": "p2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "fx1", "temperature": 9.0, "messages": msg},
        },
        {
            "custom_id": "p3",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "fx1", "messages": msg},
        },
    ]
    up = c_b.upload_file(("\n".join(json.dumps(line) for line in batch_lines) + "\n").encode())
    out["client_upload_file"] = (
        up["object"] == "file" and up["purpose"] == "batch" and up["filename"] == "input.jsonl"
    )
    bc = c_b.create_batch(up["id"], endpoint="/v1/chat/completions")
    bt = c_b.wait_batch(bc["id"], poll_s=0.01)
    wire_out = [
        json.loads(line)
        for line in c_b.file_content(bt["output_file_id"]).decode().splitlines()
        if line.strip()
    ]
    sdk_b_out, sdk_b_lines = sdk_b.openai_batch(batch_lines)
    out["sdk_openai_batch_shape"] = (
        sdk_b_out["object"] == "batch"
        and sdk_b_out["status"] == "completed"
        and sdk_b_out["request_counts"] == {"total": 3, "completed": 2, "failed": 1}
        and len(sdk_b_lines) == 3
    )
    out["batch_wire_sdk_line_parity"] = (
        bt["request_counts"] == sdk_b_out["request_counts"]
        and [o["custom_id"] for o in wire_out] == [o["custom_id"] for o in sdk_b_lines]
        and [o["response"]["status_code"] for o in wire_out]
        == [o["response"]["status_code"] for o in sdk_b_lines]
        and wire_out[0]["response"]["body"]["choices"][0]["message"]["content"]
        == sdk_b_lines[0]["response"]["body"]["choices"][0]["message"]["content"]
        and wire_out[1]["response"]["body"]["error"]["type"]
        == sdk_b_lines[1]["response"]["body"]["error"]["type"]
    )
    out["client_batch_surface"] = (
        c_b.batch(bt["id"])["status"] == "completed"
        and any(b["id"] == bt["id"] for b in c_b.batches(limit=5)["data"])
        and c_b.batches(limit=5)["object"] == "list"
        and c_b.file(up["id"])["id"] == up["id"]
        and any(f["id"] == up["id"] for f in c_b.files())
        and c_b.delete_file(up["id"])["deleted"] is True
    )
    out["client_batch_terminal_cancel_raises"] = False
    try:
        c_b.cancel_batch(bt["id"])
    except HarnessTransportError as exc:
        out["client_batch_terminal_cancel_raises"] = exc.code == "batch_terminal" and "409" in str(
            exc
        )
    out["client_batch_bad_endpoint_422"] = False
    try:
        c_b.create_batch(up["id"], endpoint="/v1/completions")
    except ValueError:
        out["client_batch_bad_endpoint_422"] = True

    # --- /v1 retrieval parity: the store flag governs the index on every
    # surface — SDK, wire, and the typed remote client fetch/drop the same
    # envelope by id and miss identically.
    _ret_req = {
        "model": "hosted_k3",
        "messages": msg,
        "fx1": {"backend": "byok"},
    }
    _ret_sdk_env, _ = sdk.openai_chat(dict(_ret_req))
    _ret_wire = client.post("/v1/chat/completions", json=_ret_req)
    _ret_sdk_stored = sdk.openai_chat_get(_ret_sdk_env.id)
    _ret_wire_stored = client.get(f"/v1/chat/completions/{_ret_wire.json()['id']}").json()
    # each surface's index returns the envelope that surface issued, and
    # the two stored envelopes agree on everything but the minted id
    # (created ticks can differ by a second across the two calls)
    out["retrieval_chat_envelope_parity"] = _ret_wire_stored == _ret_wire.json() and {
        k: v for k, v in _ret_sdk_stored.items() if k not in ("id", "created")
    } == {k: v for k, v in _ret_wire_stored.items() if k not in ("id", "created")}
    # store=false misses on both surfaces as KeyError / 404
    _ns_sdk_env, _ = sdk.openai_chat({**_ret_req, "store": False})
    _ns_wire = client.post("/v1/chat/completions", json={**_ret_req, "store": False})
    out["retrieval_store_false_parity"] = (
        _raises(lambda: sdk.openai_chat_get(_ns_sdk_env.id))[0] == "KeyError"
        and client.get(f"/v1/chat/completions/{_ns_wire.json()['id']}").status_code == 404
    )
    # delete parity — same tombstone shape (ids differ across calls), then
    # both surfaces miss
    _del_sdk = sdk.openai_chat_delete(_ret_sdk_env.id)
    _del_wire = client.delete(f"/v1/chat/completions/{_ret_wire.json()['id']}").json()
    out["retrieval_delete_parity"] = (
        {k: v for k, v in _del_sdk.items() if k != "id"}
        == {k: v for k, v in _del_wire.items() if k != "id"}
        and _raises(lambda: sdk.openai_chat_get(_ret_sdk_env.id))[0] == "KeyError"
        and client.get(f"/v1/chat/completions/{_ret_wire.json()['id']}").status_code == 404
    )
    # the typed remote client rides the same routes
    _rr_wire = client.post(
        "/v1/responses", json={"model": "fx1", "input": "hi", "fx1": {"backend": "byok"}}
    )
    out["client_retrieval_surface"] = (
        remote.retrieve_response(_rr_wire.json()["id"]) == _rr_wire.json()
        and remote.delete_response(_rr_wire.json()["id"])["deleted"] is True
    )
    out["client_retrieval_miss_404"] = (
        _raises(lambda: remote.retrieve_chat_completion("chatcmpl-miss"))[0] == "KeyError"
    )
    return out


def parity_audit_bench() -> dict[str, Any]:
    r = parity_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "parity_audit",
        "schema": "parity_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "three-surface parity holds: SDK, HTTP API, and HarnessClient "
            "return byte-identical content, envelopes, chunk lists, error "
            "classes, verifier verdicts, registry, and health over the same "
            "injected backend; the in-flight cap fails saturated work with "
            "503+Retry-After while cheap routes respond, and releases "
            "cleanly. Both surfaces keep a per-call completion log whose "
            "records share prompt/output sha256s (the completion_id itself "
            "is a per-surface mint — not part of the parity claim). Each "
            "surface also exports its own logged call as a sealed "
            "fx1_completion_record.v1 receipt — same record hashes, each "
            "doc verifiable through either surface's verifier. A terminal "
            "wire job's fx1_job_record.v1 embeds the same digested run "
            "result that Fx1Harness.run_receipt seals in-process as "
            "fx1_run_result.v1 — identical on the shared fields. "
            "Fine-tuning parity holds: the SDK's synchronous "
            "create_finetune_job produces the same terminal job fields "
            "(status, ft:-name grammar, result_files, event feed) as the "
            "wire route under the same runner, and guards map "
            "ValueError↔HarnessTransportError / KeyError↔KeyError. Flags: "
            "unknown backend names are KeyError in-process "
            "vs 422 literal rejection over the wire (request validation "
            "runs before resolution); empty batches are [] in-process vs "
            "422 over the wire."
            if ok
            else f"PARITY AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
